# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import time
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import threading

from tqdm import tqdm
from rich import print as rprint
from rich.pretty import pprint
from rich.panel import Panel
from rich.console import Console

from lighteval.data import GenerativeTaskDataset
from lighteval.models.abstract_model import LightevalModel
from lighteval.models.endpoints.endpoint_model import ModelInfo
from lighteval.models.model_output import (
    GenerativeResponse,
    LoglikelihoodResponse,
    LoglikelihoodSingleTokenResponse,
)
from lighteval.models.utils import ModelConfig
from lighteval.tasks.requests import (
    GreedyUntilRequest,
    LoglikelihoodRequest,
    LoglikelihoodRollingRequest,
    LoglikelihoodSingleTokenRequest,
)
from lighteval.utils.imports import is_litellm_available


logger = logging.getLogger(__name__)
console = Console()

if is_litellm_available():
    import litellm
    from litellm import encode
    from litellm.caching.caching import Cache, disable_cache
    from litellm.utils import ModelResponse

    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").handlers.clear()

    litellm.cache = Cache(type="disk")
    litellm.disable_cache()


class LiteLLMModelConfig(ModelConfig):
    model_name: str
    provider: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    split_n_size: int = 32


class LiteLLMClient(LightevalModel):
    _DEFAULT_MAX_LENGTH: int = 4096

    def __init__(self, config) -> None:
        """
        IMPORTANT: Your API keys should be set in the environment variables.
        If a base_url is not set, it will default to the public API.
        """
        self.model_info = ModelInfo(
            model_name=config.model_name,
            model_sha="",
            model_dtype=None,
            model_size="",
        )
        self.model = config.model_name
        self.provider = config.provider or config.model_name.split("/")[0]
        self.base_url = config.base_url
        self.api_key = config.api_key
        self.generation_parameters = config.generation_parameters
        self.split_n_size = config.split_n_size

        self.API_MAX_RETRY = 5
        self.API_RETRY_SLEEP = 3
        self.API_RETRY_MULTIPLIER = 2
        self.CONCURENT_CALLS = 100  # 100 leads to hitting Anthropic rate limits

        self._tokenizer = encode
        self.pairwise_tokenization = False
        litellm.drop_params = True
        litellm.set_verbose = False

        # Initialize throttle lock and timestamp for rate limiting
        self._throttle_lock = threading.Lock()
        self._last_request_time = 0.0

    def _prepare_stop_sequence(self, stop_sequence):
        """Prepare and validate stop sequence."""
        if self.provider == "anthropic":
            # Filter out whitespace-only stop sequences
            if stop_sequence:
                stop_sequence = [s for s in stop_sequence if s and s.strip()]
        return stop_sequence

    def _prepare_max_new_tokens(self, max_new_tokens):
        """Calculate completion tokens based on max_new_tokens."""
        if not max_new_tokens or max_new_tokens <= 0:
            return None

        if any(model_prefix in self.model for model_prefix in ["o1", "o3", "o4"]):
            # We need to allow more tokens to include reasoning tokens
            max_new_tokens = min(max_new_tokens * 10, 32000)
        return max_new_tokens

    def __call_api(self, prompt, return_logits, max_new_tokens, num_samples, stop_sequence, metadata=None):
        """Make API call with retries."""
        # If requested, split num_samples into chunks of size split_n_size
        if self.split_n_size and num_samples and num_samples > self.split_n_size:
            chunk_size = self.split_n_size
            full_chunks = num_samples // chunk_size
            remainder = num_samples % chunk_size
            sizes = [chunk_size] * full_chunks + ([remainder] if remainder else [])
            aggregated_choices = []
            max_workers = min(len(sizes), self.CONCURENT_CALLS)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        self.__call_api, prompt, return_logits, max_new_tokens, sz, stop_sequence, metadata
                    )
                    for sz in sizes
                ]
                for future in futures:
                    resp = future.result()
                    aggregated_choices.extend(resp.choices)
            agg = ModelResponse()
            agg.choices = aggregated_choices
            return agg
        response = ModelResponse()
        for attempt in range(self.API_MAX_RETRY):
            try:
                stop_sequence = self._prepare_stop_sequence(stop_sequence)
                original_max_new_tokens = max_new_tokens
                max_new_tokens = self._prepare_max_new_tokens(max_new_tokens)

                if return_logits and not self.provider == "openai":
                    logger.warning("Returning logits is not supported for this provider, ignoring.")

                # Prepare kwargs for completion call
                kwargs = {
                    "model": self.model,
                    "messages": prompt,
                    "logprobs": return_logits if self.provider == "openai" else None,
                    "base_url": self.base_url,
                    "n": num_samples,
                    "caching": False,
                    "cache": {"no-cache": True},
                    "api_key": self.api_key,
                    "request_timeout": 900,  # 15 minutes timeout
                }
                if any(model_prefix in self.model for model_prefix in ["o1", "o3", "o4"]):
                    logger.warning("OpenAI o-series models do not support temperature, top_p, stop sequence. Disabling.")
                else:
                    # Update with generation parameters
                    litellm_params = self.generation_parameters.to_litellm_dict()
                    
                    # If max_new_tokens is specified in generation parameters, handle it differently
                    # Rename max_new_tokens to max_tokens for non-o-series models
                    if "max_completion_tokens" in litellm_params and not any(model_prefix in self.model for model_prefix in ["o1", "o3", "o4"]):
                        litellm_params["max_tokens"] = litellm_params.pop("max_completion_tokens")
                    
                    kwargs.update(litellm_params)

                # Use max_tokens instead of max_completion_tokens for all models except those starting with "o1", "o3", or "o4"
                # Only set these if they aren't already set from generation_parameters
                if not kwargs.get("max_completion_tokens") and not kwargs.get("max_tokens"):
                    if any(model_prefix in self.model for model_prefix in ["o1", "o3", "o4"]):
                        kwargs["max_completion_tokens"] = max_new_tokens
                    else:
                        kwargs["max_tokens"] = max_new_tokens
                
                # Rich logging of the request details
                request_info = {
                    "model": self.model,
                    "provider": self.provider,
                    "original_max_new_tokens": original_max_new_tokens,
                    "adjusted_max_new_tokens": max_new_tokens,
                    "max_completion_tokens": kwargs.get("max_completion_tokens"),
                    "max_tokens": kwargs.get("max_tokens"),
                    "is_o_series_model": any(model_prefix in self.model for model_prefix in ["o1", "o3", "o4"]),
                    "token_param_used": "max_completion_tokens" if any(model_prefix in self.model for model_prefix in ["o1", "o3", "o4"]) else "max_tokens",
                    "num_samples": num_samples,
                    "message_count": len(prompt) if isinstance(prompt, list) else 0,
                    "generation_parameters": {k: v for k, v in kwargs.items() 
                                            if k not in ["model", "messages", "api_key", "base_url", "n", "caching"]}
                }
                
                # Prepare a base meta string for titles if metadata provided
                if metadata:
                    try:
                        idx = int(metadata.get("index", 1))
                        total = int(metadata.get("total", 0))
                    except Exception:
                        idx = metadata.get("index", 1)
                        total = metadata.get("total", 0)
                    base_meta = f"{metadata.get('benchmark', '')} {idx}/{total}"
                else:
                    base_meta = None
                # Print the request in a panel with rich formatting
                if base_meta:
                    request_title = f"LiteLLM API Request — {base_meta} — attempt {attempt+1}/{self.API_MAX_RETRY}"
                else:
                    request_title = "LiteLLM API Request"
                console.print(Panel.fit(
                    "\n".join([
                        "[bold cyan]LiteLLM Request Details:[/bold cyan]",
                        f"[green]Model:[/green] {request_info['model']}",
                        f"[green]Provider:[/green] {request_info['provider']}",
                        f"[green]O-series Model:[/green] {request_info['is_o_series_model']}",
                        f"[green]Token Parameter Used:[/green] {request_info['token_param_used']}",
                        f"[green]Original max_new_tokens:[/green] {request_info['original_max_new_tokens']}",
                        f"[green]Adjusted max_new_tokens:[/green] {request_info['adjusted_max_new_tokens']}",
                        f"[green]max_completion_tokens:[/green] {request_info['max_completion_tokens']}",
                        f"[green]max_tokens:[/green] {request_info['max_tokens']}",
                        f"[green]num_samples:[/green] {request_info['num_samples']}",
                        f"[green]message_count:[/green] {request_info['message_count']}",
                        "[green]Generation Parameters:[/green]"
                    ]),
                    title=request_title,
                    border_style="blue"
                ))
                pprint(request_info["generation_parameters"])
                
                # Throttle to ensure no two completion calls occur within 50ms
                with self._throttle_lock:
                    now = time.time()
                    elapsed = now - self._last_request_time
                    wait_time = 0.05 - elapsed
                    if wait_time > 0:
                        time.sleep(wait_time)
                    self._last_request_time = time.time()
                response = litellm.completion(**kwargs)

                # If response is empty, retry without caching (maybe the error is recoverable and solved with a retry)
                if response.choices[0].message.content is None:
                    kwargs["caching"] = False
                    logger.info("Response is empty, retrying without caching")
                    response = litellm.completion(**kwargs)
                
                # Log response details
                try:
                    # Use base_meta to build response title
                    if base_meta:
                        response_title = f"LiteLLM API Response — {base_meta} — attempt {attempt+1}/{self.API_MAX_RETRY}"
                    else:
                        response_title = "LiteLLM API Response"
                    console.print(Panel.fit(
                        "\n".join([
                            "[bold cyan]LiteLLM Response Summary:[/bold cyan]",
                            f"[green]Completion ID:[/green] {response.id}",
                            f"[green]Model:[/green] {response.model}",
                            f"[green]Created at:[/green] {response.created}",
                            f"[green]Number of choices:[/green] {len(response.choices)}",
                            f"[green]Content length:[/green] {len(response.choices[0].message.content or '') if response.choices else 0} chars",
                            f"[green]Usage - Prompt tokens:[/green] {response.usage.prompt_tokens if response.usage else 'N/A'}",
                            f"[green]Usage - Completion tokens:[/green] {response.usage.completion_tokens if response.usage else 'N/A'}",
                            f"[green]Usage - Total tokens:[/green] {response.usage.total_tokens if response.usage else 'N/A'}"
                        ]),
                        title=response_title,
                        border_style="green"
                    ))
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not print full response details: {e}[/yellow]")
                    
                return response
            except litellm.BadRequestError as e:
                if "message" in e.__dict__:
                    error_string = (
                        "The response was filtered due to the prompt triggering Microsoft's content management policy"
                    )
                    if error_string in e.__dict__["message"]:
                        logger.warning(f"{error_string}. Returning empty response.")
                        return ModelResponse()
                
                # Use the same base_meta to build error title
                if base_meta:
                    error_title = f"LiteLLM API Error — {base_meta} — attempt {attempt+1}/{self.API_MAX_RETRY}"
                else:
                    error_title = "LiteLLM API Error"
                console.print(Panel.fit(
                    f"[bold red]Error in API Call (attempt {attempt + 1}/{self.API_MAX_RETRY}):[/bold red]\n{str(e)}",
                    title=error_title,
                    border_style="red"
                ))
            except Exception as e:
                wait_time = min(64, self.API_RETRY_SLEEP * (2**attempt))  # Exponential backoff with max 64s
                
                # Use the same base_meta to build error title for retry
                if base_meta:
                    error_title = f"LiteLLM API Error — {base_meta} — attempt {attempt+1}/{self.API_MAX_RETRY}"
                else:
                    error_title = "LiteLLM API Error"
                console.print(Panel.fit(
                    f"[bold red]Error in API Call (attempt {attempt + 1}/{self.API_MAX_RETRY}):[/bold red]\n{str(e)}\n\n" +
                    f"[yellow]Waiting {wait_time} seconds before retry...[/yellow]",
                    title=error_title,
                    border_style="red"
                ))
                
                logger.warning(
                    f"Error in API call: {e}, waiting {wait_time} seconds before retry {attempt + 1}/{self.API_MAX_RETRY}"
                )
                time.sleep(wait_time)

        logger.error(f"API call failed after {self.API_MAX_RETRY} attempts, returning empty response.")
        # Use base_meta to build final failure title
        if base_meta:
            failure_title = f"LiteLLM API Failure — {base_meta} — attempts {self.API_MAX_RETRY}"
        else:
            failure_title = "LiteLLM API Failure"
        console.print(Panel.fit(
            f"[bold red]API call failed after {self.API_MAX_RETRY} attempts[/bold red]\nReturning empty response.",
            title=failure_title,
            border_style="red"
        ))
        return ModelResponse()

    def __call_api_parallel(
        self,
        prompts,
        return_logits: bool | list[bool],
        max_new_tokens: int | list[int],
        num_samples: int | list[int],
        stop_sequence: list[str] | None = None,
        metadata: list[dict] | None = None,
    ):
        results = []

        return_logitss = [return_logits for _ in prompts] if not isinstance(return_logits, list) else return_logits
        max_new_tokenss = [max_new_tokens for _ in prompts] if not isinstance(max_new_tokens, list) else max_new_tokens
        num_sampless = [num_samples for _ in prompts] if not isinstance(num_samples, list) else num_samples
        stop_sequencess = [stop_sequence for _ in prompts]
        assert (
            len(prompts) == len(return_logitss) == len(max_new_tokenss) == len(num_sampless) == len(stop_sequencess)
        ), f"Length of prompts, return_logitss, max_new_tokenss, num_sampless, stop_sequences, system_prompts should be the same but are {len(prompts)}, {len(return_logitss)}, {len(max_new_tokenss)}, {len(num_sampless)}, {len(stop_sequencess)}"

        # Align metadata list with prompts
        metadata_list = metadata if metadata is not None else [None] * len(prompts)
        assert len(prompts) == len(metadata_list), f"Length of prompts ({len(prompts)}) and metadata ({len(metadata_list)}) must be the same"

        with ThreadPoolExecutor(self.CONCURENT_CALLS) as executor:
            for entry in tqdm(
                executor.map(
                    self.__call_api,
                    prompts,
                    return_logitss,
                    max_new_tokenss,
                    num_sampless,
                    stop_sequencess,
                    metadata_list,
                ),
                total=len(prompts),
            ):
                results.append(entry)

        if None in results:
            raise ValueError("Some entries are not annotated due to errors in annotate_p, please inspect and retry.")

        return results

    def greedy_until(
        self,
        requests: list[GreedyUntilRequest],
        override_bs: Optional[int] = None,
    ) -> list[GenerativeResponse]:
        """
        Generates responses using a greedy decoding strategy until certain ending conditions are met.

        Args:
            requests (list[Request]): list of requests containing the context and ending conditions.
            override_bs (int, optional): Override the batch size for generation. Defaults to None.

        Returns:
            list[GenerativeResponse]: list of generated responses.
        """
        for request in requests:
            request.tokenized_context = self.tok_encode(request.context)

        dataset = GenerativeTaskDataset(requests=requests, num_dataset_splits=self.DATASET_SPLITS)
        total_requests = dataset.total_size
        results = []

        for split in tqdm(
            dataset.splits_iterator(),
            total=dataset.num_dataset_splits,
            desc="Splits",
            position=0,
            disable=False,
        ):
            contexts = [sample.context for sample in split]
            max_new_tokens = split[0].generation_size  # could be none
            return_logits = split[0].use_logits
            num_samples = split[0].num_samples
            stop_sequence = requests[0].stop_sequence

            # Build metadata for this batch
            metadata_list = []
            for i, sample in enumerate(split):
                # Compute index within dataset from the subset indices
                try:
                    ordinal = split.indices[i]
                except Exception:
                    ordinal = i
                metadata_list.append({
                    "benchmark": sample.task_name,
                    "index": ordinal + 1,
                    "total": total_requests,
                })

            responses = self.__call_api_parallel(contexts, return_logits, max_new_tokens, num_samples, stop_sequence, metadata_list)

            for response in responses:
                result: list[str] = [choice.message.content for choice in response.choices]

                cur_response = GenerativeResponse(
                    # In empty responses, the model should return an empty string instead of None
                    result=result if result[0] else [""],
                    logits=None,
                    generated_tokens=[],
                    input_tokens=[],
                )
                results.append(cur_response)

        return dataset.get_original_order(results)

    @property
    def tokenizer(self):
        return self._tokenizer

    def _encode(self, text: str):
        enc = encode(model=self.model, text=text)
        if hasattr(enc, "ids"):
            return enc.ids
        return enc

    def tok_encode(self, text: str | list[str]):
        if isinstance(text, list):
            toks = [self._encode(t["content"]) for t in text]
            toks = [tok for tok in toks if tok]
            return toks
        return self._encode(text)

    @property
    def add_special_tokens(self) -> bool:
        return False

    @property
    def max_length(self) -> int:
        """Return the maximum sequence length of the model."""
        return 4096

    def loglikelihood(
        self, requests: list[LoglikelihoodRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodResponse]:
        """Tokenize the context and continuation and compute the log likelihood of those
        tokenized sequences.
        """
        raise NotImplementedError

    def loglikelihood_rolling(
        self, requests: list[LoglikelihoodRollingRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodResponse]:
        """This function is used to compute the log likelihood of the context for perplexity metrics."""
        raise NotImplementedError

    def loglikelihood_single_token(
        self, requests: list[LoglikelihoodSingleTokenRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodSingleTokenResponse]:
        """Tokenize the context and continuation and compute the log likelihood of those
        tokenized sequences.
        """
        raise NotImplementedError
