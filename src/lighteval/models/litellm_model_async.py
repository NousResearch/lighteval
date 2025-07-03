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

import asyncio
import logging
import json
import time
from collections import deque
from typing import Optional

from rich import print as rprint
from rich.pretty import pprint
from rich.panel import Panel
from rich.console import Console

from lighteval.data import GenerativeTaskDataset
from lighteval.models.abstract_model import LightevalModel
from lighteval.models.endpoints.endpoint_model import ModelInfo
from lighteval.models.model_output import ModelResponse
from lighteval.models.utils import ModelConfig
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.tasks.requests import Doc
from lighteval.utils.imports import is_litellm_available


logger = logging.getLogger(__name__)
console = Console()

if is_litellm_available():
    import litellm
    from litellm import encode, acompletion
    from litellm.caching.caching import Cache, disable_cache
    from litellm.utils import ModelResponse as LitellmModelResponse

    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").handlers.clear()

    litellm.cache = Cache(type="disk")
    litellm.disable_cache()
    litellm._turn_on_debug()  # Enable detailed error debugging
else:
    from unittest.mock import Mock

    litellm = Mock()
    encode = Mock()
    acompletion = Mock()
    LitellmModelResponse = Mock()

try:
    from tqdm.asyncio import tqdm as async_tqdm
except ImportError:
    from tqdm import tqdm as async_tqdm


class AsyncLiteLLMModelConfig(ModelConfig):
    """
    Configuration class for async LiteLLM unified API client.

    This configuration is used to connect to various LLM providers through the LiteLLM
    unified API. LiteLLM provides a consistent interface to multiple providers including
    OpenAI, Anthropic, Google, and many others.

    litellm doc: https://docs.litellm.ai/docs/

    Attributes:
        model_name (str):
            Model identifier. Can include provider prefix (e.g., "gpt-4", "claude-3-sonnet")
            or use provider/model format (e.g., "openai/gpt-4", "anthropic/claude-3-sonnet").
        provider (str | None):
            Optional provider name override. If None, inferred from model_name.
            Examples: "openai", "anthropic", "google", "cohere", etc.
        base_url (str | None):
            Custom base URL for the API. If None, uses provider's default URL.
            Useful for using custom endpoints or local deployments.
        api_key (str | None):
            API key for authentication. If None, reads from environment variables.
            Environment variable names are provider-specific (e.g., OPENAI_API_KEY).
        parallel_calls_count (int):
            Maximum number of concurrent API calls. Defaults to 50.

    Example:
        ```python
        config = AsyncLiteLLMModelConfig(
            model_name="gpt-4",
            provider="openai",
            base_url="https://api.openai.com/v1",
            parallel_calls_count=20,
            generation_parameters=GenerationParameters(
                temperature=0.7,
                max_new_tokens=100
            )
        )
        ```
    """

    model_name: str
    provider: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    split_n_size: int = 8
    parallel_calls_count: int = 32
    num_workers: int = 1  # Number of workers for concurrency ramping
    master_setup_time: float | None = None  # Master setup time in seconds for ramping calculation; if None, will be measured
    max_rpm: int = 256  # Maximum requests per minute for rate limiting


class AsyncLiteLLMClient(LightevalModel):
    _DEFAULT_MAX_LENGTH: int = 4096
    DATASET_SPLITS = 1  # API-based models don't need dataset splitting like local models
    is_async = True  # Enable async mode

    def __init__(self, config) -> None:
        """
        IMPORTANT: Your API keys should be set in the environment variables.
        If a base_url is not set, it will default to the public API.
        """
        self.model_info = ModelInfo(
            model_name=config.model_name,
            model_sha="",
            model_dtype=None,
            model_size=-1,
        )
        self.model = config.model_name
        self.provider = config.provider or config.model_name.split("/")[0]
        self.base_url = config.base_url
        self.api_key = config.api_key
        self.generation_parameters = config.generation_parameters
        self.split_n_size = config.split_n_size
        self.parallel_calls_count = config.parallel_calls_count
        self.num_workers = config.num_workers
        self.master_setup_time = config.master_setup_time
        self.max_rpm = config.max_rpm

        self.API_MAX_RETRY = 5
        self.API_RETRY_SLEEP = 60  # Start with 60 seconds (1 minute) for server overload issues
        self.API_RETRY_MULTIPLIER = 2
        
        # Initialize concurrency ramping
        self.start_time = time.time()
        self.initial_concurrency = max(1, self.parallel_calls_count // self.num_workers)
        self.master_setup_time = config.master_setup_time
        self.measured_setup_time = None
        self.setup_measurement_start = self.start_time
        self.ramp_up_time = None  # Will be set once we have master_setup_time
        
        # Create semaphore for concurrency control - start with initial concurrency
        self.semaphore = asyncio.Semaphore(self.initial_concurrency)
        self._current_concurrency = self.initial_concurrency
        self._ramping_task = None
        
        # Rate limiting setup
        self.request_timestamps = deque()  # Track request timestamps for rate limiting
        self.rate_limit_lock = asyncio.Lock()  # Lock for thread-safe rate limiting
        
        # Log ramping configuration
        if self.master_setup_time is not None:
            self.ramp_up_time = self.master_setup_time * 2.2
            console.print(Panel.fit(
                f"[bold cyan]Async LiteLLM Concurrency Ramping Configuration:[/bold cyan]\n"
                f"[green]Number of workers:[/green] {self.num_workers}\n"
                f"[green]Master setup time:[/green] {self.master_setup_time:.1f}s (user-provided)\n"
                f"[green]Ramp-up time:[/green] {self.ramp_up_time:.1f}s\n"
                f"[green]Initial concurrency:[/green] {self.initial_concurrency}\n"
                f"[green]Full concurrency:[/green] {self.parallel_calls_count}\n"
                f"[green]Max RPM:[/green] {self.max_rpm}\n"
                f"[green]Concurrency scaling:[/green] {self.initial_concurrency} → {self.parallel_calls_count}",
                title="Concurrency Ramping Setup",
                border_style="blue"
            ))
        else:
            console.print(Panel.fit(
                f"[bold cyan]Async LiteLLM Concurrency Ramping Configuration:[/bold cyan]\n"
                f"[green]Number of workers:[/green] {self.num_workers}\n"
                f"[green]Master setup time:[/green] [yellow]Will be measured automatically[/yellow]\n"
                f"[green]Initial concurrency:[/green] {self.initial_concurrency}\n"
                f"[green]Full concurrency:[/green] {self.parallel_calls_count}\n"
                f"[green]Max RPM:[/green] {self.max_rpm}\n"
                f"[green]Concurrency scaling:[/green] {self.initial_concurrency} → {self.parallel_calls_count}",
                title="Concurrency Ramping Setup",
                border_style="blue"
            ))

        self._tokenizer = encode
        self.pairwise_tokenization = False
        litellm.drop_params = True
        litellm.set_verbose = False
        self.prompt_manager = PromptManager(
            use_chat_template=True, tokenizer=self.tokenizer, system_prompt=config.system_prompt
        )

    def _measure_setup_time_if_needed(self):
        """Measure master setup time if it hasn't been set yet."""
        if self.master_setup_time is None and self.measured_setup_time is None:
            # This is the first time we're called, measure the setup time
            current_time = time.time()
            self.measured_setup_time = current_time - self.setup_measurement_start
            self.master_setup_time = self.measured_setup_time
            self.ramp_up_time = self.master_setup_time * 2.2
            
            # Log the measured setup time
            console.print(Panel.fit(
                f"[bold cyan]Master Setup Time Measured:[/bold cyan]\n"
                f"[green]Measured setup time:[/green] {self.measured_setup_time:.1f}s\n"
                f"[green]Calculated ramp-up time:[/green] {self.ramp_up_time:.1f}s (2.2x setup time)\n"
                f"[green]Concurrency will scale from:[/green] {self.initial_concurrency} → {self.parallel_calls_count}",
                title="Setup Time Measurement",
                border_style="yellow"
            ))

    def _get_current_concurrency(self) -> int:
        """Get the current concurrency limit based on elapsed time and ramping schedule."""
        # Measure setup time if needed
        self._measure_setup_time_if_needed()
        
        # If we still don't have a ramp_up_time, stay at initial concurrency
        if self.ramp_up_time is None:
            return self.initial_concurrency
            
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time < self.ramp_up_time:
            # Still in initial phase, use initial concurrency
            return self.initial_concurrency
        else:
            # Ramp-up time has passed, use full concurrency
            return self.parallel_calls_count

    async def _update_semaphore_if_needed(self):
        """Update semaphore capacity if the current concurrency has changed."""
        target_concurrency = self._get_current_concurrency()
        
        if target_concurrency != self._current_concurrency:
            # Log the concurrency change
            ramp_up_status = f"{self.ramp_up_time:.1f}s" if self.ramp_up_time is not None else "Not yet determined"
            console.print(Panel.fit(
                f"[bold cyan]Concurrency Ramping:[/bold cyan]\n"
                f"[green]Elapsed time:[/green] {time.time() - self.start_time:.1f}s\n"
                f"[green]Ramping from:[/green] {self._current_concurrency} → {target_concurrency}\n"
                f"[green]Ramp-up time:[/green] {ramp_up_status}\n"
                f"[green]Initial concurrency:[/green] {self.initial_concurrency}\n"
                f"[green]Full concurrency:[/green] {self.parallel_calls_count}",
                title="Concurrency Scaling",
                border_style="magenta"
            ))
            
            # Create new semaphore with updated capacity
            old_semaphore = self.semaphore
            self.semaphore = asyncio.Semaphore(target_concurrency)
            self._current_concurrency = target_concurrency
            
            # Note: We don't need to migrate waiters since this typically happens
            # at the start of new batches, not during active API calls

    async def _wait_for_rate_limit(self):
        """Wait for rate limiting based on max_rpm to ensure requests are spread out."""
        async with self.rate_limit_lock:
            current_time = time.time()
            
            # Remove timestamps older than 60 seconds
            while self.request_timestamps and current_time - self.request_timestamps[0] > 60:
                self.request_timestamps.popleft()
            
            # If we've hit the rate limit, wait
            if len(self.request_timestamps) >= self.max_rpm:
                # Wait until the oldest request is more than 60 seconds old
                wait_time = 60 - (current_time - self.request_timestamps[0])
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    # Clean up old timestamps again after waiting
                    current_time = time.time()
                    while self.request_timestamps and current_time - self.request_timestamps[0] > 60:
                        self.request_timestamps.popleft()
            
            # Add current request timestamp
            self.request_timestamps.append(current_time)
            
            # Calculate inter-request delay to spread out requests evenly
            inter_request_delay = 60.0 / self.max_rpm
            
            # If we have recent requests, ensure minimum spacing
            if len(self.request_timestamps) > 1:
                time_since_last = current_time - self.request_timestamps[-2]
                if time_since_last < inter_request_delay:
                    additional_wait = inter_request_delay - time_since_last
                    await asyncio.sleep(additional_wait)
                    # Update the timestamp to reflect the actual send time
                    self.request_timestamps[-1] = time.time()

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

    def _format_response_with_reasoning(self, message):
        """Format response content with reasoning in <think> tags."""
        content = message.content or ""
        reasoning = getattr(message, 'reasoning_content', None)
        
        if reasoning:
            return f"<think>\n{reasoning}\n</think>\n\n{content}"
        else:
            return content

    async def __call_api(self, prompt, return_logits, max_new_tokens, num_samples, stop_sequence, metadata=None):
        """Make async API call with splitting logic."""
        # If requested, split num_samples into chunks of size split_n_size
        if self.split_n_size and num_samples and num_samples > self.split_n_size:
            chunk_size = self.split_n_size
            full_chunks = num_samples // chunk_size
            remainder = num_samples % chunk_size
            sizes = [chunk_size] * full_chunks + ([remainder] if remainder else [])
            
            # Create async tasks for each chunk - each chunk will be subject to semaphore
            tasks = [
                self.__call_api_single(prompt, return_logits, max_new_tokens, sz, stop_sequence, metadata)
                for sz in sizes
            ]
            
            # Wait for all chunks to complete
            responses = await asyncio.gather(*tasks)
            
            # Aggregate choices from all responses
            aggregated_choices = []
            for resp in responses:
                aggregated_choices.extend(resp.choices)
            
            # Create aggregated response using the same structure as LiteLLM
            from types import SimpleNamespace
            agg_response = SimpleNamespace()
            agg_response.choices = aggregated_choices
            # Use first response for metadata (id, model, created, usage)
            if responses:
                first_resp = responses[0]
                agg_response.id = getattr(first_resp, 'id', 'aggregated')
                agg_response.model = getattr(first_resp, 'model', self.model)
                agg_response.created = getattr(first_resp, 'created', None)
                # Sum up usage across all responses
                total_prompt_tokens = sum(getattr(resp, 'usage', SimpleNamespace()).prompt_tokens or 0 for resp in responses)
                total_completion_tokens = sum(getattr(resp, 'usage', SimpleNamespace()).completion_tokens or 0 for resp in responses)
                agg_usage = SimpleNamespace()
                agg_usage.prompt_tokens = total_prompt_tokens
                agg_usage.completion_tokens = total_completion_tokens
                agg_usage.total_tokens = total_prompt_tokens + total_completion_tokens
                agg_response.usage = agg_usage
            return agg_response

        # Single request - delegate to semaphore-controlled method
        return await self.__call_api_single(prompt, return_logits, max_new_tokens, num_samples, stop_sequence, metadata)

    async def __call_api_single(self, prompt, return_logits, max_new_tokens, num_samples, stop_sequence, metadata=None):
        """Make single async API call with retries."""
        # Apply rate limiting before making the request
        await self._wait_for_rate_limit()
        
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
                    "request_timeout": 3600,  # 15 minutes timeout
                }
                if num_samples > 1 and self.generation_parameters.temperature == 0:
                    raise ValueError(
                        "num_samples > 1 but temperature is set to 0, this will not sample different outputs."
                    )

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
                    "max_rpm": self.max_rpm,
                    "inter_request_delay": f"{60.0 / self.max_rpm:.2f}s",
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
                    request_title = f"Async LiteLLM API Request — {base_meta} — attempt {attempt+1}/{self.API_MAX_RETRY}"
                else:
                    request_title = "Async LiteLLM API Request"
                console.print(Panel.fit(
                    "\n".join([
                        "[bold cyan]Async LiteLLM Request Details:[/bold cyan]",
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
                        f"[green]Max RPM:[/green] {request_info['max_rpm']}",
                        f"[green]Inter-request delay:[/green] {request_info['inter_request_delay']}",
                        "[green]Generation Parameters:[/green]"
                    ]),
                    title=request_title,
                    border_style="blue"
                ))
                pprint(request_info["generation_parameters"])
                
                # Make async API call using acompletion
                response = await acompletion(**kwargs)

                # If response is empty, retry without caching (maybe the error is recoverable and solved with a retry)
                if response.choices[0].message.content is None:
                    kwargs["caching"] = False
                    logger.info("Response is empty, retrying without caching")
                    response = await acompletion(**kwargs)
                
                # Log response details
                try:
                    # Use base_meta to build response title
                    if base_meta:
                        response_title = f"Async LiteLLM API Response — {base_meta} — attempt {attempt+1}/{self.API_MAX_RETRY}"
                    else:
                        response_title = "Async LiteLLM API Response"
                    console.print(Panel.fit(
                        "\n".join([
                            "[bold cyan]Async LiteLLM Response Summary:[/bold cyan]",
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
                    error_title = f"Async LiteLLM API Error — {base_meta} — attempt {attempt+1}/{self.API_MAX_RETRY}"
                else:
                    error_title = "Async LiteLLM API Error"
                console.print(Panel.fit(
                    f"[bold red]Error in async API Call (attempt {attempt + 1}/{self.API_MAX_RETRY}):[/bold red]\n{str(e)}",
                    title=error_title,
                    border_style="red"
                ))
            except Exception as e:
                wait_time = min(600, self.API_RETRY_SLEEP * (2**attempt))  # Exponential backoff with max 10 minutes
                
                # Use the same base_meta to build error title for retry
                if base_meta:
                    error_title = f"Async LiteLLM API Error — {base_meta} — attempt {attempt+1}/{self.API_MAX_RETRY}"
                else:
                    error_title = "Async LiteLLM API Error"
                console.print(Panel.fit(
                    f"[bold red]Error in async API Call (attempt {attempt + 1}/{self.API_MAX_RETRY}):[/bold red]\n{str(e)}\n\n" +
                    f"[yellow]Waiting {wait_time} seconds before retry...[/yellow]",
                    title=error_title,
                    border_style="red"
                ))
                
                logger.warning(
                    f"Error in async API call: {e}, waiting {wait_time} seconds before retry {attempt + 1}/{self.API_MAX_RETRY}"
                )
                await asyncio.sleep(wait_time)

        logger.error(f"Async API call failed after {self.API_MAX_RETRY} attempts, returning empty response.")
        # Use base_meta to build final failure title
        if base_meta:
            failure_title = f"Async LiteLLM API Failure — {base_meta} — attempts {self.API_MAX_RETRY}"
        else:
            failure_title = "Async LiteLLM API Failure"
        console.print(Panel.fit(
            f"[bold red]Async API call failed after {self.API_MAX_RETRY} attempts[/bold red]\nReturning empty response.",
            title=failure_title,
            border_style="red"
        ))
        
        # Create a mock LitellmModelResponse with empty content
        from types import SimpleNamespace
        
        # Create a mock response that matches LiteLLM's structure
        empty_choice = SimpleNamespace()
        empty_choice.message = SimpleNamespace()
        empty_choice.message.content = ""
        
        empty_usage = SimpleNamespace()
        empty_usage.prompt_tokens = 0
        empty_usage.completion_tokens = 0
        
        mock_response = SimpleNamespace()
        mock_response.choices = [empty_choice]
        mock_response.usage = empty_usage
        
        return mock_response

    async def __call_api_parallel(
        self,
        prompts,
        return_logits: bool | list[bool],
        max_new_tokens: int | list[int] | None,
        num_samples: int | list[int],
        stop_sequence: list[str] | None = None,
        metadata: list[dict] | None = None,
    ):
        return_logitss = [return_logits for _ in prompts] if not isinstance(return_logits, list) else return_logits
        max_new_tokenss = [max_new_tokens for _ in prompts] if not isinstance(max_new_tokens, list) else max_new_tokens
        num_sampless = [num_samples for _ in prompts] if not isinstance(num_samples, list) else num_samples
        stop_sequencess = [stop_sequence for _ in prompts]
        assert (
            len(prompts) == len(return_logitss) == len(max_new_tokenss) == len(num_sampless) == len(stop_sequencess)
        ), (
            f"Length of prompts, return_logitss, max_new_tokenss, num_sampless, stop_sequences should be the same but are {len(prompts)}, {len(return_logitss)}, {len(max_new_tokenss)}, {len(num_sampless)}, {len(stop_sequencess)}"
        )

        # Align metadata list with prompts
        metadata_list = metadata if metadata is not None else [None] * len(prompts)
        assert len(prompts) == len(metadata_list), f"Length of prompts ({len(prompts)}) and metadata ({len(metadata_list)}) must be the same"

        # Update semaphore capacity if needed before starting API calls
        await self._update_semaphore_if_needed()

        # Create bounded async API call function with rate limiting
        async def bounded_api_call(prompt, return_logits, max_new_tokens, num_samples, stop_sequence, metadata):
            async with self.semaphore:
                return await self.__call_api(prompt, return_logits, max_new_tokens, num_samples, stop_sequence, metadata)

        # Create tasks for all API calls
        tasks = [
            bounded_api_call(prompt, return_logits, max_new_tokens, num_samples, stop_sequence, metadata)
            for prompt, return_logits, max_new_tokens, num_samples, stop_sequence, metadata 
            in zip(prompts, return_logitss, max_new_tokenss, num_sampless, stop_sequencess, metadata_list)
        ]

        # Wait for all tasks to complete with progress bar
        results = await async_tqdm.gather(*tasks, desc="Async API Calls")

        if None in results:
            raise ValueError("Some entries are not annotated due to errors in async API calls, please inspect and retry.")

        return results

    async def greedy_until(
        self,
        docs: list[Doc],
    ) -> list[ModelResponse]:
        """
        Generates responses using a greedy decoding strategy until certain ending conditions are met.

        Args:
            docs (list[Doc]): list of requests containing the context and ending conditions.

        Returns:
            list[ModelResponse]: list of generated responses.
        """
        # Note: LiteLLM doesn't need tokenized context like transformers models
        # since we send text directly to the API
        dataset = GenerativeTaskDataset(requests=docs, num_dataset_splits=self.DATASET_SPLITS)
        total_requests = dataset.total_size
        results = []

        for split in dataset.splits_iterator():
            contexts = [self.prompt_manager.prepare_prompt_api(doc) for doc in split]
            max_new_tokens = split[0].generation_size  # could be none
            return_logits = split[0].use_logits
            num_samples = split[0].num_samples
            stop_sequence = split[0].stop_sequences

            if num_samples > 1 and self.generation_parameters.temperature == 0:
                raise ValueError(
                    "num_samples > 1 is not supported with temperature=0, please set temperature > 0 or use non sampling metrics."
                )

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

            responses = await self.__call_api_parallel(contexts, return_logits, max_new_tokens, num_samples, stop_sequence, metadata_list)

            for response, context in zip(responses, contexts):
                result: list[str] = [self._format_response_with_reasoning(choice.message) for choice in response.choices]

                # Extract token usage from LiteLLM response for logging
                input_token_count = response.usage.prompt_tokens if response.usage else 0
                output_token_count = response.usage.completion_tokens if response.usage else 0
                
                # Use token counts directly - logging only needs these for hashing
                input_tokens = [input_token_count]  # Just the count for hashing
                output_tokens = [[output_token_count]] if result[0] else [[0]]

                cur_response = ModelResponse(
                    # In empty responses, the model should return an empty string instead of None
                    text=result if result[0] else [""],
                    input=context,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
                results.append(cur_response)

        return dataset.get_original_order(results)

    @property
    def tokenizer(self):
        return self._tokenizer

    def tok_encode(self, str_to_encode: str | list[str], add_special_tokens: bool | None = None) -> list[int] | list[list[int]]:
        """Encode string(s) using LiteLLM's encode function.
        
        Args:
            str_to_encode: String or list of strings to encode
            add_special_tokens: Ignored for LiteLLM (compatibility parameter)
            
        Returns:
            List of token IDs or list of lists of token IDs
        """
        if isinstance(str_to_encode, str):
            return self._tokenizer(model=self.model, text=str_to_encode)
        else:
            return [self._tokenizer(model=self.model, text=text) for text in str_to_encode]

    @property
    def add_special_tokens(self) -> bool:
        return False

    @property
    def max_length(self) -> int:
        """Return the maximum sequence length of the model."""
        return 4096

    def loglikelihood(self, docs: list[Doc]) -> list[ModelResponse]:
        """Tokenize the context and continuation and compute the log likelihood of those
        tokenized sequences.
        """
        raise NotImplementedError

    def loglikelihood_rolling(self, docs: list[Doc]) -> list[ModelResponse]:
        """This function is used to compute the log likelihood of the context for perplexity metrics."""
        raise NotImplementedError