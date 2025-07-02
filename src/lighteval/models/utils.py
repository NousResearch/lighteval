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

import json
import os
import re
from itertools import islice
from typing import Optional, Union

import torch
import yaml
from huggingface_hub import HfApi
from pydantic import BaseModel
from transformers.models.auto.configuration_auto import AutoConfig

from lighteval.models.model_input import GenerationParameters


class ModelConfig(BaseModel, extra="forbid"):
    """
    Base configuration class for all model types in Lighteval.

    This is the foundation class that all specific model configurations inherit from.
    It provides common functionality for parsing configuration from files and command-line arguments,
    as well as shared attributes that are used by all models like generation parameters and system prompts.

    Attributes:
        generation_parameters (GenerationParameters):
            Configuration parameters that control text generation behavior, including
            temperature, top_p, max_new_tokens, etc. Defaults to empty GenerationParameters.
        system_prompt (str | None):
            Optional system prompt to be used with chat models. This prompt sets the
            behavior and context for the model during evaluation.

    Methods:
        from_path(path: str):
            Load configuration from a YAML file.
        from_args(args: str):
            Parse configuration from a command-line argument string.
        _parse_args(args: str):
            Static method to parse argument strings into configuration dictionaries.

    Example:
        ```python
        # Load from YAML file
        config = ModelConfig.from_path("model_config.yaml")

        # Load from command line arguments
        config = ModelConfig.from_args("model_name=meta-llama/Llama-3.1-8B-Instruct,system_prompt='You are a helpful assistant.',generation_parameters={temperature=0.7}")

        # Direct instantiation
        config = ModelConfig(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            generation_parameters=GenerationParameters(temperature=0.7),
            system_prompt="You are a helpful assistant."
        )
        ```
    """

    generation_parameters: GenerationParameters = GenerationParameters()
    system_prompt: str | None = None

    @classmethod
    def from_path(cls, path: str):
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        return cls(**config["model_parameters"])

    @classmethod
    def from_args(cls, args: str):
        config = cls._parse_args(args)
        return cls(**config)

    @staticmethod
    def _parse_args(args: str) -> dict:
        """Parse a string of arguments into a configuration dictionary.

        This function parses a string containing model arguments and generation parameters
        into a structured dictionary. It properly handles quoted strings that may contain commas.

        Args:
            args (str): A string containing comma-separated key-value pairs, where generation
                parameters can be specified in a nested JSON-like format.

        Returns:
            dict: A dictionary containing the parsed configuration parameters

        Examples:
            >>> parse_args("model_name=gpt2,max_length=100")
            {'model_name': 'gpt2', 'max_length': '100'}

            >>> parse_args("model_name=gpt2,system_prompt='Hello, world!'")
            {'model_name': 'gpt2', 'system_prompt': 'Hello, world!'}

            >>> parse_args("model_name=gpt2,generation_parameters={temperature:0.7,top_p:0.9}")
            {'model_name': 'gpt2', 'generation_parameters': {'temperature': 0.7, 'top_p': 0.9}}
        """
        def split_args_respecting_quotes(text):
            """Split on commas but respect quoted strings and nested braces"""
            parts = []
            current = ""
            in_quotes = False
            quote_char = None
            brace_depth = 0
            i = 0
            
            while i < len(text):
                char = text[i]
                
                if char in ('"', "'") and not in_quotes:
                    in_quotes = True
                    quote_char = char
                    current += char
                elif char == quote_char and in_quotes:
                    in_quotes = False
                    quote_char = None
                    current += char
                elif char == '{' and not in_quotes:
                    brace_depth += 1
                    current += char
                elif char == '}' and not in_quotes:
                    brace_depth -= 1
                    current += char
                elif char == ',' and not in_quotes and brace_depth == 0:
                    parts.append(current.strip())
                    current = ""
                else:
                    current += char
                i += 1
            
            if current.strip():
                parts.append(current.strip())
            
            return parts

        # Split arguments respecting quotes and braces
        arg_parts = split_args_respecting_quotes(args)
        
        model_config = {}
        generation_parameters_dict = None
        
        for part in arg_parts:
            part = part.strip()
            if not part:
                continue
                
            if "=" in part:
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip()
                
                # Handle quoted strings
                if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
                    value = value[1:-1]  # Remove quotes
                
                # Handle generation_parameters specially
                if key == "generation_parameters":
                    if value.startswith("{") and value.endswith("}"):
                        gen_params = re.sub(r"(\w+):", r'"\1":', value)
                        generation_parameters_dict = json.loads(gen_params)
                    continue
                
                model_config[key] = value
            else:
                # Boolean flag (no =)
                model_config[part] = True
        
        if generation_parameters_dict is not None:
            model_config["generation_parameters"] = generation_parameters_dict

        return model_config


def _get_dtype(dtype: Union[str, torch.dtype, None], config: Optional[AutoConfig] = None) -> Optional[torch.dtype]:
    """
    Get the torch dtype based on the input arguments.

    Args:
        dtype (Union[str, torch.dtype]): The desired dtype. Can be a string or a torch dtype.
        config (Optional[transformers.AutoConfig]): The model config object. Defaults to None.

    Returns:
        torch.dtype: The torch dtype based on the input arguments.
    """

    if config is not None and hasattr(config, "quantization_config"):
        # must be infered
        return None

    if dtype is not None:
        if isinstance(dtype, str) and dtype not in ["auto", "4bit", "8bit"]:
            # Convert `str` args torch dtype: `float16` -> `torch.float16`
            return getattr(torch, dtype)
        elif isinstance(dtype, torch.dtype):
            return dtype

    if config is not None:
        return config.torch_dtype

    return None


def _simplify_name(name_or_path: str) -> str:
    """
    If the model is loaded from disk, then the name will have the following format:
    /p/a/t/h/models--org--model_name/revision/model_files
    This function return the model_name as if it was loaded from the hub:
    org/model_name

    Args:
        name_or_path (str): The name or path to be simplified.

    Returns:
        str: The simplified name.
    """
    if os.path.isdir(name_or_path) or os.path.isfile(name_or_path):  # Loading from disk
        simple_name_list = name_or_path.split("/")
        # The following manages files stored on disk, loaded with the hub model format:
        # /p/a/t/h/models--org--model_name/revision/model_files
        if any("models--" in item for item in simple_name_list):  # Hub format
            simple_name = [item for item in simple_name_list if "models--" in item][0]
            simple_name = simple_name.replace("models--", "").replace("--", "/")
            return simple_name
        # This is for custom folders
        else:  # Just a folder, we don't know the shape
            return name_or_path.replace("/", "_")

    return name_or_path


def _get_model_sha(repo_id: str, revision: str):
    api = HfApi()
    try:
        model_info = api.model_info(repo_id=repo_id, revision=revision)
        return model_info.sha
    except Exception:
        return ""


def batched(iterable, n):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch
