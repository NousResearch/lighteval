# MIT License

# Copyright (c) 2025 The HuggingFace Team

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
"""Usage:
lighteval vllm \
    "pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,dtype=bfloat16,data_parallel_size=8,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={temperature:0.6,top_p:0.95}" \
    "extended|lcb:codegeneration|0|0"

lighteval vllm \
    "pretrained=Qwen/Qwen2.5-Coder-3B-Instruct,dtype=bfloat16,data_parallel_size=8,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={temperature:0.2,top_p:0.95}" \
    "extended|lcb:codegeneration|0|0"
"""

import json
import os
from datetime import datetime
from typing import Any, Optional

import numpy as np
from aenum import extend_enum

from lighteval.metrics.metrics import Metrics, SampleLevelMetric
from lighteval.tasks.extended.lcb.codegen_metrics import (
    codegen_metrics,
    extract_code,
    translate_private_test_cases,
)
from lighteval.tasks.extended.lcb.streaming_evaluator import (
    enhanced_codegen_metrics,
    streaming_codegen_metrics,
)
from lighteval.tasks.lighteval_task import Doc, LightevalTaskConfig, LightevalTask
from datasets import load_dataset, Dataset
import pandas as pd
from lighteval.tasks.requests import SamplingMethod


class LCBPromptConstants:
    """System prompts adapted from livecodebench-nous for different model types."""
    
    SYSTEM_MESSAGE_GENERIC = "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."
    
    SYSTEM_MESSAGE_DEEPSEEK = "You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you answer questions related to computer science."
    
    SYSTEM_MESSAGE_GEMINI = "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. Do NOT use system calls like `exit` in the generated program. Ensure that the first code block contains the solution."
    
    SYSTEM_MESSAGE_CODEQWEN = "You are a helpful assistant."
    
    SYSTEM_MESSAGE_DEEPSEEK_R1 = (
        "A conversation between User and Assistant. "
        "The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively."
    )
    
    FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
    
    FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."


def get_system_prompt(model_name: str = "", custom_prepend: str = "") -> str:
    """Get appropriate system prompt based on model name."""
    model_name_lower = model_name.lower()
    
    if "deepseek-r1" in model_name_lower:
        system_prompt = LCBPromptConstants.SYSTEM_MESSAGE_DEEPSEEK_R1
    elif "deepseek" in model_name_lower:
        system_prompt = LCBPromptConstants.SYSTEM_MESSAGE_DEEPSEEK
    elif "gemini" in model_name_lower:
        system_prompt = LCBPromptConstants.SYSTEM_MESSAGE_GEMINI
    elif "codeqwen" in model_name_lower or "qwen" in model_name_lower:
        system_prompt = LCBPromptConstants.SYSTEM_MESSAGE_CODEQWEN
    else:
        system_prompt = LCBPromptConstants.SYSTEM_MESSAGE_GENERIC
    
    if custom_prepend:
        return f"{custom_prepend}\n\n{system_prompt}"
    return system_prompt


def prepare_prompt(line: dict[str, Any]) -> str:
    """Prepare the user prompt following livecodebench-nous format."""
    prompt = f"### Question:\n{line['question_content']}\n\n"
    
    if starter_code := line.get("starter_code", None):
        prompt += f"### Format: {LCBPromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {LCBPromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt




def lcb_codegeneration_prompt_fn(line, task_name: str = "lcb:codegeneration") -> Doc:
    """Create a Doc for code generation task with system prompt support.
    
    Note: When using with evaluate.sh/alitellm, the system prompt should be 
    passed via MODEL_ARGS rather than the instruction field for proper compatibility.
    The instruction field is kept for backwards compatibility with other runners.
    """
    query = prepare_prompt(line)
    
    # Get system prompt for fallback (when not using alitellm MODEL_ARGS)
    # For alitellm compatibility, system prompts should be in MODEL_ARGS
    model_name = os.getenv("LCB_MODEL_NAME", "")
    custom_prepend = os.getenv("LCB_PREPEND_SYSTEM_PROMPT", "")
    fallback_system_prompt = get_system_prompt(model_name, custom_prepend)
    
    # List of dicts of the form: [{"input": "6\nabc\nacb\nbac\nbca\ncab\ncba\n", "output": "YES\nYES\nYES\nNO\nNO\nYES\n", "testtype": "stdin"}]
    public_test_cases = json.loads(line["public_test_cases"])
    private_test_cases = translate_private_test_cases(line["private_test_cases"])
    inputs = [test["input"] for test in public_test_cases + private_test_cases]
    outputs = [test["output"] for test in public_test_cases + private_test_cases]
    
    return Doc(
        task_name=task_name,
        query=query,
        instruction=fallback_system_prompt,  # Fallback for non-alitellm usage
        choices=[""],
        gold_index=0,
        specific={
            "inputs": inputs,
            "outputs": outputs,
            "fn_name": json.loads(line["metadata"]).get("func_name", None),
        },
    )


# Note: Date filtering is now handled by pre-filtering the dataset with pandas
# The lcb_codegeneration_aug2024_prompt_fn is no longer needed


def codegen_metric(doc: Doc, model_response, **kwargs) -> float:
    """Estimates the Pass@1 metric for the code generation task.
    Extract the code from each prediction, Runs it for each sample and generations,
    and computes the Pass@1 over the outputs.
    """
    # Note: Filtering is now handled by pre-filtering the dataset, so no need to check here
    
    # Extract generated code snippets from model response
    predictions = model_response.text if hasattr(model_response, 'text') else [str(model_response)]
    
    # Extract code and handle empty extractions
    extracted_codes = []
    for pred in predictions:
        code = extract_code(pred)
        if not code.strip():  # If extraction failed, use the full prediction
            code = pred
        extracted_codes.append(code)
    
    generated_code_snippets = [extracted_codes]
    
    # Prepare evaluation data from doc
    evaluation_sample = {
        "inputs": doc.specific["inputs"],
        "outputs": doc.specific["outputs"],
        "fn_name": doc.specific["fn_name"],
    }
    # This is a list of lists because codegen_metrics expects this format
    evaluation_sample = [{"input_output": json.dumps(evaluation_sample)}]

    # Use enhanced streaming evaluation with overlapped inference/scoring
    use_streaming = os.getenv("LCB_USE_STREAMING", "true").lower() == "true"
    max_workers = int(os.getenv("LCB_MAX_WORKERS", "16"))
    
    # Debug: Check if we have valid data
    if not extracted_codes or all(not code.strip() for code in extracted_codes):
        return 0.0  # No valid code found
    
    try:
        metrics, _ = enhanced_codegen_metrics(
            evaluation_sample,
            generated_code_snippets,
            k_list=[1],  # Only run for Pass@1
            num_process_evaluate=max_workers,
            use_streaming=use_streaming,
            enable_progress=False,  # Disable progress bar in metric computation
        )
        return metrics.get("pass@1", 0.0)
    except Exception as e:
        # Fallback to original implementation if streaming fails
        print(f"Streaming evaluation failed, falling back to original: {e}")
        from .codegen_metrics import codegen_metrics
        metrics, _ = codegen_metrics(
            evaluation_sample,
            generated_code_snippets,
            k_list=[1],
            num_process_evaluate=max_workers,
        )
        return metrics.get("pass@1", 0.0)


lcb_codegen_metric = SampleLevelMetric(
    metric_name="codegen_pass@1:16",  # This is the way of informing the number of generations currently
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=codegen_metric,
    corpus_level_fn=np.mean,
)


extend_enum(Metrics, "lcb_codegen_metric", lcb_codegen_metric)

# Dataset configurations with date information
configs = [
    "release_v1",    # May 2023 - Mar 2024 (400 problems)
    "release_v2",    # May 2023 - May 2024 (511 problems)
    "release_v3",    # May 2023 - Jul 2024 (612 problems)
    "release_v4",    # May 2023 - Sep 2024 (713 problems)
    "release_v5",    # May 2023 - Jan 2025 (880 problems)
    "release_v6",    # May 2023 - Apr 2025 (1055 problems)
    "release_latest",
    "v1",
    "v2",
    "v3",
    "v4",
    "v5",
    "v6",
    "v1_v2",
    "v1_v3",
    "v1_v4",
    "v1_v5",
    "v2_v3",
    "v2_v4",
    "v2_v5",
    "v3_v4",
    "v3_v5",
    "v4_v5",
]

tasks = []

# Create tasks for all configurations
for subset in configs:
    # To keep the base subset as the default, the others are named "lcb:codegeneration_v4", "lcb:codegeneration_v5"... etc
    name = "lcb:codegeneration" if subset == "v4_v5" else f"lcb:codegeneration_{subset}"
    task = LightevalTaskConfig(
        name=name,
        suite=["extended"],
        prompt_function=lcb_codegeneration_prompt_fn,
        hf_repo="livecodebench/code_generation_lite",
        hf_subset=subset,  # https://github.com/LiveCodeBench/LiveCodeBench/tree/main?tab=readme-ov-file#dataset-versions
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        generation_size=32768,
        metrics=[Metrics.lcb_codegen_metric],
        stop_sequence=[],  # no stop sequence, will use EOS token
        trust_dataset=True,
        version=0,
    )
    tasks.append(task)

# Create tasks that use standard datasets (date filtering handled by native LCB --start_date)
# Note: For August 2024+ filtering, use release_v4 or later and pass --start_date to native LCB runner

def create_date_filter(start_date_str: str = "2024-08-01"):
    """Create an hf_filter function for date-based filtering."""
    import pandas as pd
    from rich.console import Console
    
    console = Console()
    start_date = pd.to_datetime(start_date_str)
    
    def date_filter(example):
        """Filter function to keep only examples after start_date."""
        try:
            contest_date = pd.to_datetime(example['contest_date'])
            return contest_date >= start_date
        except:
            # If date parsing fails, include the example
            return True
    
    # Print filtering info when filter is created
    console.print(f"[blue]📅 Created date filter for problems >= {start_date_str}[/blue]")
    
    return date_filter

# Create a task that uses hf_filter for date filtering
start_date_str = os.getenv("LCB_START_DATE", "2024-08-01")
august_2024_task = LightevalTaskConfig(
    name="lcb:codegeneration_aug2024_plus",
    suite=["extended"],
    prompt_function=lcb_codegeneration_prompt_fn,
    hf_repo="livecodebench/code_generation_lite",
    hf_subset="release_v6",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    generation_size=32768,
    metrics=[Metrics.lcb_codegen_metric],
    stop_sequence=[],
    trust_dataset=True,
    hf_filter=create_date_filter(start_date_str),  # Apply date filtering
    version=0,
)

# Add the filtered task to the tasks list
tasks.append(august_2024_task)


TASKS_TABLE = tasks

# Environment variable configuration guide:
# LCB_MODEL_NAME: Set to model name for automatic system prompt selection
# LCB_PREPEND_SYSTEM_PROMPT: Add custom text to prepend to system prompt
# LCB_USE_STREAMING: Enable streaming evaluation (default: true)
# LCB_MAX_WORKERS: Batch size for pipeline processing (execution is always single-core)
# LCB_START_DATE: Start date for filtering problems (format: YYYY-MM-DD, default: 2024-08-01)
# 
# Example usage with evaluate.sh (alitellm compatible):
# 
# For configurable date filtering:
# export LCB_USE_STREAMING=false  # Single-core for reliability
# export LCB_MAX_WORKERS=1
# export LCB_START_DATE=2024-08-01  # Filter for August 2024+ problems
# 
# ./evaluate.sh --base_url https://openrouter.ai/api/v1 \
#               --api_key $OPENROUTER_API_KEY \
#               --model "deepseek/deepseek-r1-0528" \
#               --system_prompt "You are an expert programming assistant." \
#               --parallel_calls 8 \
#               --out_dir ./results
#
# Note: The "aug2024_plus" task now implements proper date filtering using LCB_START_DATE.
# It loads release_v6 dataset (1055 problems) but only evaluates problems >= LCB_START_DATE.
# Change LCB_START_DATE to any date in YYYY-MM-DD format to adjust the filtering.
#
# The LCB tasks now support:
# - Overlapped inference and scoring via streaming evaluation
# - Single-core code execution for reliability (eliminates race conditions)
# - Queue-based producer-consumer pipeline for efficiency
# - Progress monitoring and intermediate results
#
# For compatibility with evaluate.sh, the system prompt is handled via the MODEL_ARGS
# rather than the instruction field, ensuring proper integration with alitellm
