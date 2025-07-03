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

"""
Batch-level metric for LiveCodeBench that can process multiple samples simultaneously
with overlapped inference and scoring.
"""

import json
import os
from typing import List

import numpy as np

from lighteval.tasks.lighteval_task import Doc
from .codegen_metrics import extract_code
from .streaming_evaluator import streaming_codegen_metrics


class BatchCodegenMetric:
    """
    A batch-level metric implementation that can process multiple samples together
    for better performance with overlapped inference and scoring.
    """
    
    def __init__(self, max_workers: int = None, use_streaming: bool = True):
        self.max_workers = max_workers or int(os.getenv("LCB_MAX_WORKERS", "16"))
        self.use_streaming = use_streaming and (os.getenv("LCB_USE_STREAMING", "true").lower() == "true")
        
        # Accumulate samples for batch processing
        self.pending_samples = []
        self.pending_docs = []
        self.pending_responses = []
        
    def add_sample(self, doc: Doc, model_response) -> None:
        """Add a sample to the batch for processing."""
        # Extract generated code snippets from model response
        predictions = model_response.text if hasattr(model_response, 'text') else [str(model_response)]
        
        # Prepare evaluation data from doc
        evaluation_sample = {
            "inputs": doc.specific["inputs"],
            "outputs": doc.specific["outputs"],
            "fn_name": doc.specific["fn_name"],
        }
        
        self.pending_samples.append({"input_output": json.dumps(evaluation_sample)})
        self.pending_docs.append(doc)
        self.pending_responses.append([extract_code(pred) for pred in predictions])
    
    def process_batch(self) -> List[float]:
        """Process all accumulated samples and return individual scores."""
        if not self.pending_samples:
            return []
        
        if self.use_streaming:
            metrics, detailed_results = streaming_codegen_metrics(
                samples=self.pending_samples,
                generations=self.pending_responses,
                k_list=[1],
                max_workers=self.max_workers,
                enable_progress=True,
            )
        else:
            from .codegen_metrics import codegen_metrics
            metrics, detailed_results = codegen_metrics(
                self.pending_samples,
                self.pending_responses,
                k_list=[1],
                num_process_evaluate=self.max_workers,
            )
        
        # Extract individual scores
        scores = []
        for i in range(len(self.pending_samples)):
            if i in detailed_results:
                # Calculate pass@1 for this specific sample
                all_correct = []
                for generation_result in detailed_results[i]:
                    gen = np.array(generation_result)
                    all_correct.append(np.all(gen > 0))
                
                # Pass@1 is 1.0 if any generation passed, 0.0 otherwise
                scores.append(1.0 if any(all_correct) else 0.0)
            else:
                scores.append(0.0)  # Failed
        
        # Clear batch
        self.clear_batch()
        
        return scores
    
    def clear_batch(self):
        """Clear the accumulated batch."""
        self.pending_samples.clear()
        self.pending_docs.clear()
        self.pending_responses.clear()
    
    def size(self) -> int:
        """Return the current batch size."""
        return len(self.pending_samples)


# Global batch processor instance
_batch_processor = None

def get_batch_processor() -> BatchCodegenMetric:
    """Get or create the global batch processor."""
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = BatchCodegenMetric()
    return _batch_processor


def batched_codegen_metric(doc: Doc, model_response, **kwargs) -> float:
    """
    Batched version of codegen metric that accumulates samples for batch processing.
    
    Note: This requires special handling in the evaluation pipeline to trigger
    batch processing at appropriate intervals.
    """
    processor = get_batch_processor()
    
    # Add this sample to the batch
    processor.add_sample(doc, model_response)
    
    # Check if we should process the batch
    # Use max_workers as the batch size threshold
    if processor.size() >= processor.max_workers:
        # Process the batch and return the score for this sample
        scores = processor.process_batch()
        return scores[-1]  # Return score for the last (current) sample
    else:
        # Return a placeholder - this will be updated when batch is processed
        # Note: This approach requires careful handling in the evaluation framework
        return 0.0


def flush_batch_processor() -> List[float]:
    """Force process any remaining samples in the batch."""
    processor = get_batch_processor()
    if processor.size() > 0:
        return processor.process_batch()
    return []