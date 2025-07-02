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
Streaming evaluation system for LiveCodeBench that overlaps inference and scoring.
Uses queues to pipeline results and multiprocessing for parallel code execution.
"""

import json
import multiprocessing as mp
import queue
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .codegen_metrics import (
    check_correctness,
    compute_metrics_from_results,
    estimate_pass_at_k,
    extract_code,
)


@dataclass
class EvaluationTask:
    """A single evaluation task."""
    task_id: int
    sample: Dict[str, Any]
    generation: str
    timeout: int = 6


@dataclass
class EvaluationResult:
    """Result of evaluating a single generation."""
    task_id: int
    generation_idx: int
    result: List[int | bool]
    execution_time: float


class ParallelCodeExecutor:
    """Code executor (reverted to single-core for reliability)."""
    
    def __init__(self, max_workers: int = None, timeout: int = 6):
        # Use environment variable for max workers, default to reasonable value
        self.max_workers = max_workers or int(os.getenv("LCB_MAX_WORKERS", "16"))
        self.timeout = timeout
        
    def execute_batch(self, tasks: List[EvaluationTask]) -> Dict[int, List[int | bool]]:
        """Execute a batch of tasks sequentially (single-core) for reliability."""
        if not tasks:
            return {}
            
        results = {}
        
        # Execute tasks sequentially to avoid multiprocessing race conditions
        with tqdm(total=len(tasks), desc="Executing code", leave=False) as pbar:
            for task in tasks:
                try:
                    result = self._execute_single_task(task)
                    results[task.task_id] = result
                except Exception as e:
                    # Handle execution failures gracefully
                    results[task.task_id] = [-4]  # Execution error
                finally:
                    pbar.update(1)
        
        return results
    
    @staticmethod
    def _execute_single_task(task: EvaluationTask) -> List[int | bool]:
        """Execute a single task sequentially."""
        try:
            return check_correctness(
                task.sample, 
                task.generation, 
                timeout=task.timeout
            )
        except Exception:
            return [-4]  # Execution error


class StreamingEvaluator:
    """
    Streaming evaluator that overlaps inference and scoring using producer-consumer pattern.
    """
    
    def __init__(
        self,
        max_workers: int = None,
        queue_size: int = 100,
        timeout: int = 6,
        enable_progress: bool = True
    ):
        # Single-core execution but keep batch_size for pipeline efficiency
        self.max_workers = 1  # Always single-core for reliable execution
        self.queue_size = queue_size
        self.timeout = timeout
        self.enable_progress = enable_progress
        # Use reasonable batch size for efficient pipeline processing
        self.batch_size = max_workers or 16  # Batch size for queue processing, not execution
        
        # Queues for pipeline
        self.task_queue = queue.Queue(maxsize=queue_size)
        self.result_queue = queue.Queue()
        
        # Tracking
        self.total_tasks = 0
        self.completed_tasks = 0
        self.results = {}
        
        # Executor for code execution
        self.executor = ParallelCodeExecutor(max_workers=max_workers, timeout=timeout)
        
        # Threading control
        self._stop_event = threading.Event()
        self._threads = []
    
    def start_workers(self):
        """Start background worker threads."""
        # Scorer worker - processes tasks from queue in batches
        scorer_thread = threading.Thread(
            target=self._scorer_worker,
            name="ScorerWorker",
            daemon=True
        )
        scorer_thread.start()
        self._threads.append(scorer_thread)
        
        # Progress monitor
        if self.enable_progress:
            progress_thread = threading.Thread(
                target=self._progress_monitor,
                name="ProgressMonitor", 
                daemon=True
            )
            progress_thread.start()
            self._threads.append(progress_thread)
    
    def stop_workers(self):
        """Stop all worker threads."""
        self._stop_event.set()
        
        # Signal end of tasks
        self.task_queue.put(None)
        
        # Wait for threads to finish
        for thread in self._threads:
            thread.join(timeout=5)
        
        self._threads.clear()
    
    def submit_task(self, task_id: int, sample: Dict[str, Any], generation: str):
        """Submit a single evaluation task."""
        task = EvaluationTask(
            task_id=task_id,
            sample=sample,
            generation=extract_code(generation),
            timeout=self.timeout
        )
        
        # This will block if queue is full, providing backpressure
        self.task_queue.put(task)
        self.total_tasks += 1
    
    def submit_batch(self, samples: List[Dict], generations: List[List[str]]):
        """Submit a batch of tasks."""
        for task_id, (sample, gen_list) in enumerate(zip(samples, generations)):
            # For each generation of this sample
            for gen_idx, generation in enumerate(gen_list):
                # Use unique task_id that encodes both sample and generation
                unique_task_id = task_id * 1000 + gen_idx
                self.submit_task(unique_task_id, sample, generation)
    
    def _scorer_worker(self):
        """Worker thread that scores tasks in batches."""
        batch = []
        
        while not self._stop_event.is_set():
            try:
                # Get task from queue with timeout
                task = self.task_queue.get(timeout=1.0)
                
                # None signals end of tasks
                if task is None:
                    break
                
                batch.append(task)
                
                # Process batch when full or queue is empty
                if len(batch) >= self.batch_size or self.task_queue.empty():
                    if batch:
                        self._process_batch(batch)
                        batch = []
                
            except queue.Empty:
                # Process any remaining tasks in batch
                if batch:
                    self._process_batch(batch)
                    batch = []
                continue
        
        # Process final batch
        if batch:
            self._process_batch(batch)
    
    def _process_batch(self, batch: List[EvaluationTask]):
        """Process a batch of tasks."""
        batch_results = self.executor.execute_batch(batch)
        
        # Store results and update counters
        for task in batch:
            if task.task_id in batch_results:
                self.results[task.task_id] = batch_results[task.task_id]
            else:
                self.results[task.task_id] = [-4]  # Error fallback
            
            self.completed_tasks += 1
            self.result_queue.put(task.task_id)
    
    def _progress_monitor(self):
        """Monitor and display progress."""
        pbar = tqdm(desc="Evaluating", unit="tasks", dynamic_ncols=True)
        
        while not self._stop_event.is_set():
            try:
                # Wait for a result
                self.result_queue.get(timeout=1.0)
                pbar.update(1)
                pbar.set_postfix({
                    'completed': self.completed_tasks,
                    'total': self.total_tasks,
                    'queue_size': self.task_queue.qsize()
                })
            except queue.Empty:
                continue
        
        pbar.close()
    
    def get_results(self) -> Dict[int, List[int | bool]]:
        """Get all results collected so far."""
        return self.results.copy()
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> Dict[int, List[int | bool]]:
        """Wait for all submitted tasks to complete."""
        start_time = time.time()
        
        while self.completed_tasks < self.total_tasks:
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Evaluation timed out after {timeout}s")
            
            time.sleep(0.1)
        
        return self.get_results()


def streaming_codegen_metrics(
    samples: List[Dict],
    generations: List[List[str]],
    k_list: List[int] = [1, 5],
    max_workers: int = None,
    timeout: int = 6,
    enable_progress: bool = True
) -> Tuple[Dict[str, float], Dict[int, List[int | bool]]]:
    """
    Enhanced codegen metrics with overlapped inference/scoring pipeline.
    
    Uses single-core execution for reliability but overlaps the inference and
    scoring phases using a queue-based producer-consumer pattern.
    
    Args:
        samples: List of problem samples
        generations: List of lists of code generations
        k_list: Values of k for pass@k calculation
        max_workers: Batch size for pipeline processing (execution is always single-core)
        timeout: Timeout per test case
        enable_progress: Whether to show progress bar
    
    Returns:
        Tuple of (metrics, detailed_results)
    """
    evaluator = StreamingEvaluator(
        max_workers=max_workers,
        timeout=timeout,
        enable_progress=enable_progress
    )
    
    try:
        # Start workers
        evaluator.start_workers()
        
        # Submit all tasks
        evaluator.submit_batch(samples, generations)
        
        # Wait for completion
        raw_results = evaluator.wait_for_completion()
        
        # Convert results to expected format
        results = {}
        for task_id, result in raw_results.items():
            sample_id = task_id // 1000
            gen_idx = task_id % 1000
            
            if sample_id not in results:
                results[sample_id] = []
            
            # Ensure we have enough slots
            while len(results[sample_id]) <= gen_idx:
                results[sample_id].append([-4])
            
            results[sample_id][gen_idx] = result
        
        # Compute final metrics
        metrics = compute_metrics_from_results(results, k_list=k_list)
        
        return metrics, results
        
    finally:
        evaluator.stop_workers()


# Backwards compatibility wrapper
def enhanced_codegen_metrics(
    samples,
    generations,
    k_list=[1, 5],
    num_process_evaluate=16,
    timeout=6,
    use_streaming=True,
    **kwargs
):
    """Enhanced version of codegen_metrics with optional streaming."""
    if use_streaming:
        return streaming_codegen_metrics(
            samples=samples,
            generations=generations,
            k_list=k_list,
            max_workers=num_process_evaluate,
            timeout=timeout,
            enable_progress=kwargs.get('enable_progress', True)
        )
    else:
        # Fall back to original implementation
        from .codegen_metrics import codegen_metrics
        return codegen_metrics(
            samples,
            generations,
            k_list=k_list,
            num_process_evaluate=num_process_evaluate,
            timeout=timeout,
        )