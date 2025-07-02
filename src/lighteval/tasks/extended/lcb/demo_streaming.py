#!/usr/bin/env python3
"""
Demo script showing the performance difference between traditional and streaming evaluation.
"""

import time
import json
from .codegen_metrics import codegen_metrics
from .streaming_evaluator import streaming_codegen_metrics

def create_demo_data(num_samples=10, num_generations=3):
    """Create demo data for testing."""
    samples = []
    generations = []
    
    for i in range(num_samples):
        # Simple demo problem
        sample = {
            "input_output": json.dumps({
                "inputs": ["1 2", "3 4"],
                "outputs": ["3", "7"],
                "fn_name": None  # Standard input
            })
        }
        samples.append(sample)
        
        # Generate some demo code (some correct, some incorrect)
        gen_list = []
        for j in range(num_generations):
            if j == 0:  # First generation is correct
                code = """
a, b = map(int, input().split())
print(a + b)
"""
            else:  # Others have errors
                code = f"""
# This is generation {j} with potential errors
a, b = map(int, input().split())
print(a * b)  # Wrong operation
"""
            gen_list.append(code)
        
        generations.append(gen_list)
    
    return samples, generations

def benchmark_evaluation():
    """Benchmark traditional vs streaming evaluation."""
    print("Creating demo data...")
    samples, generations = create_demo_data(num_samples=20, num_generations=5)
    
    print(f"Testing with {len(samples)} samples, {len(generations[0])} generations each")
    print("=" * 60)
    
    # Traditional evaluation
    print("Testing traditional evaluation...")
    start_time = time.time()
    traditional_metrics, traditional_results = codegen_metrics(
        samples,
        generations,
        k_list=[1],
        num_process_evaluate=8,
        timeout=3
    )
    traditional_time = time.time() - start_time
    
    print(f"Traditional: {traditional_time:.2f}s")
    print(f"Traditional Pass@1: {traditional_metrics['pass@1']:.3f}")
    
    # Streaming evaluation
    print("\nTesting streaming evaluation...")
    start_time = time.time()
    streaming_metrics, streaming_results = streaming_codegen_metrics(
        samples,
        generations,
        k_list=[1],
        max_workers=8,
        timeout=3,
        enable_progress=True
    )
    streaming_time = time.time() - start_time
    
    print(f"Streaming: {streaming_time:.2f}s")
    print(f"Streaming Pass@1: {streaming_metrics['pass@1']:.3f}")
    
    # Compare results
    print(f"\nSpeedup: {traditional_time / streaming_time:.2f}x")
    print(f"Results match: {abs(traditional_metrics['pass@1'] - streaming_metrics['pass@1']) < 0.001}")

if __name__ == "__main__":
    benchmark_evaluation()