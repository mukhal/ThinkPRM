import argparse
import numpy as np
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from datasets import load_dataset
from tqdm.auto import tqdm
import pandas as pd
from datasets import Dataset
from concurrent.futures import ProcessPoolExecutor, as_completed
import json, os

from grader import *

from parser import *
from utils import load_jsonl
from python_executor import PythonExecutor


def pass_at_k(n: int, c: int, k: int) -> float:
    """A numerically stable method for calculating an unbiased estimate of pass@k.

    Taken from OpenAI's Codex paper: https://arxiv.org/abs/2107.03374

    Args:
        n (`int`): total number of samples
        c (`int`): number of correct samples
        k (`int`): k in pass@$k$

    Returns:
        `float`: an unbiased estimate of pass@k
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def compute_pass_at_1(benchmark: str, data_file: str, output_file: str, max_num_samples=None):
    samples = load_jsonl(data_file)
    samples = Dataset.from_list(samples)
    if "idx" not in samples.column_names:
        samples = samples.map(lambda x, idx: {"idx": idx}, with_indices=True)
        
    if max_num_samples:
        print(f"max_num_samples: {max_num_samples} / {len(samples)}")
        samples = samples[:max_num_samples]

    def parse_gt(x):
        x['gt_cot'], x['gt'] = parse_ground_truth(x, benchmark)
        return x
    samples = samples.map(parse_gt, desc="Parsing ground truth", num_proc=1, load_from_cache_file=False)
    
    # Calculate pass@1 for each sample
    results = []
    # Define a function to process each sample
    def process_sample(sample):
        if 'completions' not in sample:
            sample["pass@1"] = 0.0
            return sample
        
        gt = sample['gt']
        correct_count = 0
        total_count = len(sample['completions'])
        
        for completion in sample['completions']:
            try:
                # Extract answer from completion
                extracted = extract_answer_map({"pred": completion}, benchmark, "pred")["pred"]
                # Check equality
                result = math_equal_process((0, extracted, gt))
                if result:
                    correct_count += 1
            except:
                continue
        
        # Use pass@1 formula to calculate the score
        pass_at_1_score = pass_at_k(total_count, correct_count, 1)
        sample["pass@1"] = pass_at_1_score
        return sample
    
    # Use dataset.map for parallel processing
    samples_with_pass_at_1 = samples.map(
        process_sample,
        desc="Computing pass@1",
        num_proc=8,  # Use 8 processes for parallelization
        load_from_cache_file=False
    )
    
    # Convert to list for the results
    results = list(samples_with_pass_at_1)
    
    # Save the updated dataset
    with open(output_file, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')
    
    print(f"Saved results to {output_file}")
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="math")
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--max_num_samples", type=int, default=None)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    # If output file not specified, use the same directory as data_file
    if args.output_file is None:
        save_dir = os.path.dirname(args.data_file)
        base_name = os.path.basename(args.data_file).split('.')[0]
        args.output_file = os.path.join(save_dir, f"{base_name}_with_pass_at_1.jsonl")
    
    compute_pass_at_1(
        benchmark=args.benchmark,
        data_file=args.data_file,
        output_file=args.output_file,
        max_num_samples=args.max_num_samples
    )
