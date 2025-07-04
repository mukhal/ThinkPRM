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
from scipy import stats

from grader import *
from parser import *
from utils import load_jsonl
from python_executor import PythonExecutor

def evaluate_correlation(benchmark: str, data_file: str, dataset_col: str = "pred", samples: list=None, max_num_samples=None):
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
    
    # Get correctness and confidence scores
    all_correctness = []  # List of lists - correctness for each completion per sample
    all_scores = []  # List of lists - confidence scores for each completion per sample
    timeout_cnt = 0
    
    # Process each sample
    for idx, completions, gt in tqdm(zip(samples['idx'], samples['completions'], samples['gt']), total=len(samples), desc="Evaluate"):
        sample_correctness = []
        params = [(idx, completion, gt) for completion in completions]
        
        # Process completions in parallel
        with ProcessPool(max_workers=16) as pool:
            future = pool.map(math_equal_process, params, timeout=3)
            iterator = future.result()
            
            for _ in range(len(params)):
                try:
                    result = next(iterator)
                    sample_correctness.append(result)
                except TimeoutError:
                    sample_correctness.append(False)
                    timeout_cnt += 1
                except StopIteration:
                    break
                except Exception:
                    exit()
        
        all_correctness.append(sample_correctness)
        all_scores.append(samples[idx]['agg_scores'])

    # Compute correlation metrics
    spearman_corrs = []
    kendall_taus = []

    for correctness, scores in zip(all_correctness, all_scores):
        # Convert to numpy arrays
        correctness = np.array(correctness, dtype=float)
        scores = np.array(scores)
        
        # Spearman correlation
        spearman_corr, _ = stats.spearmanr(scores, correctness)
        spearman_corrs.append(spearman_corr)
        
        # Kendall's tau
        kendall_tau, _ = stats.kendalltau(scores, correctness)
        kendall_taus.append(kendall_tau)

    # Average metrics across all samples
    result_json = {
        "num_samples": len(samples),
        "timeout_samples": timeout_cnt,
        "spearman": float(np.mean(spearman_corrs)),
        "kendall": float(np.mean(kendall_taus))
    }
    return samples, result_json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="math")
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--max_num_samples", type=int, default=None)
    parser.add_argument("--voting_n", type=int, nargs='+', required=True)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    data = {"n": [], "spearman": [], "kendall": []}

    def evaluate_for_n(n):
        local_data = {"n": n, "spearman": None, "kendall": None}
        for agg in ["weighted"]:
            _, scores = evaluate_correlation(
                benchmark=args.benchmark,
                data_file=args.data_file,
                dataset_col=f"pred_{agg}@{n}",
                max_num_samples=args.max_num_samples,
            )
            local_data["spearman"] = scores["spearman"]
            local_data["kendall"] = scores["kendall"]
        return local_data

    with tqdm(total=len(args.voting_n), desc="Evaluating correlation metrics") as progress_bar:
        for n in args.voting_n:
            try:
                result = evaluate_for_n(n)
                data["n"].append(result["n"])
                data["spearman"].append(result["spearman"])
                data["kendall"].append(result["kendall"])
            except Exception as e:
                print(f"Error processing n={n}: {e}")
            progress_bar.update(1)
            
    print("Correlation metrics:", data)
