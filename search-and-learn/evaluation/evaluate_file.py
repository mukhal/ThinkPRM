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

def evaluate(benchmark: str, data_file: str, dataset_col: str = "pred", samples: list=None, max_num_samples=None):
    samples = load_jsonl(data_file)
    samples = Dataset.from_list(samples)
    if "idx" not in samples.column_names:
        samples = samples.map(lambda x, idx: {"idx": idx}, with_indices=True)
        
    # Check if 'solution' column is missing but 'answer' column exists
    if 'solution' not in samples.column_names and 'answer' in samples.column_names:
        print("'solution' column not found. Copying from 'answer' column.")
        # Create a new column while preserving all existing columns
        samples = samples.map(lambda x: {**x, "solution": f"\\boxed{{{x['answer']}}}"})
    
    if max_num_samples:
        print(f"max_num_samples: {max_num_samples} / {len(samples)}")
        samples = samples[:max_num_samples]

    def parse_gt(x):
        x['gt_cot'], x['gt'] = parse_ground_truth(x, benchmark)
        return x
    samples = samples.map(parse_gt, desc="Parsing ground truth", num_proc=1, load_from_cache_file=False)
    
    n = int(dataset_col.split("@")[1])
    # Calculate coverage
    coverage_scores = []
    for sample in tqdm(samples, desc="Calculating coverage"):
        has_correct = False
        if 'completions' in sample:
            gt = sample['gt']
            for completion in sample['completions'][:n]:
                try:
                    # Extract answer from completion
                    extracted = extract_answer_map({"pred": completion}, benchmark, "pred")["pred"]
                    # Check equality
                    result = math_equal_process((0, extracted, gt))
                    if result:
                        has_correct = True
                        break
                except:
                    continue
        coverage_scores.append(has_correct)
    
    # Calculate regular accuracy
    samples = samples.map(extract_answer_map, fn_kwargs={"data_name": benchmark, "col": dataset_col}, desc="Parsing predictions", num_proc=1, load_from_cache_file=False)
    params = [(idx, pred, gt) for idx, pred, gt in zip(samples['idx'], samples['pred'], samples['gt'])]

    scores = []
    timeout_cnt = 0 

    with ProcessPool(max_workers=8) as pool:
        future = pool.map(math_equal_process, params, timeout=3)
        iterator = future.result()
        with tqdm(total=len(samples), desc="Evaluate") as progress_bar:
            while True:
                try:
                    result = next(iterator)
                    scores.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    scores.append(False)
                    timeout_cnt += 1
                except Exception as error:
                    exit()
                progress_bar.update(1) 

    mean_score = np.mean(scores) * 100
    coverage = np.mean(coverage_scores) * 100

    result_json = {
        "num_samples": len(samples),
        "num_scores": len(scores),
        "timeout_samples": timeout_cnt,
        "acc": mean_score,
        "cov": coverage
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
    data = {"n": [], "acc_naive": [], "acc_weighted": [], "acc_maj": [], "cov": []}

    def evaluate_for_n(n):
        local_data = {"n": n, "acc_naive": None, "acc_weighted": None, "acc_maj": None, "cov": None}
        for agg in ["naive", "weighted", "maj"]:
            _, scores = evaluate(
                benchmark=args.benchmark,
                data_file=args.data_file,
                dataset_col=f"pred_{agg}@{n}",
                max_num_samples=args.max_num_samples,
            )
            local_data[f"acc_{agg}"] = scores["acc"]
            if agg == "naive":  # Only need to get coverage once
                local_data["cov"] = scores["cov"]
        return local_data

    with tqdm(total=len(args.voting_n), desc="Evaluating voting_n") as progress_bar:
        for n in args.voting_n:
            try:
                result = evaluate_for_n(n)
                data["n"].append(result["n"])
                data["acc_naive"].append(result["acc_naive"]) 
                data["acc_weighted"].append(result["acc_weighted"])
                data["acc_maj"].append(result["acc_maj"])
                data["cov"].append(result["cov"])
            except Exception as e:
                print(f"Error processing n={n}: {e}")
            progress_bar.update(1)  

    ### save to same directory as data_file, under results.jsonl
    save_dir = os.path.dirname(args.data_file)
    with open(os.path.join(save_dir, "metrics.jsonl"), "w") as f:
        json.dump(data, f)
