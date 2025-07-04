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
from netcal.metrics import ECE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import roc_auc_score

from grader import *
from parser import *
from utils import load_jsonl
from python_executor import PythonExecutor

def evaluate_calibration(benchmark: str, data_file: str, dataset_col: str = "pred", samples: list=None, max_num_samples=None, normalize=False, confidence_threshold=0.5, n_bins=10):
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
    samples = samples.map(extract_answer_map, fn_kwargs={"data_name": benchmark, "col": dataset_col}, desc="Parsing predictions", num_proc=1, load_from_cache_file=False)
    
    # Get correctness and confidence scores
    correctness = []
    confidence_scores = []
    timeout_cnt = 0
    
    # Flatten the samples into (idx, completion, gt) tuples
    params = []
    for idx, completions, gt in zip(samples['idx'], samples['completions'], samples['gt']):
        for completion in completions:
            pred_answer = extract_answer(completion, data_name=benchmark)
            params.append((idx, pred_answer, gt))
            
    # Process samples in parallel
    with ProcessPool(max_workers=16) as pool:
        future = pool.map(math_equal_process, params, timeout=3)
        iterator = future.result()
        
        with tqdm(total=len(params), desc="Evaluating calibration") as progress_bar:
            for i in range(len(params)):
                try:
                    result = next(iterator)
                    solution_correctness = result
                except TimeoutError:
                    solution_correctness = False 
                    timeout_cnt += 1
                except StopIteration:
                    break
                except Exception:
                    exit()
                    
                # Get confidence score for this completion
                sample_idx = i // len(samples[0]['completions'])
                completion_idx = i % len(samples[0]['completions'])
                confidence = samples[sample_idx]['agg_scores'][completion_idx]
                confidence_scores.append(confidence)
                
                # Compute verifier correctness: verifier is correct if it correctly identifies solution correctness
                # (confidence > threshold for correct solutions, confidence < threshold for incorrect solutions)
                #verifier_correctness = (solution_correctness and confidence > confidence_threshold) or (not solution_correctness and confidence < confidence_threshold)
                #verifier_correctness = solution_correctness
                correctness.append(solution_correctness)
                
                progress_bar.update(1)
    
    assert len(correctness) == len(confidence_scores)

    # Compute calibration metrics using netcal
    correctness = np.array(correctness, dtype=float)
    confidence_scores = np.array(confidence_scores)
    
    # Normalize confidence scores if requested
    if normalize and len(confidence_scores) > 0:
        max_confidence = np.max(confidence_scores)
        if max_confidence > 0:
            confidence_scores = confidence_scores / max_confidence
    
    # Expected Calibration Error
    ece = ECE(n_bins)
    ece_score = ece.measure(confidence_scores, correctness)
    
    # Compute AUC-ROC
    auc_roc = roc_auc_score(correctness, confidence_scores) if len(np.unique(correctness)) > 1 else 0.5
    
    # Create calibration plot
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidence_scores, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    
    bin_accuracies = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for i in range(len(correctness)):
        bin_idx = bin_indices[i]
        bin_accuracies[bin_idx] += correctness[i]
        bin_counts[bin_idx] += 1
    
    # Calculate average accuracy and confidence per bin
    for i in range(n_bins):
        if bin_counts[i] > 0:
            bin_accuracies[i] /= bin_counts[i]
    
    # Create plot data
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    result_json = {
        "num_samples": len(samples),
        "timeout_samples": timeout_cnt,
        "ece": float(ece_score),
        "auc_roc": float(auc_roc),
        "bin_centers": bin_centers.tolist(),
        "bin_accuracies": bin_accuracies.tolist(),
        "bin_counts": bin_counts.tolist(), 
        "bin_edges": bin_edges.tolist()
    }
    
    return samples, result_json

def plot_calibration(data, save_path):
    """Create and save a calibration plot showing accuracy vs confidence by bin."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.figure(figsize=(10, 6))
    
    # Use the last n value for the plot
    n = data["n"][-1]
    ece = data["ece_weighted"][-1]
    auc_roc = data["auc_roc_weighted"][-1]
    bin_edges = data[f"bin_edges_{n}"]
    bin_accuracies = data[f"bin_accuracies_{n}"]
    bin_counts = data[f"bin_counts_{n}"]
    bin_centers = data[f"bin_centers_{n}"]
    
    # Plot bars for accuracy using bin centers
    width = (bin_edges[1] - bin_edges[0]) * 0.95  # 95% of bin width
    plt.bar(bin_centers, bin_accuracies, width=width, alpha=0.7, label=f'ECE={ece:.3f}, AUC-ROC={auc_roc:.3f}')
    
    # Plot perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Calibration Plot')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(np.linspace(0, 1, 11))  # Set x-ticks from 0 to 1
    plt.ylim(0, 1.1)  # Set y-axis limit to accommodate annotations
    plt.savefig(save_path)
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="math")
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--max_num_samples", type=int, default=None)
    parser.add_argument("--voting_n", type=int, nargs='+', required=True)
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--normalize", action="store_true", help="Normalize confidence scores by dividing by the maximum")
    parser.add_argument("--confidence_threshold", type=float, default=0.5, help="Threshold for confidence scores to determine correctness")
    
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    data = {"n": [], "ece_naive": [], "ece_weighted": [], "ece_maj": [], "auc_roc_naive": [], "auc_roc_weighted": [], "auc_roc_maj": []}

    def evaluate_for_n(n):
        local_data = {"n": n, "ece_naive": None, "ece_weighted": None, "ece_maj": None, "auc_roc_naive": None, "auc_roc_weighted": None, "auc_roc_maj": None}
        for agg in ["weighted"]:
            _, scores = evaluate_calibration(
                benchmark=args.benchmark,
                data_file=args.data_file,
                dataset_col=f"pred_{agg}@{n}",
                max_num_samples=args.max_num_samples,
                normalize=args.normalize,
                confidence_threshold=args.confidence_threshold,
                n_bins=args.n_bins,
            )
            local_data[f"ece_{agg}"] = scores["ece"]
            local_data[f"auc_roc_{agg}"] = scores["auc_roc"]
            # Store bin data for plotting
            data[f"bin_centers_{n}"] = scores["bin_centers"]
            data[f"bin_accuracies_{n}"] = scores["bin_accuracies"]
            data[f"bin_counts_{n}"] = scores["bin_counts"]
            data[f"bin_edges_{n}"] = scores["bin_edges"]
        
        return local_data

    with tqdm(total=len(args.voting_n), desc="Evaluating calibartion voting_n") as progress_bar:
        for n in args.voting_n:
            try:
                result = evaluate_for_n(n)
                data["n"].append(result["n"])
                data["ece_weighted"].append(result["ece_weighted"])
                data["auc_roc_weighted"].append(result["auc_roc_weighted"])
            except Exception as e:
                print(f"Error processing n={n}: {e}")
            progress_bar.update(1)
            
    print("ECE: ", data)
    
    # Create and save calibration plot
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        plot_path = os.path.join(args.save_dir, "calibration_plot.png")
        plot_calibration(data, plot_path)
        print(f"Calibration plot saved to {plot_path}")
        
        # Save metrics as JSON
        metrics_path = os.path.join(args.save_dir, "calibration_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Calibration metrics saved to {metrics_path}")
