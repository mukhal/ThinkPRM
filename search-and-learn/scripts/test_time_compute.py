#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import pathlib
import torch
from vllm import LLM
import sys 
import os
import json

sys.path.append('./')
sys.path.append('./search-and-learn')
sys.path.append('./search-and-learn/src')

from sal.config import Config
from sal.search import beam_search, best_of_n, dvts
from sal.utils.data import get_dataset, save_dataset
from sal.utils.parser import H4ArgumentParser
from sal.utils.score import score as score_dataset
from sal.models.reward_models import MathShepherd, RLHFFlow, QwenPRM

from prm import DiscriminativePRM
from prm.thinkprm_api import APIThinkPRMVerifier

def load_prm(config: Config):
    if config.prm_path == "peiyi9979/math-shepherd-mistral-7b-prm":
        return MathShepherd(config)

    if config.prm_path == "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data":
        return RLHFFlow(config)
    
    if config.prm_path == "Qwen/Qwen2.5-Math-PRM-7B":
        return QwenPRM(config)
        
    elif config.prm_type == "thinkprm":
        prm = APIThinkPRMVerifier(
            endpoint=config.verifier_endpoint,
            max_length=config.max_verification_length,
            n=config.prm_n,
            temperature=config.prm_temperature,
            seed=0,
            decision_temperature=config.prm_decision_temperature,
            score_label_idx=config.prm_score_label_idx,
            label_categories=config.prm_label_categories,
            predecision_string=config.prm_predecision_string,
            process_verifier=config.approach == "beam_search",
            long_cot=config.long_cot,
            n_thinking_rounds=config.n_thinking_rounds,
            trigger_phrase=config.trigger_phrase,
            verifier_instruction=config.prm_verifier_instruction,
        )
        print("Initialized ThinkPRM client...")
        
    elif config.prm_type == "discriminative":
        prm = DiscriminativePRM(
            model_name_or_path=config.prm_path,
            device=config.prm_device,
            max_length=int(config.max_tokens * 1.2), # 1.2x longer than max tokens to account for step delimiters. 
            batch_size=config.prm_batch_size,
            long_cot=config.long_cot,
        )
        print("Initialized discriminative PRM client...")
    
    return prm


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


APPROACHES = {
    "beam_search": beam_search,
    "dvts": dvts,
    "best_of_n": best_of_n,
}


def main():
    parser = H4ArgumentParser(Config)
    config = parser.parse()
    
    approach_fn = APPROACHES[config.approach]

    num_gpus = torch.cuda.device_count()
    
    if not config.cached_solutions_path:
        llm = LLM(
            model=config.model_path,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enable_prefix_caching=True,
            max_model_len=16384,
            seed=config.seed,
            tensor_parallel_size=1,
        )
    else:
        llm = None
            
    if config.just_sample:
        prm = None
    else:
        prm = load_prm(config)

    dataset = get_dataset(config)
    problem_to_outputs = None

    if config.cached_solutions_path:
        print(f"Loading cached solutions from {config.cached_solutions_path}")
        print(f"n: {config.n}")
        with open(config.cached_solutions_path, 'r') as f:
            examples = [json.loads(line.strip()) for line in f.readlines()]            
        # Create dict mapping problems to completions and tokens
        problem_to_outputs = {}
        for e in examples:
            if len(e['completions']) < config.n:
                raise ValueError(f"Example {e['problem']} has only {len(e['completions'])} completions instead of {config.n}")
                
            problem_to_outputs[e['problem']] = {
                'completions': e['completions'],
                'completion_tokens': e['completion_tokens']
            }
        
        # Update dataset with cached completions if available
        def update_example(example):
            if example['problem'] in problem_to_outputs:
                outputs = problem_to_outputs[example['problem']]
                assert len(outputs['completions']) >= config.n, f"Cached completions for {example['problem']} have only {len(outputs['completions'])} completions instead of {config.n}"
                example['completions'] = outputs['completions'][:config.n]
                example['completion_tokens'] = outputs['completion_tokens'][:config.n]
                
                assert len(example['completions']) == len(example['completion_tokens']), f"Number of completions {len(example['completions'])} does not match number of completion tokens {len(example['completion_tokens'])}"
            return example
            
        dataset = dataset.map(
            update_example,
            desc="Updating dataset with cached completions",
            load_from_cache_file=False
        )

    dataset = dataset.map(
        approach_fn,
        batched=True,
        batch_size=config.search_batch_size,
        fn_kwargs={"config": config, "llm": llm, "prm": prm},
        desc="Running search",
        load_from_cache_file=False,
    )

    dataset = score_dataset(dataset, config)
    save_dataset(dataset, config)
    logger.info("Done 🔥!")
    
    # Compute voting_n sequence based on config.n
    if config.voting_n:
        voting_n = config.voting_n
    else:
        voting_n = ' '.join([str(2**i) for i in range(int.bit_length(config.n))])
    
    # Run evaluation script
    if config.approach == "best_of_n":
        os.system(f"python search-and-learn/evaluation/evaluate_file.py --data_file {config.output_dir}/best_of_n_completions.jsonl --voting_n {voting_n}")
    elif config.approach == "beam_search":
        os.system(f"python search-and-learn/evaluation/evaluate_file.py --data_file {config.output_dir}/beam_search_completions.jsonl --voting_n {voting_n}")
        
    ### save verifications 
    if hasattr(prm, 'all_verifications'):
        print("💾 Saving verifications to ", config.output_dir)
        # Create parent directory if it doesn't exist
        save_to_path = os.path.join(config.output_dir, "verifications.jsonl")
        parent_dir = os.path.dirname(save_to_path)
        if parent_dir:
            pathlib.Path(parent_dir).mkdir(parents=True, exist_ok=True)
            
        with open(save_to_path, 'w') as f:
            for verification in prm.all_verifications:
                score, info = verification
                record = {
                    "input": info['input_text'],
                    "score": score,
                    "verifications": info['output_texts']
                }
                f.write(json.dumps(record) + "\n")
                
    # Save config to output directory
    config_save_path = os.path.join(config.output_dir, "config.json")
    with open(config_save_path, 'w') as f:
        # Convert config object to dict, excluding any non-serializable items
        config_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}
        json.dump(config_dict, f, indent=2)
        
    # Clean exit
    sys.exit(0)

if __name__ == "__main__":
    main()
