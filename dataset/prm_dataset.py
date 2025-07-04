import json, os, pickle as pkl
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from dataclasses import dataclass
import random
import re
from utils.helper import format_verification_cot_for_thinkprm, format_verification_cot_no_think, format_train_verification_cot_for_thinkprm

def clean_traj(traj_str):
    ## remove all that comes after "The answer is"
    ### remove 'Step x:'
    traj_str = re.sub(r'\nStep \d+:', '', traj_str).strip()
    traj_str = re.sub(r'Step \d+:', '', traj_str).strip()
    traj_str = traj_str.replace('The answer is:', 'The answer is').strip()
    ## make sure there is a dot before "The answer is". If there is, do nothing
    traj_str = traj_str.replace('. The answer is', ' The answer is').replace(' The answer is', '. The answer is')
    ## remove all substrings that are <<xxx>> 
    traj_str = re.sub(r'<<.*?>>', '', traj_str).strip()
    traj_str = traj_str.split('\n\n')[0].strip()
    
    return traj_str


class PRMTrajectoryDataset(Dataset):
    def __init__(self, 
                 data_path: str, 
                 tokenizer, 
                 config=None, 
                 split='train'
                 ):
        
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.max_length
        self.split = split
        self.step_sep = ' ки'
        self.pos_step_token = '+'
        self.neg_step_token = '-'
        self.step_sep_id = self.tokenizer.encode(self.step_sep, add_special_tokens=False)[-1]
        self.pos_step_id = self.tokenizer.encode(self.pos_step_token, add_special_tokens=False)[-1]
        self.neg_step_id = self.tokenizer.encode(self.neg_step_token, add_special_tokens=False)[-1]
        self.num_samples = getattr(self.config, 'num_samples', 100000)
    
        self.data = self._load_data()
            
    def _load_data(self):
        raise NotImplementedError
            
    def process_data(self, data):
        raise NotImplementedError

    
    @staticmethod
    def tokenize_example(example, tokenizer, step_sep_id, pos_step_id, neg_step_id, max_length, config, split, add_step_tokens=True):
        question = example['question']
        steps_with_labels = example['steps_with_labels']
        # Tokenize question
        question_tokens = tokenizer.encode(question , add_special_tokens=False)
        
        input_ids = []
        labels = []
        loss_mask = []
        loss_mask_with_first_error_only = []
        # Add question tokens
        input_ids.extend(question_tokens)
        labels.extend([tokenizer.pad_token_id] * len(question_tokens))
        loss_mask.extend([0] * len(question_tokens))
        loss_mask_with_first_error_only.extend([0] * len(question_tokens)) ## Needed to compute TRACE error.
        after_first_error = False
        
        if hasattr(config, 'to_dict'):
            config = config.to_dict()
            
        # Process each step
        for step_idx, step_info in enumerate(steps_with_labels):
            step = step_info['step']
            step_label = step_info['label']

            if add_step_tokens:
                step = f'\nStep {step_idx+1}: {step}'
            
            # Tokenize step
            step_tokens = tokenizer.encode(step, add_special_tokens=False)
            input_ids.extend(step_tokens)
            labels.extend([tokenizer.pad_token_id] * len(step_tokens))
            loss_mask.extend([0] * len(step_tokens))
            loss_mask_with_first_error_only.extend([0] * len(step_tokens))
            
            # Add step separator
            input_ids.append(step_sep_id)
            labels.append(pos_step_id if step_label in ['+', 1] else neg_step_id)
            loss_mask.append(1 if not config.get('full_prefix_only', False) else (1 if step_idx == len(steps_with_labels) - 1 else 0))

            if not after_first_error:
                loss_mask_with_first_error_only.append(1)
            else:
                loss_mask_with_first_error_only.append(0)

            if labels[-1] == neg_step_id and not after_first_error:
                after_first_error = True

        assert len(input_ids) == len(labels) == len(loss_mask), "Input ids, labels, and loss mask should be the same length"
        assert len(input_ids) == len(loss_mask_with_first_error_only), "Input ids and loss mask with first error only should be the same length"
        # Convert to tensors
        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        loss_mask = torch.tensor(loss_mask)
        loss_mask_with_first_error_only = torch.tensor(loss_mask_with_first_error_only)
        attention_mask = torch.ones_like(input_ids)

        # Truncate if necessary
        if len(input_ids) > max_length:
            #print("Warning: truncating input ids from {} to {}".format(len(input_ids), max_length))
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
            loss_mask = loss_mask[:max_length]
            loss_mask_with_first_error_only = loss_mask_with_first_error_only[:max_length]
            attention_mask = attention_mask[:max_length]

        if getattr(config, 'distribute_final_answer_labels', False) and split == 'train':
            solution_label = example['solution_label']
            labels[loss_mask == 1] = pos_step_id if solution_label == 1 else neg_step_id
            
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'loss_mask': loss_mask,
            'loss_mask_with_first_error_only': loss_mask_with_first_error_only
        }


    def _tokenize_example(self, example):
        return self.tokenize_example(example, self.tokenizer, self.step_sep_id, self.pos_step_id, self.neg_step_id, self.max_length, self.config, self.split)

    def __getitem__(self, idx):
        example = self.data[idx]
        return self._tokenize_example(example)

    def collate_fn(self, batch):
        input_ids = pad_sequence([b['input_ids'] for b in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence([b['attention_mask'] for b in batch], batch_first=True, padding_value=0)
        labels = pad_sequence([b['labels'] for b in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

        if 'loss_mask' in batch[0]:
            loss_mask = pad_sequence([b['loss_mask'] for b in batch], batch_first=True, padding_value=0)
            loss_mask_with_first_error_only = pad_sequence([b['loss_mask_with_first_error_only'] for b in batch], batch_first=True, padding_value=0)
            return_dict['loss_mask'] = loss_mask
            return_dict['loss_mask_with_first_error_only'] = loss_mask_with_first_error_only

        return return_dict
    
    def __len__(self):
        return len(self.data)
    
    def clean_trajectory(self, traj):
        return traj


class PRMCoTDataset(PRMTrajectoryDataset):
    def __init__(self, 
                 data_path: str, 
                 tokenizer, 
                 config=None, 
                 split='train',
                 ):
        super().__init__(data_path=data_path, tokenizer=tokenizer, config=config, split=split)
        if getattr(self.config, 'train_with_gold_solutions', False):
            print("🤯🤯🤯 Training with gold solutions too 🤯🤯🤯")
        self.data = self.process_data(self.data)
        
    def _load_data(self):
        ## load all .json files in the data_path
        data = []
        data_dirs = self.config.data_dir if isinstance(self.config.data_dir, list) else [self.config.data_dir]
        
        for data_dir in data_dirs:
            for file in os.listdir(data_dir):
                if 'all_outputs' in file:
                    with open(os.path.join(data_dir, file), 'r') as f:
                        loaded_data = json.load(f)
                        data.extend(loaded_data)
    
        print(f"Loaded {len(data)} initial data points")
            
        return data
    
    def fix_step_labels(self, step_labels):
        ## Find first '+' and turn any '-' before it into '+'
        fixed_labels = step_labels.copy()
        # Find the last '+' label
        step_labels = ''.join(step_labels)
        last_plus_idx = step_labels.rfind('+')
        
        if last_plus_idx == -1: # No '+' found, return original labels
            return fixed_labels

        # set all labels before the last '+' to '+'
        for i in range(last_plus_idx):
            fixed_labels[i] = '+'
                
        return fixed_labels

    def process_data(self, data):
        ## data is a dictionary where keys are question@@@prefix and values future sampling information. We want to process such that each prefix will be
        processed_data = []
        n_skipped_bad_format = 0
        n_skipped_bad_decisions = 0
        n_total = 0
        max_cots_per_solution = self.config.max_cots_per_solution

        for example in data:
            problem = example['problem']
            prefix = example['prefix']
            gt_labels = example['traj_gt_labels'] # list of '+' or '-'
            gt_labels = ['+' if lbl in ['+', 1] else '-' for lbl in gt_labels]
            gt_labels = self.fix_step_labels(gt_labels)
            gold_traj = [l for l in example['prompt'].split('\n') if l.startswith('Correct solution')][-1].replace('Correct solution:', '').strip()
                        
            ## get gt_labels until and including the first '-' if there is one. If there's no '-' get all of them
            if not gt_labels:
                continue

            steps = prefix.split('\n')
            is_correct = gt_labels[-1] == '+'
            n_trajs_so_far = 0
            
            example['generations'] = list(set(example['generations'])) # remove duplicates if any

            for generation in example['generations']:
                # Extract the final decision from the generation
                n_total += 1
                decisions = [s for s in generation.split('\n') if 'correct? yes' in s.lower() or 'correct? no' in s.lower()]
                if '-' in gt_labels:
                    labels_until_error = gt_labels[:gt_labels.index('-') + 1]
                else:
                    labels_until_error = gt_labels
                    
                
                valid_decisions = [d for d in decisions if 'correct? yes' in d.lower() or 'correct? no' in d.lower()]
                
                if len(valid_decisions) != len(steps):
                    n_skipped_bad_format += 1
                    continue
                
                decisions = ['+' if 'correct? yes' in d.lower() else '-' for d in valid_decisions]
                
                # Check if the decision matches all ground truth labels until the first '-'
                if all([(decision == gt_label) for decision, gt_label in zip(decisions, labels_until_error)]):
                    n_trajs_so_far += 1
                    if n_trajs_so_far > max_cots_per_solution:
                        break
                    
                    ### process cot, but finding the Step x:, where x is len(labels_until_error) + 1 and replacing the rest with Step x+1: The step is incorrect since it follows an incorrect step. 
                    incorrect_step_index = len(labels_until_error) - 1
                    cot_steps = [line for line in generation.split('\n') if line.startswith('Step')]
                    ## remove steps after incorrect_step_index
                    cot_steps = cot_steps[:incorrect_step_index+1]
                    ## add the rest of the steps with the cot "The step is incorrect since it follows an incorrect step."
                    for _ in range(len(steps) - len(cot_steps)):
                        cot_steps.append(f'Step {len(cot_steps)+1}: Follows an incorrect step. Correct? No.')

                    assert len(cot_steps) == len(steps) # sanity check
                    
                    if getattr(self.config, 'cot_incorrect_only', False):
                        cot_steps = [re.sub(r'(Step .*:).*?(Correct\? Yes)', r'\1 \2', step) for step in cot_steps]

                    if getattr(self.config, 'direct_prm', False):
                        cot_steps = [re.sub(r'(Step .*:).*?(Correct\? (Yes|No))', r'\1 \2', step) for step in cot_steps]
                        
                    if getattr(self.config, 'single_label', False):
                        ### we will have a single YES/NO label for the entire solution
                        cot_steps = [step.replace('Correct? Yes.', '').replace('Correct? No.', '') for step in cot_steps]
                        cot_steps += ['Is the solution correct? Yes' if is_correct else 'Is the solution correct? No']
                                                
                    cot = '\n'.join(cot_steps)  
                    
                    labels = labels_until_error + ['-'] * (len(steps) - len(labels_until_error))
                                        
                    processed_data.append({
                        'problem': problem,
                        'solution': prefix,
                        'cot': cot,
                        'solution_steps': steps,
                        'cot_steps': cot_steps,
                        'labels': labels,
                        'is_correct': is_correct,
                    })
                    
                    if getattr(self.config, 'train_with_gold_solutions', False) and random.random() < 0.3: # 30% of the time we will include the gold solution
                        ## duplicate the example but with the gold solution included
                        processed_data.append({
                            'problem': problem,
                            'solution': prefix,
                            'cot': cot,
                            'solution_steps': steps,
                            'cot_steps': cot_steps,
                            'labels': labels,
                            'is_correct': is_correct,
                            'gold_traj': gold_traj
                        })
                        
                    if getattr(self.config, 'add_partial_prefixes', False) and random.random() < 0.05 and len(steps) >= 3:
                        ## will add anothjer example with a partial prefix cut at random step to teach the model to work with incomplete solutions
                        ## truncate the prefix to 10 steps
                        k = random.randint(1, len(steps) - 1) # don't cut at the last step
                        partial_prefix = '\n'.join(steps[:k])
                        if getattr(self.config, 'single_label', False):
                            # remove the "Is the solution correct? Yes" or "Is the solution correct? No" part
                            cot_steps = cot_steps[:-1]
                        
                        partial_cot_steps = cot_steps[:k] 
                        partial_labels = labels[:k]
                        
                        assert len(partial_labels) == len(partial_cot_steps)

                        if getattr(self.config, 'single_label', False):
                            partial_cot_steps.append('Is the solution correct? Yes' if partial_labels[-1] == '+' else 'Is the solution correct? No')
                                                              
                        processed_data.append({
                            'problem': problem,
                            'solution': partial_prefix,
                            'cot': '\n'.join(partial_cot_steps),
                            'solution_steps': steps[:k],
                            'cot_steps': partial_cot_steps,
                            'labels': partial_labels,
                            'is_correct': partial_labels[-1] == '+',
                        })
                else:
                    n_skipped_bad_decisions += 1

        print(f"Skipped {n_skipped_bad_format}/{n_total} examples due to extraction errors")
        print(f"Skipped {n_skipped_bad_decisions}/{n_total} examples due to incorrect decisions")
        
        if self.config.balance_data: # TODO revisit this
            # Count number of correct/incorrect examples
            correct_examples = []
            incorrect_examples = []
            for example in processed_data:
                if example['is_correct']:
                    correct_examples.append(example)
                else:
                    incorrect_examples.append(example)
            
            correct_count = len(correct_examples)
            incorrect_count = len(incorrect_examples)
            
            print(f"Before balancing - Correct examples: {correct_count}, Incorrect examples: {incorrect_count}")
            
            # Determine majority class and target count
            majority_examples = correct_examples if correct_count > incorrect_count else incorrect_examples
            minority_examples = incorrect_examples if correct_count > incorrect_count else correct_examples
            target_count = min(correct_count, incorrect_count)
            
            # Sample from majority class to match minority class size
            sampled_majority = random.sample(majority_examples, target_count)
            
            # Create balanced dataset
            balanced_data = minority_examples + sampled_majority
            
            # Recount after balancing
            correct_count = len([ex for ex in balanced_data if ex['is_correct']])
            incorrect_count = len(balanced_data) - correct_count
            
            print(f"After balancing - Correct examples: {correct_count}, Incorrect examples: {incorrect_count}")
            
            processed_data = balanced_data
                                
        return processed_data

    def format_cot_data(self, problem, solution, cot=None):
        return format_verification_cot_no_think(self.tokenizer, problem, solution, cot=cot)
            
    def format_gold_solution_cot_data(self, problem, gold_solution, solution, cot=None):
        instruction = ("Given a math question, a correct solution, and a proposed solution, analyze each step in the proposed solution, then determine whether it is correct. Provide the analysis for each step first, then indicate with 'Yes' or 'No' whether it is correct.")
        if cot:
            return self.tokenizer.apply_chat_template([
                {'role': "user", "content": f"{instruction}\n\nQuestion: {problem}\n\nCorrect solution:\n{gold_solution}\n\nproposed Solution:\n{solution}"},
                {'role': "assistant", "content": f"Analysis:\n{cot}"}
            ], tokenize=False)
        else:
            s = self.tokenizer.apply_chat_template([
                {'role': "user", "content": f"{instruction}\n\nQuestion: {problem}\n\nCorrect solution:\n{gold_solution}\n\nProposed solution:\n{solution}"},
                {'role': "assistant", "content": ""}
            ], tokenize=False, add_generation_prompt=False)
            return s.replace(self.tokenizer.eos_token, '')
    
    def _tokenize_example(self, example):
        ret_dict = {}
        if 'cot' in example:  # Training example
            # Combine instruction, problem, prefix, and COT
            if getattr(self.config, 'train_with_gold_solutions', False) and 'gold_traj' in example:
                input_text = self.format_gold_solution_cot_data(example['problem'], example['gold_traj'], example['solution'], example['cot'])
            else:
                input_text = self.format_cot_data(example['problem'], example['solution'], example['cot'])
            tokenized = self.tokenizer(input_text, padding=True, truncation=False, return_tensors='pt', add_special_tokens=False)
            
            if len(tokenized['input_ids'][0]) > self.max_length:
                print(f"Truncating input_ids because it's too long: {len(tokenized['input_ids'][0])}")
                input_ids = tokenized['input_ids'][0][:self.max_length]
                attention_mask = tokenized['attention_mask'][0][:self.max_length]
            else:
                input_ids = tokenized['input_ids'][0]
                attention_mask = tokenized['attention_mask'][0]

            if "<|im_start|>assistant" in input_text:
                cot_start = input_text.index("<|im_start|>assistant") + len("<|im_start|>assistant")
            elif "<｜Assistant｜>" in input_text:
                cot_start = input_text.index("<｜Assistant｜>") + len("<｜Assistant｜>")
            else:
                raise ValueError("No <|im_start|>assistant found in the input text")
                        
            cot_tokens = self.tokenizer(input_text[cot_start:], return_tensors='pt')['input_ids'][0]
            # Set loss mask to 1 for the COT tokens

            # Create labels
            labels = torch.full_like(input_ids, -100)  # -100 is the ignore index for CrossEntropyLoss
            labels[-len(cot_tokens):] = input_ids[-len(cot_tokens):]

            ret_dict = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
            }
        else:  # Evaluation example
            solution = example['solution']
            input_text = self.format_cot_data(example['problem'], solution)

            tokenized = self.tokenizer(input_text, padding=True, return_tensors='pt', max_length=self.max_length, add_special_tokens=False)
            input_ids = tokenized['input_ids'][0]
            attention_mask = tokenized['attention_mask'][0]

            ret_dict = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'input_text': input_text,
            }

            if 'traj_gt_labels' in example: # eval
                step_labels = [1 if lbl in ['+', 1] else 0 for lbl in example['traj_gt_labels']]
                step_labels = torch.tensor(step_labels)
                ret_dict['step_labels'] = step_labels
        
        return ret_dict
        
    def collate_fn(self, batch):
        return_dict = {}
        keys_to_pad = ['input_ids', 'attention_mask']
        
        for key in keys_to_pad:
            return_dict[key] = pad_sequence([b[key] for b in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id if key == 'input_ids' else 0)
        
        if 'labels' in batch[0]:
            return_dict['labels'] = pad_sequence([b['labels'] for b in batch], batch_first=True, padding_value=-100)
        elif 'step_labels' in batch[0]:
            return_dict['step_labels'] = [b['step_labels'] for b in batch]

        if 'input_text' in batch[0]:
            return_dict['input_text'] = [b['input_text'] for b in batch]

        return return_dict


class LongThoughtCritiqueDataset(PRMCoTDataset):
    def __init__(self, 
                 data_path: str, 
                 tokenizer, 
                 config=None, 
                 split='train'):
        self.predecision_string = 'Is the solution correct?'
        if hasattr(config, 'add_think_token'):
            self.add_think_token = config.add_think_token
        else:
            self.add_think_token = '<think>' in tokenizer.vocab
            
        print(f"Adding think 🤔 token? {self.add_think_token}")
        super().__init__(data_path=data_path, tokenizer=tokenizer, config=config, split=split)

    def extract_boxed_predictions(self, solution_text: str):
        solution_text = re.sub(r'\\text\{([^}]*)\}', r'\1', solution_text)
        matches = re.findall(r'\\boxed\{([^}]*)\}', solution_text)
        return [match.strip() for match in matches if match in ['correct', 'incorrect']]
        
    def process_data(self, data):
        processed_data = []
        n_skipped_bad_format = 0
        n_skipped_bad_decisions = 0
        n_skipped_bad_length = 0
        n_total = 0
        max_cots_per_solution = self.config.max_cots_per_solution
        match_all = getattr(self.config, 'match_all_step_labels', True)
        
        for example in data:
            problem = example['problem']
            prefix = example['prefix']
            steps = prefix.split('\n')
            
            gt_labels = example['traj_gt_labels']
            gt_labels = ['+' if lbl in ['+', 1] else '-' for lbl in gt_labels]
            gt_labels = self.fix_step_labels(gt_labels)
            
            if not gt_labels:
                continue

            labels_until_error = gt_labels[:gt_labels.index('-') + 1] if '-' in gt_labels else gt_labels
            is_correct = gt_labels[-1] == '+'
            n_trajs_so_far = 0
            
            for generation in example['generations']:
                n_total += 1
                decisions = self.extract_boxed_predictions(generation)
                decisions = ['+' if d == 'correct' else '-' for d in decisions]
                
                if not decisions:
                    n_skipped_bad_format += 1
                    continue
                
                if len(decisions) != len(labels_until_error) and len(decisions) != len(steps):
                    n_skipped_bad_format += 1
                    continue
               
                if not match_all:
                    # Looser filtering - only compare final decisions
                    if decisions[-1] != gt_labels[-1] or 'The answer is' not in generation: # outcome-based, so we need the outcome i.e., answer to be there
                        n_skipped_bad_decisions += 1
                        continue
                
                elif not all(decision == gt_label for decision, gt_label in zip(decisions, labels_until_error)): # stricter filtering based on process labels (used in the paper)
                        n_skipped_bad_decisions += 1
                        continue
                    
                if n_trajs_so_far > max_cots_per_solution:
                    break
                                    
                # Clean the generation
                generation = generation.replace('\\boxed{\\text{correct}}', '\\boxed{correct}').replace('\\boxed{\\text{incorrect}}', '\\boxed{incorrect}')
                
                # Clean text after the last decision
                last_boxed_index = -1
                for match in re.finditer(r'\\boxed\{[^}]*\}', generation):
                    boxed_content = match.group(0)
                    if any(decision in boxed_content.lower() for decision in ['correct', 'incorrect']):
                        last_boxed_index = match.end()
                
                if last_boxed_index != -1:
                    generation = generation[:last_boxed_index]
                    
                if '<think>' in generation:
                    generation = generation[:generation.index('<think>') + len('<think>')]
                    
                if '</think>' in generation:
                    generation = generation[:generation.index('</think>')]
                
                if getattr(self.config, 'filter_based_on_length', False) and len(self.tokenizer(generation, return_tensors='pt')['input_ids'][0]) > self.config.max_length:
                    n_skipped_bad_length += 1
                    continue
                
                if self.add_think_token:
                    generation = f"<think>\n{generation}\n</think>"
                        
                generation += f"\n{self.predecision_string} {'Yes' if is_correct else 'No'}"
                                                                
                processed_data.append({
                    'problem': problem,
                    'solution': prefix,
                    'cot': generation,
                    'solution_steps': steps,
                    'cot_steps': generation.split('\n'),
                    'labels': gt_labels,
                    'is_correct': is_correct,
                })
                
                # Add partial prefixes if configured
                if self._should_add_partial_prefix(steps):
                    partial_example = self._create_partial_prefix_example(steps, generation, gt_labels)
                    if partial_example:
                        processed_data.append(partial_example)
                
        print(f"Skipped {n_skipped_bad_format}/{n_total} examples due to extraction errors")
        print(f"Skipped {n_skipped_bad_decisions}/{n_total} examples due to incorrect decisions")
        print(f"Skipped {n_skipped_bad_length}/{n_total} examples due to length")
        print(f"Got {len(processed_data)} chains from {len(set([ex['problem'] for ex in processed_data]))} unique questions")
        
        # Balance data if configured
        if self.config.balance_data:
            processed_data = self._balance_data(processed_data)
                    
        # Limit number of samples if configured
        if getattr(self.config, 'num_samples', None):
            ## shuffle and take first num_samples
            random.seed(42)
            random.shuffle(processed_data)
            processed_data = processed_data[:self.config.num_samples]
                                
        return processed_data
    
    def _should_add_partial_prefix(self, steps):
        return (getattr(self.config, 'add_partial_prefixes', False) 
                and random.random() < 0.1 
                and len(steps) >= 4)
    
    def _create_partial_prefix_example(self, steps, generation, gt_labels):
        k = random.randint(1, len(steps) - 1)  # Don't cut at the last step
        partial_prefix = '\n'.join(steps[:k])
        
        # Extract all lines containing boxed decisions up to k-th decision
        cot_lines = []
        decision_count = 0
        for line in generation.split('\n'):
            if decision_count >= k:
                break
            cot_lines.append(line)
            if '\\boxed{' in line and any(d in line.lower() for d in ['correct', 'incorrect']):
                decision_count += 1
                
        if k > decision_count:
            return None
                
        partial_cot = '\n'.join(cot_lines)
        partial_labels = gt_labels[:k]
        
        # Verify decision count matches step count
        n_decisions = sum(1 for line in partial_cot.split('\n') 
                          if '\\boxed{' in line and any(d in line.lower() for d in ['correct', 'incorrect']))
        if n_decisions != k:
            print(f"Number of decisions ({n_decisions}) does not match the number of steps ({k})")
        
        if self.add_think_token:
            partial_cot = f"<think>\n{partial_cot}\n</think>"
            
        # Add final prediction line
        partial_cot += f"\n{self.predecision_string} {'Yes' if partial_labels[-1] == '+' else 'No'}"
                                                
        return {
            'problem': steps[0],  # First step usually contains the problem
            'solution': partial_prefix,
            'cot': partial_cot,
            'solution_steps': steps[:k],
            'cot_steps': partial_cot.split('\n'),
            'labels': partial_labels,
            'is_correct': partial_labels[-1] == '+',
        }
    
    def _balance_data(self, data):
        # Separate correct and incorrect examples
        correct_examples = [ex for ex in data if ex['is_correct']]
        incorrect_examples = [ex for ex in data if not ex['is_correct']]
        
        correct_count = len(correct_examples)
        incorrect_count = len(incorrect_examples)
        
        print(f"Before balancing - Correct examples: {correct_count}, Incorrect examples: {incorrect_count}")
        
        # Balance the dataset
        target_count = min(correct_count, incorrect_count)
        majority_examples = correct_examples if correct_count > incorrect_count else incorrect_examples
        minority_examples = incorrect_examples if correct_count > incorrect_count else correct_examples
        
        balanced_data = minority_examples + random.sample(majority_examples, target_count)
        
        # Report balancing results
        correct_count = len([ex for ex in balanced_data if ex['is_correct']])
        incorrect_count = len(balanced_data) - correct_count
        
        print(f"After balancing - Correct examples: {correct_count}, Incorrect examples: {incorrect_count}")
        
        return balanced_data
    
    def format_cot_data(self, problem, solution, cot=None):
        if '</think>' in cot:
            return format_train_verification_cot_for_thinkprm(self.tokenizer, problem, solution, cot=cot)
        else:
            return format_verification_cot_for_thinkprm(self.tokenizer, problem, solution, cot=cot)
            

class PRMCoTPairwiseDataset(PRMCoTDataset):
    def __init__(self, 
                 data_path: str, 
                 tokenizer, 
                 config=None, 
                 split='train',
                 ):
        super().__init__(data_path=data_path, tokenizer=tokenizer, config=config, split=split)
        
        print("⚖️ loading pairwise data...")
        self.data = self.prepare_pairwise_data(self.data)
        
    def process_data(self, data):
        ## data is a dictionary where keys are question@@@prefix and values future sampling information. We want to process such that each prefix will be
        processed_data = []
        n_skipped = 0
        n_total = 0
        max_cots_per_solution = self.config.max_cots_per_solution

        for example in data:
            problem = example['problem']
            prefix = example['prefix']
            gt_labels = example['traj_gt_labels'] # list of '+' or '-'
            gt_labels = ['+' if lbl in ['+', 1] else '-' for lbl in gt_labels]
            gt_labels = self.fix_step_labels(gt_labels)
            gold_traj = [l for l in example['prompt'].split('\n') if l.startswith('Correct solution')][-1].replace('Correct solution:', '').strip()
                        
            ## get gt_labels until and including the first '-' if there is one. If there's no '-' get all of them
            if not gt_labels:
                continue

            steps = prefix.split('\n')
            
            chosen_trajs = []
            rejected_trajs = []

            for generation in example['generations']:
                # Extract the final decision from the generation
                n_total += 1
                decisions = [s for s in generation.split('\n') if s.strip() and 'correct?' in s.lower()]
                if '-' in gt_labels:
                    labels_until_error = gt_labels[:gt_labels.index('-') + 1]
                else:
                    labels_until_error = gt_labels

                if len(decisions) != len(steps) or any('correct? yes' not in d.lower() and 'correct? no' not in d.lower() for d in decisions):
                    n_skipped += 1
                    continue

                decisions = ['+' if 'correct? yes' in d.lower() else '-' for d in decisions]
                # Check if the decision matches all ground truth labels until the first '-'
                ### process cot, but finding the Step x:, where x is len(labels_until_error) + 1 and replacing the rest with Step x+1: The step is incorrect since it follows an incorrect step. 
                incorrect_step_index = len(labels_until_error) - 1
                cot_steps = [line for line in generation.split('\n') if line.startswith('Step')]
                ## remove steps after incorrect_step_index
                cot_steps = cot_steps[:incorrect_step_index+1]
                ## add the rest of the steps with the cot "The step is incorrect since it follows an incorrect step."
                for _ in range(len(steps) - len(cot_steps)):
                    cot_steps.append(f'Step {len(cot_steps)+1}: Follows an incorrect step. Correct? No.')

                assert len(cot_steps) == len(steps) # sanity check
                cot = '\n'.join(cot_steps)   
                
                if all([(decision == gt_label) for decision, gt_label in zip(decisions, labels_until_error)]): 
                    chosen_trajs.append(cot)
                else:
                    ## compute how much are the labels different
                    labels_diff = sum(1 for decision, gt_label in zip(decisions, labels_until_error) if decision != gt_label)
                    rejected_trajs.append((labels_diff, cot))
            
            if len(chosen_trajs) == 0 or len(rejected_trajs) == 0:
                continue
            
            ## sort rejected_trajs by labels_diff in descending order
            rejected_trajs.sort(key=lambda x: x[0], reverse=True)
            rejected_trajs = [r[1] for r in rejected_trajs]
            ### take cfg.train.max_cots_per_solution pairs of chosen_trajs and rejected_trajs
            chosen_trajs = chosen_trajs[:max_cots_per_solution]
            rejected_trajs = rejected_trajs[:max_cots_per_solution]

            for chosen_traj, rejected_traj in zip(chosen_trajs, rejected_trajs):
                processed_data.append({
                    'problem': problem,
                    'solution': prefix,
                    'chosen_cot': chosen_traj,
                    'rejected_cot': rejected_traj,
                    'solution_steps': steps,
                })
        
        return processed_data
    
    def prepare_pairwise_data(self, data):
        ### will format each example into a dict of prompt, chosen, rejected
        new_data = []
        for example in data:
            new_data.append({
                'prompt': self.format_cot_data(example['problem'], example['solution']),
                'chosen': example['chosen_cot'],
                'rejected': example['rejected_cot'],
            })
        return new_data

class PRMCoTEvalDataset(PRMCoTDataset):
    def __init__(self, 
                 examples: list, 
                 tokenizer, 
                 config=None, 
                 split='eval',
                 process_data: bool = True,
                 ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.config = config
        self.split = split

        super().__init__(data_path=None, tokenizer=tokenizer, config=config, split=split)
        if process_data:
            self.data = self.process_data(self.examples)
        
        self._validate_data(self.data)

    def _validate_data(self, data):
        for example in data:
            if any(k not in example for k in ['problem', 'traj_gt_labels', 'traj_steps']):
                raise ValueError("Invalid example format")
            
    def _load_data(self):
        return self.examples

    def process_data(self, data):
        processed_data = [] 
        for item in data:
            question = item['question']
            steps = [step_info['step'] for step_info in item['steps_with_labels']]
            traj_gt_labels = [step_info['label'] for step_info in item['steps_with_labels']]

            solution = '\n'.join([f'Step {j+1}: {step}' for j, step in enumerate(steps)])

            cot_example = {
                'problem': question,
                'traj_gt_labels': traj_gt_labels,
                'traj_steps': steps,
                'solution': solution
            }
            processed_data.append(cot_example)

        return processed_data
    