"""
ThinkPRM: Process Reward Models That Think

This module provides implementations of various Process Reward Models (PRMs)
for evaluating step-by-step reasoning processes.

Available PRM types:
- ThinkPRM: Generative PRMs that can think longer and scale compute
- Generative PRM: Standard generative process reward models
- Discriminative PRM: Traditional discriminative process reward models
"""


from typing import Optional
import sys, os
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import torch
from vllm import LLM, SamplingParams
from dataset.prm_dataset import PRMTrajectoryDataset, PRMCoTEvalDataset
from utils.config import Config
from utils.answer_utils import extract_step_cots_and_labels
import re

class DiscriminativePRM: # Discriminative PRM
    def __init__(self,
                model_name_or_path: str,
                step_sep: str = ' ки',
                pos_label_step_token: str = '+',
                neg_label_step_token: str = '-',
                random_reward: bool = False,
                max_length: int = 1024,
                device: str = 'cuda',
                batch_size: int = 8,
                long_cot: bool = False,
                **kwargs
                ) -> None:
        super().__init__()
        
        print("Loading PRM model from {}".format(model_name_or_path))
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16).to(device).eval() # bf16/fp16 might lead to inconsistent results
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.step_sep_id = self.tokenizer.encode(step_sep, add_special_tokens=False)[-1]
        self.pos_step_id = self.tokenizer.encode(pos_label_step_token, add_special_tokens=False)[-1]
        self.neg_step_id = self.tokenizer.encode(neg_label_step_token, add_special_tokens=False)[-1]
        self.random_reward = random_reward
        self.max_length = max_length
        self.batch_size = batch_size
        self.long_cot = long_cot

    def predict_correctness(self, question: str, prefix_steps: list[str]) -> tuple[float, dict]:
        # Tokenize the input
        inputs = self.process_example(question, prefix_steps)
        
        if inputs['input_ids'][-1] != self.step_sep_id:
            print("Warning: step separator not found in the input ids, adding it...")
            inputs['input_ids'] = torch.cat([inputs['input_ids'], torch.tensor([self.step_sep_id])])
            inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.tensor([1])])

        input_ids = inputs['input_ids'].unsqueeze(0)  # Add batch dimension
        attention_mask = inputs['attention_mask'].unsqueeze(0)
        
        # Move tensors to the same device as the model
        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)

        candidate_tokens = [self.pos_step_id, self.neg_step_id]

        # Get model outputs
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask=attention_mask).logits[:, :, candidate_tokens]
            scores = logits.softmax(dim=-1)[:, :, 0] # 1 x seq_len

            step_scores = scores[input_ids == self.step_sep_id] # 1 x 
            step_scores = step_scores.cpu().tolist()

        full_prefix_score = step_scores[-1]
        step_labels = [1 if score > 0.5 else 0 for score in step_scores]
        step_cots = [""] * len(step_labels)
                
        info = {
            'full_prefix_score': full_prefix_score,
            'step_scores': step_scores,
            'step_cots': step_cots,
            'step_labels': step_labels,
            'input_text': inputs['input_text'],
            'output_texts': [""],
        }

        return full_prefix_score, info

    def predict_correctness_batch(self, questions: list[str], prefix_steps_list: list[list[str]]) -> list[tuple[float, dict]]:
        # Tokenize all inputs at once
        batch_inputs = []
        for question, prefix_steps in zip(questions, prefix_steps_list):
            inputs = self.process_example(question, prefix_steps)
            if inputs['input_ids'][-1] != self.step_sep_id:
                print("Warning: step separator not found in the input ids, adding it...")    
                inputs['input_ids'] = torch.cat([inputs['input_ids'], torch.tensor([self.step_sep_id])])
                inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.tensor([1])])
            batch_inputs.append(inputs)

        # Pad sequences to max length in batch
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [inputs['input_ids'] for inputs in batch_inputs], 
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        ).to(self.model.device)
                
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [inputs['attention_mask'] for inputs in batch_inputs],
            batch_first=True,
            padding_value=0
        ).to(self.model.device)
        

        candidate_tokens = [self.pos_step_id, self.neg_step_id]
        
        # Get model outputs for entire batch at once
        with torch.inference_mode():
            logits = self.model(input_ids, attention_mask=attention_mask).logits[:, :, candidate_tokens]
            scores = logits.softmax(dim=-1)[:, :, 0]  # batch_size x seq_len


        # Process results for each example
        results = []
        for i, prefix_steps in enumerate(prefix_steps_list):
            step_mask = (input_ids[i] == self.step_sep_id) & (attention_mask[i] == 1)  # Only consider valid positions
            step_scores = scores[i][step_mask].cpu().tolist()
            if len(step_scores) != len(prefix_steps):
                print("Warning: step scores and prefix steps are not of the same length. This is likely due to a very long chain.")
            full_prefix_score = step_scores[-1]
            
            step_labels = [1 if score > 0.5 else 0 for score in step_scores]
            step_cots = [""] * len(step_labels)
            
            info = {
                'full_prefix_score': full_prefix_score,
                'step_scores': step_scores,
                'step_cots': step_cots,
                'step_labels': step_labels,
                'input_text': self.tokenizer.decode(input_ids[i]),
                'output_texts': [""],
            }
                

            results.append((full_prefix_score, info))

        return results

    def process_example(self, question: str, prefix_steps: list[str]):
        # Prepare the example for tokenization
        
        example = {
            'question': question,
            'steps_with_labels': [{'step': step, 'label': '+'} for step in prefix_steps], # placeholder labels
            'solution_label': -1 # placeholder label
        }
                
        # Call tokenize_example from prm_dataset.py
        tokenized_example = PRMTrajectoryDataset.tokenize_example(
            example, 
            self.tokenizer, 
            self.step_sep_id, 
            self.pos_step_id, 
            self.neg_step_id, 
            self.max_length,
            config={},
            split='test',
            add_step_tokens=not self.long_cot
        )

        # Extract the required fields
        input_ids = tokenized_example['input_ids']
        attention_mask = tokenized_example['attention_mask']
                
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        
    def score(self, questions: list[str], outputs: list[list[str]], batch_size: int = 8):
        """ to be used with beam search/best of n"""
        
        # Flatten inputs
        flat_questions = []
        flat_completions = []
        for question, answers in zip(questions, outputs):
            for answer in answers:
                flat_questions.append(question)
                flat_completions.append(answer)
                
        # Process completions
        if not self.long_cot:
            flat_completions = [completion.replace("## Step", "Step").strip() for completion in flat_completions]
            flat_completions = [re.split(r'Step \d+:', completion) for completion in flat_completions]
            flat_completions = [[s.strip() for s in completion if s.strip()] for completion in flat_completions]
        else:
            #### long cots will not have step separators. So treat them as a single step
            flat_completions = [[completion.strip()] for completion in flat_completions]
                            
        # Run inference in batches
        flat_results = []
        for i in range(0, len(flat_questions), batch_size):
            batch_questions = flat_questions[i:i+batch_size]
            batch_completions = flat_completions[i:i+batch_size]
            batch_results = self.predict_correctness_batch(batch_questions, batch_completions)
            flat_results.extend(batch_results)
            
        assert len(flat_results) == len(flat_questions), f"Number of results {len(flat_results)} does not match number of questions {len(flat_questions)}"
            
        # Reshape results to match input shape
        scores = []
        idx = 0
        for answers in outputs:
            answer_scores = []
            for _ in answers:
                res_info_tuple = flat_results[idx]
                answer_scores.append([res_info_tuple[0]])
                idx += 1
            scores.append(answer_scores)
            
        return scores
    
class CoTProcessRewardModel:
    def __init__(self,
                model_name_or_path: str,
                max_length: int = 1024,
                n: int = 1,
                temperature: float = 0.0,
                seed: int = 0,
                enable_prefix_caching: bool = False,
                decision_temperature: float = 1.0,
                tensor_parallel_size: Optional[int] = None,
                ) -> None:
        super().__init__()
        
        print(f"Loading PRM model from {model_name_or_path}")
        
        if tensor_parallel_size is None:
            tensor_parallel_size = torch.cuda.device_count()
        
        self.llm = LLM(model_name_or_path,
                    tensor_parallel_size=tensor_parallel_size,
                    seed=seed,
                    gpu_memory_utilization=0.98,
                    max_model_len=max_length,
                    max_logprobs=100,
                    enable_prefix_caching=enable_prefix_caching,
                    )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.seed = seed
        self.max_length = max_length
        self.decision_temperature = decision_temperature
        
        self.sampling_params = SamplingParams(
            max_tokens=self.max_length,
            seed=self.seed,
            temperature=temperature,
            n=n,
            logprobs=100,
            frequency_penalty=0,
        )
               
       ### create a fake config 
        self.config = {
            'task': 'gsm8k',
            'data_dir': '',
            'debug': True,
            'max_length': self.max_length,
            'train_with_gold_solutions': False,
        }

        ### convert to a namespace object we can access by config.
        dataset_config = Config(self.config)

        ### create a dataset objective
        self.dataset_obj = PRMCoTEvalDataset(
            examples=[],
            tokenizer=self.tokenizer,
            process_data=False,
            config=dataset_config,
            split='test'
        )

        self.correct_token = " Yes"
        self.incorrect_token = " No"
        self.correct_token_id = self.tokenizer.encode(self.correct_token, add_special_tokens=False)[-1]
        self.incorrect_token_id = self.tokenizer.encode(self.incorrect_token, add_special_tokens=False)[-1]
    

    def predict_correctness(self, question: str, prefix_steps: list[str]) -> tuple[float, dict]:
        # Tokenize the input
        input_text = self.process_example(question, prefix_steps)
        
        output = self.generation_pipeline([input_text])
        output = output[0][0]['generated_text']
        
        step_cots, step_labels = extract_step_cots_and_labels(output, correct_token=self.correct_token, incorrect_token=self.incorrect_token)
        ## score would be avg of step_labels
        score = sum(step_labels) / len(step_labels)

        info = {
            'step_cots': step_cots,
            'step_labels': step_labels,
        }

        return score, info
    
   
    def predict_correctness_batch(self, questions: list[str], prefix_steps_batch: list[list[str]]) -> list[tuple[float, dict]]:
        # Process all examples in the batch
        inputs = [self.process_example(question, prefix_steps) 
                       for question, prefix_steps in zip(questions, prefix_steps_batch)]
                
        # Generate outputs for the entire batch
        if isinstance(inputs[0], str):
            outputs = self.llm.generate(inputs, self.sampling_params, use_tqdm=False)
        else:
            outputs = self.llm.generate(prompt_token_ids=inputs, sampling_params=self.sampling_params, use_tqdm=False)

        results = []
        for i, input_text in enumerate(inputs):
            # Process all generations for this input
            all_step_scores = []
            all_step_cots = []
            all_step_labels = []
            all_outputs = []
            
            for output in outputs[i].outputs:
                verification_output = output.text
                verification_logprobs = output.logprobs
                
                if os.environ.get('DEBUG'):
                    import ipdb; ipdb.set_trace()
                
                                                                
                step_scores = self.get_step_scores_from_logprobs(verification_logprobs, verification_output, prefix_steps_batch[i])
                #if all(score == -1 for score in step_scores):
                    #print("Warning: No correctness scores found")
                        
                step_cots, step_labels = extract_step_cots_and_labels(verification_output, 
                                                                      correct_token=self.correct_token, 
                                                                      incorrect_token=self.incorrect_token)
                if not step_labels:
                    import ipdb; ipdb.set_trace()
                    print("Warning: No Yes/No decisions found")

                all_step_scores.append(step_scores)
                all_step_cots.append(step_cots)
                all_step_labels.append(step_labels)
                all_outputs.append(verification_output)
                
    
            if not all_step_scores:
                all_step_scores = [-1]
                
            if not all_step_labels:
                all_step_labels = [None]
             
            full_prefix_score = all_step_scores[-1]

            info = {
                'step_cots': all_step_cots,
                'step_labels': all_step_labels, 
                'step_scores': all_step_scores,
                'input_text': input_text,
                'output_texts': all_outputs,
            }
            
            results.append((full_prefix_score, info))
        
        return results
    
    def get_step_scores_from_logprobs(self, verification_logprobs, verification_output, prefix_steps):
        """Extract step scores from logprobs by comparing Yes/No token probabilities."""
        # Map of positive tokens to their negative counterparts
        token_pairs = {
            self.correct_token: self.incorrect_token,
        }
        
        # Create mapping of token IDs for positive/negative pairs
        pos_to_neg_token_ids = {
            self.tokenizer.encode(pos_token, add_special_tokens=False)[-1]: 
            self.tokenizer.encode(neg_token, add_special_tokens=False)[-1]
            for pos_token, neg_token in token_pairs.items()
        }
        
        neg_to_pos_token_ids = {v: k for k, v in pos_to_neg_token_ids.items()}
        
        # Get set of all token IDs we care about
        valid_token_ids = set(pos_to_neg_token_ids.keys()) | set(pos_to_neg_token_ids.values())
    
        scores = []
        label_positions = []
        label_tokens = []
        
        # First pass: find positions of all Yes/No decisions
        for position, logprob_info in enumerate(verification_logprobs):
            top_token = next(token_id for token_id, info in logprob_info.items() if info.rank == 1)
            if top_token in valid_token_ids:
                label_positions.append(position)
                label_tokens.append(top_token)
        
        if not label_positions:
            return [-1] * len(prefix_steps)  # Return default scores if no decisions found
            
        # Second pass: compute confidence scores for each decision
        for position, token in zip(label_positions, label_tokens):
            logprob_info = verification_logprobs[position]
            
            
            if token in pos_to_neg_token_ids:
                pos_token = token
                neg_token = pos_to_neg_token_ids[token]
            elif token in neg_to_pos_token_ids:
                pos_token = neg_to_pos_token_ids[token]
                neg_token = token
            else:
                raise ValueError(f"Token {token} is not in the token mapping")
            
            try:
                pos_logprob = next(lp.logprob for token_id, lp in logprob_info.items() 
                                if token_id == pos_token)
                neg_logprob = next(lp.logprob for token_id, lp in logprob_info.items() 
                                if token_id == neg_token)
                
                # Calculate confidence score using softmax with temperature
                pos_score = np.exp(pos_logprob / self.decision_temperature)
                neg_score = np.exp(neg_logprob / self.decision_temperature)
                confidence = pos_score / (pos_score + neg_score)
                
                scores.append(confidence)
                
            except StopIteration:
                continue
        
        return scores

    def process_example(self, question: str, prefix_steps: list[str]):
        assert len(prefix_steps) > 0, "Prefix steps should not be empty"

        new_prefix_steps = []
        for i, step in enumerate(prefix_steps):
            step = re.sub(r'Step \d+:', '', step).strip()
            step = f'Step {i+1}: {step}'
            new_prefix_steps.append(step)

        solution = '\n'.join(new_prefix_steps)
        # Call tokenize_example from prm_dataset.py
        input_text = self.dataset_obj.format_cot_data(problem=question, solution=solution)

        return input_text
    
    def process_example_with_gold_solution(self, question: str, prefix_steps: list[str], gold_solution: str):
        # Prepare the example for tokenization

        assert len(prefix_steps) > 0, "Prefix steps should not be empty"

        new_prefix_steps = []
        for i, step in enumerate(prefix_steps):
            step = re.sub(r'Step \d+:', '', step).strip()
            step = f'Step {i+1}: {step}'
            new_prefix_steps.append(step)

        solution = '\n'.join(new_prefix_steps)
        # Call tokenize_example from prm_dataset.py
        input_text = self.dataset_obj.format_gold_solution_cot_data(problem=question, gold_solution=gold_solution, solution=solution)

        return input_text
    
    
    def predict_correctness_with_gold_solution_batch(self, questions: list[str], prefix_steps_batch: list[list[str]], gold_solution: list[str]) -> list[tuple[float, dict]]:
        # Process all examples in the batch
        input_texts = [self.process_example_with_gold_solution(question, prefix_steps, gold_solution) 
                       for question, prefix_steps, gold_solution in zip(questions, prefix_steps_batch, gold_solution)]
                
        n = self.sampling_params.n
        # Generate outputs for the entire batch
        outputs = self.llm.generate(input_texts, self.sampling_params, use_tqdm=False)

        results = []
        for i, input_text in enumerate(input_texts):
            # Process all generations for this input
            all_step_scores = []
            all_step_cots = []
            all_step_labels = []
            all_outputs = []
            
            for output in outputs[i].outputs:
                solution_output = output.text
                solution_logprobs = output.logprobs
                                                
                step_scores = self.get_step_scores_from_logprobs(solution_logprobs, solution_output, prefix_steps_batch[i])
                step_cots, step_labels = extract_step_cots_and_labels(solution_output, correct_token=self.correct_token, incorrect_token=self.incorrect_token)
                
                all_step_scores.append(step_scores)
                all_step_cots.append(step_cots)
                all_step_labels.append(step_labels)
                all_outputs.append(solution_output)
            
            # Calculate average scores, ignoring NaN values
            avg_step_scores = np.nanmean(all_step_scores, axis=0)
            # If all scores for a step are NaN, use neutral score
            avg_step_scores = np.nan_to_num(avg_step_scores, nan=0.5)
            full_prefix_score = avg_step_scores[-1]

            avg_step_scores = avg_step_scores.tolist()

            info = {
                'step_cots': all_step_cots,
                'step_labels': all_step_labels, 
                'step_scores': avg_step_scores,
                'input_text': input_text,
                'output_texts': all_outputs,
            }
            
            results.append((full_prefix_score, info))
        
        return results
       

class FewshotCoTProcessRewardModel(CoTProcessRewardModel):
    def __init__(self,
                model_name_or_path: str,
                max_length: int = 1024,
                seed: int = 0,
                n: int = 1,
                temperature: float = 0.7,
                prompt_template: str = None,
                enable_prefix_caching: bool = True,
                ) -> None:
        super().__init__(model_name_or_path, max_length, n=n, temperature=temperature, seed=seed, enable_prefix_caching=enable_prefix_caching)
        self.prompt_template = prompt_template


    def process_example(self, question: str, prefix_steps: list[str]):
        # Prepare the example for tokenization
        if not all([step.startswith('Step') for step in prefix_steps]):
            prefix_steps = [f'Step {i+1}: {step.strip()}' for i, step in enumerate(prefix_steps)]
        
        solution = '\n'.join(prefix_steps)
        # Call tokenize_example from prm_dataset.py
        input_text = self.prompt_template.replace('{problem}', question).replace('{solution}', solution)

        return input_text

    
class MathShepherdPRM:
    def __init__(self,
                device: str = 'cuda',
                ) -> None:
        
        good_token = '+'
        bad_token = '-'
        step_tag = 'ки'

        self.tokenizer = AutoTokenizer.from_pretrained('peiyi9979/math-shepherd-mistral-7b-prm')
        self.candidate_tokens = self.tokenizer.encode(f"{good_token} {bad_token}")[1:]  # [648, 387]


        print("Loading PRM model from peiyi9979/math-shepherd-mistral-7b-prm")
        self.model = AutoModelForCausalLM.from_pretrained('peiyi9979/math-shepherd-mistral-7b-prm', torch_dtype=torch.bfloat16).eval()
        self.device = device
        self.model.to(self.device)
        
        self.step_tag_id = self.tokenizer.encode(f"{step_tag}")[-1] # 12902
        self.step_tag = step_tag

    def predict_correctness(self, question: str, prefix_steps: list[str]) -> tuple[float, dict]:
        output = ""
        for i, step in enumerate(prefix_steps, 1):
            output += f"Step {i}: {step} {self.step_tag}\n"
        
        output = output.strip()
        input_for_prm = f"{question} {output}"
        input_ids = torch.tensor([self.tokenizer.encode(input_for_prm)]).to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids).logits[:, :, self.candidate_tokens]
            scores = logits.softmax(dim=-1)[:, :, 0] # 1 x seq_len
            step_scores = scores[input_ids == self.step_tag_id] # 1 x 

        step_scores = step_scores.cpu().tolist()

        if len(step_scores) != len(prefix_steps):
            print("warning: something probably wrong happened with tokenization that add/removed a step tag")

        prefix_score = step_scores[-1]
        step_labels = [1 if score > 0.5 else 0 for score in step_scores]
        step_cots = [""] * len(step_labels)
        
        info = {
            'full_prefix_score': prefix_score,
            'step_scores': step_scores,
            'step_cots': step_cots,
            'step_labels': step_labels,
            'input_text': input_for_prm,
            'output_texts': [""],
        }

        return prefix_score, info


    def predict_correctness_batch(self, questions: list[str], prefix_steps_batch: list[list[str]]) -> list[tuple[float, dict]]:
        # Process each example into formatted input string
        batch_inputs = []
        for question, prefix_steps in zip(questions, prefix_steps_batch):
            output = ""
            for i, step in enumerate(prefix_steps, 1):
                output += f"Step {i}: {step} {self.step_tag}\n"
            output = output.strip()
            
            ###### MATHShepherd expects the answer format: 'The answer is: <answer>'
            output = output.replace('The answer is', 'The answer is:').replace('the answer is', 'The answer is:').replace('Final Answer: The final answer is', 'The answer is:')
            batch_inputs.append(f"{question} {output}")

        # Tokenize all inputs
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        input_ids = self.tokenizer(batch_inputs, padding=True, return_tensors="pt").input_ids.to(self.device)
                
        # Get model predictions
        with torch.no_grad():
            logits = self.model(input_ids).logits[:, :, self.candidate_tokens]
            scores = logits.softmax(dim=-1)[:, :, 0]  # batch_size x seq_len
        
        # Extract scores for each step tag
        results = []
        for i, prefix_steps in enumerate(prefix_steps_batch):
            step_mask = (input_ids[i] == self.step_tag_id)
            step_scores = scores[i][step_mask].cpu().tolist()
            
            if len(step_scores) != len(prefix_steps):
                print("warning: something probably wrong happened with tokenization that add/removed a step tag")
            
            prefix_score = step_scores[-1] # last step score is the full prefix score
            step_labels = [1 if score > 0.5 else 0 for score in step_scores]
            step_cots = [""] * len(step_labels)
            
            info = {
                'full_prefix_score': prefix_score,
                'step_scores': step_scores,
                'step_cots': step_cots,
                'step_labels': step_labels,
                'input_text': batch_inputs[i],
                'output_texts': [""],
            }
            
            results.append((prefix_score, info))
            
        return results


class RLHFFlowPRM:
    def __init__(self,
                device: str = 'cuda',
                ) -> None:
        self.device = device
        self.model, self.tokenizer = self.load_model_and_tokenizer()

    def load_model_and_tokenizer(
        self, **model_kwargs
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data"
        )
        model = AutoModelForCausalLM.from_pretrained(
            "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            **model_kwargs,
        ).eval()
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

        plus_tag_id = tokenizer.encode("+")[-1]
        minus_tag_id = tokenizer.encode("-")[-1]
        self.candidate_tokens = [plus_tag_id, minus_tag_id]

        return model, tokenizer

    def predict_correctness_batch(
        self, questions: list[str], prefix_steps_batch: list[list[str]], batch_size: int = 2
    ) -> list[tuple[float, dict]]:
        # The RLHFlow models are trained to predict the "+" or "-" tokens in a dialogue, but since these are not unique
        # we need to introduce a dummy special token here for masking.
        
        special_tok_id = self.tokenizer("ки", return_tensors="pt").input_ids[0, 1]
        # We construct two parallel dialogues, one with a "+" token per assistant turn, the other with the dummy token "ки" for masking
        conversations = []
        conversations2 = []
        for question, prefix_steps in zip(questions, prefix_steps_batch, strict=True):
            conversation = []
            conversation2 = []
            for k, step in enumerate(prefix_steps):
                if k == 0:
                    text = question + " " + step
                else:
                    text = step
                conversation.append({"content": text, "role": "user"})
                conversation.append({"content": "+", "role": "assistant"})

                # we track to location of the special token with ки in order to extract the scores
                conversation2.append({"content": text, "role": "user"})
                conversation2.append({"content": "ки", "role": "assistant"})

            conversations.append(conversation)
            conversations2.append(conversation2)

        results = []
        for i in range(0, len(conversations), batch_size):
            convs_batch = conversations[i : i + batch_size]
            convs2_batch = conversations2[i : i + batch_size]
            inputs_batch = self.tokenizer.apply_chat_template(
                convs_batch, padding=True, return_tensors="pt"
            ).to(self.model.device)
            inputs2_batch = self.tokenizer.apply_chat_template(
                convs2_batch, padding=True, return_tensors="pt"
            ).to(self.model.device)
            assert inputs_batch.shape == inputs2_batch.shape
            with torch.no_grad():
                logits = self.model(inputs_batch).logits[:, :, self.candidate_tokens]
                scores = logits.softmax(dim=-1)[
                    :, :, 0
                ]  # 0 means the prob of + (1 mean -)

                for j in range(len(convs_batch)):
                    # We slice on the N-1 token since the model is trained to predict the Nth one ("+" in this case)
                    step_scores = scores[j, :-1][
                        inputs2_batch[j, 1:] == special_tok_id
                    ].tolist()
                    
                    prefix_score = step_scores[-1] # last step score is the full prefix score
                    step_labels = [1 if score > 0.5 else 0 for score in step_scores]
                    step_cots = [""] * len(step_labels)
                    
                    info = {
                        'full_prefix_score': prefix_score,
                        'step_scores': step_scores,
                        'step_cots': step_cots,
                        'step_labels': step_labels,
                        'input_text': self.tokenizer.decode(inputs_batch[j]),
                        'output_texts': [""],
                    }
                    
                    results.append((prefix_score, info))

        return results


