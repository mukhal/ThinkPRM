"""
ThinkPRM: Process Reward Models That Think

This module provides the ThinkPRM implementation, a generative PRM that can
think longer and scale compute through multi-round reasoning and verification.
"""

from typing import List, Tuple, Dict, Optional
import os
import re
import logging
import numpy as np
import torch
from transformers import AutoTokenizer
from utils.answer_utils import extract_step_labels
from utils.helper import format_verification_cot_for_thinkprm
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor

# Conditional imports for optional dependencies
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)


class ThinkPRM:
    """
    ThinkPRM: A generative Process Reward Model that can think longer and scale compute.
    
    This PRM implementation supports multi-round reasoning and verification, allowing
    the model to engage in extended thinking processes before making final decisions.
    
    Attributes:
        llm: The underlying language model for generation
        tokenizer: Tokenizer for the model
        correct_token: Token used to indicate correct reasoning
        incorrect_token: Token used to indicate incorrect reasoning
        n_thinking_rounds: Number of thinking rounds for multi-round verification
        trigger_phrases: Phrases used to trigger additional thinking rounds
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = 4096,
        seed: int = 0,
        n: int = 1,
        temperature: float = 0.7,
        decision_temperature: float = 1.0,
        enable_prefix_caching: bool = True,
        max_gen_tokens: Optional[int] = None,
        n_thinking_rounds: int = 1,
        multiround_verifier: bool = False,
        tensor_parallel_size: Optional[int] = None,
    ) -> None:
        """
        Initialize ThinkPRM.
        
        Args:
            model_name_or_path: Path or name of the model to load
            max_length: Maximum sequence length for the model
            seed: Random seed for reproducibility
            n: Number of samples to generate
            temperature: Sampling temperature for generation
            decision_temperature: Temperature for decision confidence calculation
            enable_prefix_caching: Whether to enable prefix caching
            max_gen_tokens: Maximum tokens to generate (defaults to max_length)
            n_thinking_rounds: Number of thinking rounds for multi-round verification
            multiround_verifier: Whether to use multi-round verification
            tensor_parallel_size: Number of GPUs for tensor parallelism
        """
            
        logger.info(f"Loading ThinkPRM model from {model_name_or_path}")
        
        # Set up tensor parallelism
        if tensor_parallel_size is None:
            tensor_parallel_size = torch.cuda.device_count()
            
        logger.info(f"Max CoT verification length: {max_length}")
        
        # Initialize the language model
        self.llm = LLM(
            model_name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            seed=seed,
            gpu_memory_utilization=0.98,
            max_model_len=max_length,
            max_logprobs=100,
            enable_prefix_caching=enable_prefix_caching,
        )
        
        # Initialize tokenizer and configuration
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.seed = seed
        self.max_length = max_length
        self.decision_temperature = decision_temperature
        
        # Set up sampling parameters
        self.sampling_params = SamplingParams(
            max_tokens=max_length,
            seed=seed,
            temperature=temperature,
            n=n,
            logprobs=100,
            frequency_penalty=0,
        )

        # Set up decision tokens
        self.correct_token = " Yes" #### for the whole prefix
        self.incorrect_token = " No"
        self.step_correct_token = "correct" #### for the step-by-step verification
        self.step_incorrect_token = "incorrect"
        
        
        self.correct_token_id = self.tokenizer.encode(self.correct_token, add_special_tokens=False)[-1]
        self.incorrect_token_id = self.tokenizer.encode(self.incorrect_token, add_special_tokens=False)[-1]
        self.step_correct_token_id = self.tokenizer.encode(self.step_correct_token, add_special_tokens=False)[-1]
        self.step_incorrect_token_id = self.tokenizer.encode(self.step_incorrect_token, add_special_tokens=False)[-1]
        
        # Set up thinking configuration
        self.max_gen_tokens = max_gen_tokens if max_gen_tokens is not None else max_length
        self.n_thinking_rounds = n_thinking_rounds
        self.trigger_phrases = [
            "Let me double check", 
            "Let's verify again", 
            "Did I miss something?"
        ]
        
        # Set up multi-round verification if enabled
        if multiround_verifier:
            self.predict_correctness_batch = self.predict_correctness_batch_multi_round

    def format_cot_data(self, problem: str, solution: str) -> str:
        """
        Format chain-of-thought data for ThinkPRM.
        
        Args:
            problem: The problem statement
            solution: The solution steps
            
        Returns:
            Formatted input text for the model
        """
        return format_verification_cot_for_thinkprm(self.tokenizer, problem, solution)
        
    def process_example(self, question: str, prefix_steps: List[str]) -> str:
        """
        Process a single example by formatting the question and steps.
        
        Args:
            question: The question to be answered
            prefix_steps: List of reasoning steps
            
        Returns:
            Formatted input text for the model
        """
        # Format steps with tags
        formatted_steps = ''
        for sdx, step in enumerate(prefix_steps):
            formatted_steps += f'Step {sdx+1}: {step}\n'
                    
        formatted_steps = formatted_steps.strip()
        # Format prompt with tagged response
        input_text = self.format_cot_data(question, formatted_steps)
        return input_text

    def _extract_score_from_logprobs(
        self, 
        verification_logprobs: List[Dict], 
        verification_output: str, 
        prefix_steps: List[str]
    ) -> List[float]:
        """
        Extract step scores from logprobs by comparing Yes/No token probabilities.
        
        Args:
            verification_logprobs: Log probabilities from the model
            verification_output: Generated verification text
            prefix_steps: List of reasoning steps
            
        Returns:
            List of confidence scores for each step
        """
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

    def _process_batch_outputs(
        self, 
        outputs: List, 
        inputs: List[str], 
        prefix_steps_batch: List[List[str]]
    ) -> List[Tuple[float, Dict]]:
        """
        Process outputs from a batch generation.
        
        Args:
            outputs: Model outputs from generation
            inputs: Input texts used for generation
            prefix_steps_batch: Batch of prefix steps
            
        Returns:
            List of tuples containing scores and info dictionaries
        """
        results = []
        for i, input_text in enumerate(inputs):
            # Process all generations for this input
            all_prefix_scores = []
            all_step_labels = []
            all_outputs = []
            
            for output in outputs[i].outputs:
                verification_output = output.text
                verification_logprobs = output.logprobs
                
                if os.environ.get('DEBUG'):
                    import ipdb; ipdb.set_trace()
                
                prefix_scores = self._extract_score_from_logprobs(
                    verification_logprobs, verification_output, prefix_steps_batch[i]
                )
                        
                step_labels = extract_step_labels(
                    verification_output, 
                    correct_token=self.step_correct_token, 
                    incorrect_token=self.step_incorrect_token
                )
                if not step_labels:
                    logger.warning("No correct/incorrect decisions generated")
                    
                if len(step_labels) < len(prefix_steps_batch[i]):
                    #### model stopped after an incorrect step, pad with 0
                    step_labels = step_labels + [0] * (len(prefix_steps_batch[i]) - len(step_labels))

                all_prefix_scores.append(prefix_scores[-1]) #### the last step score is the full prefix score
                all_step_labels.append(step_labels)
                all_outputs.append(verification_output)
                
            if not all_prefix_scores:
                all_prefix_scores = [-1]
                
            if not all_step_labels:
                all_step_labels = [None]
             
            full_prefix_score = np.mean(all_prefix_scores) ## average across parallel cots

            info = {
                'step_labels': all_step_labels, 
                'inputs': input_text,
                'outputs': all_outputs,
                'prefix_score': full_prefix_score,  
            }
            
            results.append(info)
        
        return results

    def predict_correctness_batch(
        self, 
        questions: List[str], 
        prefix_steps_batch: List[List[str]]
    ) -> List[Tuple[float, Dict]]:
        """
        Predict correctness for a batch of questions and reasoning steps.
        
        Args:
            questions: List of questions to evaluate
            prefix_steps_batch: List of reasoning steps for each question
            
        Returns:
            List of tuples containing correctness scores and detailed info
        """
        # Process all examples in the batch
        inputs = [
            self.process_example(question, prefix_steps) 
            for question, prefix_steps in zip(questions, prefix_steps_batch)
        ]
                
        # Generate outputs for the entire batch
        if isinstance(inputs[0], str):
            outputs = self.llm.generate(inputs, self.sampling_params, use_tqdm=False)
        else:
            outputs = self.llm.generate(
                prompt_token_ids=inputs, 
                sampling_params=self.sampling_params, 
                use_tqdm=False
            )

        return self._process_batch_outputs(outputs, inputs, prefix_steps_batch)
    
    
    def predict_correctness_batch_sequential_scaling(self, questions: list[str], prefix_steps_batch: list[list[str]]) -> list[dict]:
        """
        Predict correctness for a batch of questions and reasoning steps using sequential scaling.
        
        Args:
            questions: List of questions to evaluate
            prefix_steps_batch: List of reasoning steps for each question
            
        Returns:
            List of dictionaries containing correctness scores and detailed info
        """
        # Process all examples in the batch
        inputs = [self.process_example(question, prefix_steps) 
                 for question, prefix_steps in zip(questions, prefix_steps_batch)]
        
        input_prompts = inputs
        
        n = self.sampling_params.n
        self.sampling_params.stop = ['Is the solution correct?']
        self.sampling_params.max_tokens = self.max_gen_tokens - 10
        self.sampling_params.logits_processors = []
        
        logger.info("Generating up to {} tokens for {} rounds".format(self.sampling_params.max_tokens, self.n_thinking_rounds))
        
        # Generate multiple thinking rounds
        for round in range(self.n_thinking_rounds):
            cot_outputs = self.llm.generate(inputs, self.sampling_params, use_tqdm=False)
            
            # Add trigger phrases between rounds except for last round
            if round < self.n_thinking_rounds - 1:
                inputs = [input + output.outputs[0].text.strip() + "\n\n" + self.trigger_phrases[round] 
                         for input, output in zip(inputs, cot_outputs)]
                inputs = [input.replace('</think>', '').replace(self.tokenizer.eos_token, '') 
                         for input in inputs]
        
        # Setup for final decision generation
        predecision_string = 'Is the solution correct?'
        self.correct_token = ' Yes'
        self.incorrect_token = ' No'
        multiple_choice_processor = MultipleChoiceLogitsProcessor(
            tokenizer=self.tokenizer, 
            choices=[self.correct_token, self.incorrect_token]
        )
        
        self.sampling_params.stop = []
        self.sampling_params.max_tokens = 1
        self.sampling_params.logits_processors = [multiple_choice_processor]

        # Prepare inputs for final decision
        predecision_input_ids = []
        for i, batch_outputs in enumerate(cot_outputs):
            input_ids = self.tokenizer(inputs[i], add_special_tokens=False).input_ids
            predecision_string_ids = self.tokenizer(predecision_string, add_special_tokens=False).input_ids
            
            for output in batch_outputs.outputs:
                output_ids = [token_id for token_id in output.token_ids]
                combined_ids = input_ids + output_ids
                
                # Add predecision string if not already present
                if predecision_string_ids[-3:] != output_ids[-3:]:
                    combined_ids += predecision_string_ids
                    
                predecision_input_ids.append(combined_ids)
        
        # Generate final decisions
        final_decisions = self.llm.generate(
            prompt_token_ids=predecision_input_ids, 
            sampling_params=self.sampling_params, 
            use_tqdm=False
        )
        
        # Process results
        results = []
        output_idx = 0
        
        for i, (input_text, cot_output) in enumerate(zip(inputs, cot_outputs)):
            all_prefix_scores = []
            all_step_labels = []
            all_outputs = []
            
            for j in range(n):
                cot_text = cot_output.outputs[j].text
                final_decision = final_decisions[output_idx].outputs[0].text
                final_decision_logprobs = final_decisions[output_idx].outputs[0].logprobs
                
                # Extract scores from logprobs
                step_scores = self._extract_score_from_logprobs(
                    final_decision_logprobs, final_decision, prefix_steps_batch[i]
                )
                
                if all(score == -1 for score in step_scores):
                    logger.warning("Warning: No correctness scores found")
                
                # Convert final decision to step labels
                assert final_decision.strip() in [self.correct_token.strip(), self.incorrect_token.strip()], \
                    f"Final decision is not in {self.correct_token} or {self.incorrect_token}: {final_decision.strip()}"
                
                step_labels = [1 if final_decision.strip() == self.correct_token.strip() else 0]
                
                all_prefix_scores.append(step_scores[-1] if step_scores else -1)
                all_step_labels.append(step_labels)
                output = input_text[input_text.find('<think>'):] + '\n' + cot_text + '\n' + final_decision
                all_outputs.append(output)
                
                output_idx += 1
            
            if not all_prefix_scores:
                all_prefix_scores = [-1]
                
            if not all_step_labels:
                all_step_labels = [None]
             
            full_prefix_score = np.mean(all_prefix_scores)

            info = {
                'step_labels': all_step_labels, 
                'inputs': input_prompts[i],
                'outputs': all_outputs,
                'prefix_score': full_prefix_score,
            }
            
            results.append(info)
                    
        return results
    
    
