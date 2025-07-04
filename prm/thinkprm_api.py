from sglang import function, gen, set_default_backend, RuntimeEndpoint
import numpy as np
import re
from transformers import AutoTokenizer
from utils.helper import format_verification_cot_for_thinkprm
from typing import List, Tuple, Dict
import os, json, pathlib


def preprocess_longcot(long_cot: str) -> str:
    """Process the chain-of-thought text before verification.
    
    Args:
        long_cot: The chain-of-thought text to process
        
    Returns:
        The processed text with think tokens removed and content extracted
    """
    if not long_cot:
        print("WARNING: Empty solution provided")
        return ""
        
    return long_cot.split('</think>')[0].strip()

class APIThinkPRMVerifier:
    def __init__(self,
                max_length: int = 8192,
                seed: int = 0,
                n: int = 1,
                temperature: float = 0.7,
                endpoint: str = "http://localhost:30000",
                model_name_or_path: str = "launch/ThinkPRM-1.5B",
                decision_temperature: float = 1.0,
                predecision_string: str = "\nIs the solution correct? ",
                label_categories: str = "yes,no",
                score_label_idx: int = 0,
                long_cot: bool = False,
                process_verifier: bool = False,
                n_thinking_rounds: int = 1,
                trigger_phrase: str = "\nWait, let me double check...\n",
                verifier_instruction: str = None,
                ) -> None:
        
        set_default_backend(RuntimeEndpoint(endpoint))
    
        self.max_length = max_length
        self.n = n
        self.temperature = temperature
        self.decision_temperature = decision_temperature
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.all_verifications = []
        
        self.predecision_string = predecision_string
        self.label_categories = label_categories.split(",")
        self.score_label_idx = score_label_idx
        if process_verifier:
            self.step_string = "Step {i}: "
        else:
            self.step_string = ""
            
        assert self.score_label_idx < len(self.label_categories), "score_label_idx must be less than the number of label categories"
        
        self.correct_token = self.label_categories[self.score_label_idx]
        self.long_cot = long_cot
        self.process_verifier = process_verifier
        self.n_thinking_rounds = n_thinking_rounds
        self.trigger_phrase = trigger_phrase
        self.verifier_instruction = verifier_instruction
        print("CALIBRATION TEMPERATURE: ", self.decision_temperature)
        print("PREDECISION STRING: ", self.predecision_string)
        print("LABEL CATEGORIES: ", self.label_categories)
        print("SCORE LABEL: ", self.label_categories[self.score_label_idx])
        print("MAX VERIFICATION LENGTH: ", self.max_length)
        
        self.trigger_phrases = ["Let me double check", "Let's verify again",  "Did I miss something?"] # used for budget forcing experiments
        
        if self.n_thinking_rounds > 1:
            assert n == 1, "Sequential scaling with {} thinking rounds requires n=1".format(self.n_thinking_rounds)
            print("Sequential scaling with {} thinking rounds".format(self.n_thinking_rounds))
            print("TRIGGER PHRASES: ", self.trigger_phrases)
        elif n > 1:
            print("Parallel scaling with n={}".format(n))
        
        print("VERIFIER INSTRUCTION: ", self.verifier_instruction)
        
        @function
        def cot_eval(s, prompt: str, n):
            # Extract max step number from prompt
            stop_patterns = ["Is the solution correct?", self.tokenizer.eos_token, "</think>"]
            if self.process_verifier:
                ## in case of search, generation should pause in case the verifier tries to verify steps that do not exist. 
                step_matches = re.findall(r'Step (\d+):', prompt)
                max_step = int(max(step_matches)) if step_matches else 1
                stop_patterns.append(f"Step {max_step + 1}:")

            s += prompt

            forks = s.fork(n)
            for fork in forks:
                if self.n_thinking_rounds == 1:
                    fork += gen("verification", max_tokens=self.max_length - 20, temperature=self.temperature, stop=stop_patterns)
                else:
                    for r in range(self.n_thinking_rounds):
                        fork += gen(f"verification_round_{r}", max_tokens=self.max_length - 20, temperature=self.temperature, stop=stop_patterns)
                        if r < self.n_thinking_rounds - 1:
                            fork += self.trigger_phrases[r]
                    
                fork += self.predecision_string
                fork += gen("decision", choices=self.label_categories)
                if prompt not in self.prompt_to_states:
                    self.prompt_to_states[prompt] = []
                self.prompt_to_states[prompt].append(fork)
        
        self.cot_eval = cot_eval

    def process_example(self, question: str, prefix_steps: list[str]):
        if len(prefix_steps) == 0:
            print("WARNING: Empty solution provided")
            prefix_steps = [""]

        new_prefix_steps = []
        for i, step in enumerate(prefix_steps):
            if not self.long_cot:
                step = re.sub(r'Step \d+:', '', step).strip()
                step = self.step_string.format(i=i+1) + step
            new_prefix_steps.append(step)
            
        solution = '\n'.join(new_prefix_steps)
        return format_verification_cot_for_thinkprm(self.tokenizer, problem=question, solution=solution, long_cot=self.long_cot, instruction=self.verifier_instruction)

    def predict_correctness_batch(self, questions: List[str], prefix_steps_batch: List[List[str]]) -> List[Tuple[float, Dict]]:
        prompts = [self.process_example(q, steps) for q, steps in zip(questions, prefix_steps_batch)]
        
        self.prompt_to_states = {}
        _ = self.cot_eval.run_batch([{'prompt': prompt, 'n': self.n} for prompt in prompts])
        
        results = []
        i = 0
        while i < len(prompts):
            _scores = []
            _cots = []
            _decisions = []
            
            for j in range(self.n):
                try:
                    state = self.prompt_to_states[prompts[i]][j]
                    if self.n_thinking_rounds == 1:
                        cot = state["verification"]
                    else:
                        cot = self.trigger_phrase.join([state[f"verification_round_{r}"] for r in range(self.n_thinking_rounds)])
                    
                    cot += self.predecision_string + state["decision"]
                    
                    decision = 1 if self.correct_token in state["decision"] else 0
                    logprobs = np.array(state.get_meta_info("decision")["normalized_prompt_logprobs"])
                except Exception as e:
                    ## usually OOM error, we pause until we restart the sglang server
                    scores = np.array([0])
                    cot = ""
                    decision = 0
                    logprobs = np.array([0, 1]) # consider it as incorrect
                    
                scores = np.exp(logprobs / self.decision_temperature)
                scores /= np.sum(scores)
                score = scores[self.score_label_idx]
                
                _scores.append(score)
                _cots.append(cot)
                _decisions.append(decision)
                
                
            if os.environ.get("DEBUG"):
                import ipdb; ipdb.set_trace()
            
            score = sum(_scores) / len(_scores)
            info = {
                'step_cots': [],
                'step_labels': [], 
                'step_scores': [score],
                'input_text': prompts[i],
                'output_texts': _cots,
            }
            
            results.append((score, info))
            i += 1
           
        return results
    
    def score(self, questions: list[str], outputs: list[list[str]], batch_size: int = 8):
        flat_questions = []
        flat_completions = []
        for question, answers in zip(questions, outputs):
            for answer in answers:
                flat_questions.append(question)
                flat_completions.append(answer)
                
        if self.long_cot: ## in case the solution is a long chain of thought, we need to process it before verification (not used in the paper)
            flat_completions = [[preprocess_longcot(completion)] for completion in flat_completions]
        else:
            flat_completions = [completion.replace("## Step", "Step").strip() for completion in flat_completions]
            flat_completions = [re.split(r'Step \d+:', completion) for completion in flat_completions]
            flat_completions = [[s.strip() for s in completion if s.strip()] for completion in flat_completions]
                        
        flat_results = []
        for i in range(0, len(flat_questions), batch_size):
            batch_questions = flat_questions[i:i+batch_size]
            batch_completions = flat_completions[i:i+batch_size]
            batch_results = self.predict_correctness_batch(batch_questions, batch_completions)
            flat_results.extend(batch_results)
            
        assert len(flat_results) == len(flat_questions), f"Number of results {len(flat_results)} does not match number of questions {len(flat_questions)}"
            
        scores = []
        idx = 0
        for answers in outputs:
            answer_scores = []
            for _ in answers:
                res_info_tuple = flat_results[idx]
                answer_scores.append([res_info_tuple[0]])
                idx += 1
            scores.append(answer_scores)
            
        # Store verifications for later saving
        self.all_verifications.extend(flat_results)
            
        return scores