from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from .answer_utils import get_action_trace_from_plan_str
import sys, os, re 
import sympy
import random

sys.path.append("..")
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)
import wandb
from decision_transformer.game24 import Game24Engine
import torch

class BWEvalCallback(TrainerCallback):
    def __init__(self, eval_dataset, bw_evaluator, dt_model, batch_size=64,
                 model_save_dir=None,
                 train_dataset=None
                 ):
        self.eval_dataset = eval_dataset
        self.bw_evaluator = bw_evaluator
        self.dt_model = dt_model
        self.batch_size = batch_size
        self.best_accuracy = 0.0
        self.model_save_dir = model_save_dir
        self.train_dataset = train_dataset

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        dataset = self.eval_dataset
        bw_evaluator = self.bw_evaluator
        batch_size = self.batch_size
        dt_model = self.dt_model
        
        n_correct = 0
        # Iterate over the dataset in batches
        logger.error("***** Running evaluation *****")
        condition_bin = max(self.train_dataset.rets)
        logger.error("  Conditioning on {}".format(condition_bin))

        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[i:i + batch_size]
            generated_sols = dt_model.get_solution_greedy(batch, condition_bin=condition_bin)
            for ex, sol in zip(batch, generated_sols):
                action_trace = get_action_trace_from_plan_str(sol)
                print("action_trace: ", action_trace)
                is_correct = bw_evaluator.eval_output(answer=ex, output=action_trace)
                n_correct += is_correct


        accuracy = n_correct / len(dataset)
        print('**************\nAccuracy: {:.2f}\n**************'.format(accuracy))
        ### log to wandb
        wandb.log({'eval_accuracy': accuracy})
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            print("saving best model to: ", self.model_save_dir)
            self.dt_model.base_model.save_pretrained(os.path.join(self.model_save_dir, 'best_model'))
            ### save accuracy to json file
            import json
            with open(os.path.join(self.model_save_dir, 'best_model', 'metrics.json'), 'w') as f:
                json.dump({"accuracy": accuracy}, f)

class Game24EvalCallBack(TrainerCallback):
    def __init__(self, eval_dataset, dt_model, batch_size=64, model_save_dir=None,
                 train_dataset=None, config=None):
        self.eval_dataset = eval_dataset
        self.dt_model = dt_model
        self.batch_size = batch_size
        self.best_accuracy = 0.0
        self.model_save_dir = model_save_dir
        self.train_dataset = train_dataset
        self.config = config

    def test_output(self, question: str, output: str):

        if '(left: 24)' in output:
            print("output: ", output)
        ### strip away '\n##' from output 
        output = output.strip().split('##')[0].strip()
        expression = output.strip().split('\n')[-1].lower().replace('answer: ', '').split('=')[0]
        numbers = re.findall(r'\d+', expression)
        problem_numbers = re.findall(r'\d+', question)
        if sorted(numbers) != sorted(problem_numbers):
            return False
        try:
            return int(sympy.simplify(expression) == 24)
        except Exception as e:
            return False

    
    def on_save(self, args: TrainingArguments=None, state: TrainerState=None, 
                control: TrainerControl=None, 
                test=False, 
                **kwargs):
        dataset = self.eval_dataset
        batch_size = self.batch_size
        dt_model = self.dt_model
        dt_model.base_model.eval()
        
        n_correct = 0
        # Iterate over the dataset in batches
        logger.error("***** Running evaluation *****")
        if self.train_dataset is None:
            max_ret = None 
        else:
            max_ret = max(self.train_dataset.rets)

        bins = [max_ret]
        
        with torch.no_grad():
            for i in tqdm(range(0, len(dataset), batch_size), desc='evaluating..'):
                batch = dataset[i:i + batch_size]
                
                if self.config.data.quantize_reward:
                    n_samples = dt_model.config['self_consistency_samples']
                else:
                    n_samples = len(bins) * dt_model.config['self_consistency_samples']

                generated_sols = dt_model.sample_solutions(batch, condition_bins=bins)
                assert len(generated_sols) == len(batch) * n_samples, 'Number of generated solutions must be equal to the number of questions * number of bins * number of samples'
                
                for i in range(0, len(generated_sols), n_samples):
                    traces = generated_sols[i:i + n_samples]
                    example_idx = i // n_samples
                    example = batch[example_idx]
                    could_solve = [Game24EvalCallBack.eval_dt_output_with_left_numbers(problem=example, output=trace) for trace in traces]
                    #if any(could_solve):
                    #    print("Found a correct solution to the problem: ", example)
                    n_correct += any(could_solve)

        accuracy = n_correct / len(dataset)
        print('**************\nSolve Rate: {:.2f}\n**************'.format(accuracy))
        subset = 'test' if test else 'dev'

        ### log to wandb
        wandb.log({'{}_solve_rate'.format(subset): accuracy})
        if accuracy > self.best_accuracy and not test:
            self.best_accuracy = accuracy
            print("saving best model to: ", self.model_save_dir)
            self.dt_model.base_model.save_pretrained(os.path.join(self.model_save_dir, 'best_model'))
            ### save accuracy to json file
            import json
            with open(os.path.join(self.model_save_dir, 'best_model', 'metrics.json'), 'w') as f:
                json.dump({"accuracy": accuracy}, f)

        dt_model.base_model.train()

    def _on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        dataset = self.eval_dataset
        batch_size = self.batch_size
        dt_model = self.dt_model
        
        n_correct = 0
        # Iterate over the dataset in batches
        logger.error("***** Running evaluation *****")
        max_ret = max(self.train_dataset.rets)
        
        if self.config.data.quantize_reward:
            logger.error("Conditioning on <RET{}>".format(max_ret))
        else:
            assert isinstance(max_ret, float), 'RETURN-TO-GO must be a float'
            #condition_bin = round(condition_bin, self.config.data.reward_precision)
            #logger.error("Conditioning on R: {}".format(condition_bin))
            ## bins are top 5 values
            ## round returns first. 
            rets = [round(r, self.config.data.reward_precision) for r in self.train_dataset.rets]
            ## keepy unique values
            bins = list(set(rets))
            bins = sorted(bins, reverse=True)[:3]
            for b in bins:
                print("conditioning on R: ", b)

        if self.config.eval.get('intervene', False):
            logger.error("Intervening on intemediate steps by conditioning on the highest return.")
        
        n_samples = dt_model.config['self_consistency_samples']
        for i in tqdm(range(0, len(dataset), batch_size), desc='evaluating with engine..'):
            batch = dataset[i:i + batch_size]
            for example in batch:
                for _ in range(n_samples):
                    ### sample a bin with a probability proportional to the return -- the higher the return, the higher the probability
                    if self.config.data.quantize_reward:
                        bin = max_ret
                    else:
                        bin = random.choices(bins, weights=[1 / (1 + abs(r - max_ret)) for r in bins])[0]
                    
                    sol = dt_model.sample_one_solution_with_engine(example, condition_bin=bin, 
                    intervene=self.config.eval.get('intervene', False))

                    if Game24EvalCallBack.eval_dt_output(sol):
                        n_correct += 1
                        sol = sol.split('Answer:')[0]
                        #print(">>> CORRECT sol: ", sol)
                        break
                

        accuracy = n_correct / len(dataset)
        print('**************\nSolve Rate: {:.2f}\n**************'.format(accuracy))
        ### log to wandb
        wandb.log({'eval_solve_rate': accuracy})
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            print("saving best model to: ", self.model_save_dir)
            self.dt_model.base_model.save_pretrained(os.path.join(self.model_save_dir, 'best_model'))
            ### save accuracy to json file
            import json
            with open(os.path.join(self.model_save_dir, 'best_model', 'metrics.json'), 'w') as f:
                json.dump({"accuracy": accuracy}, f)

    @staticmethod    
    def eval_dt_output(output):
        #### will make sure that each formula is correct then make sure that (left: 24) is in the output
        output = output.split('Answer:')[0]
        
        if '(left: 24)' not in output:
            return False
        
        formulas = [line for line in output.split('\n') if '=' in line]
        for f in formulas:
            f = f.split('(left')[0]
            lhs = f.split('=')[0].strip()
            rhs = f.split('=')[1].strip()
            try:
                if sympy.simplify(lhs) != sympy.simplify(rhs):
                    #print("incorrect formula: ", f)
                    return False
            except Exception as e:
                print("error with expression: ", f)
                return False
            
        return True
    
    @staticmethod    
    def eval_dt_output_with_left_numbers(problem, output):
        #### will make sure that each formula is correct then make sure that (left: 24) is in the output
        output = output.split('done')[0].split('Answer:')[0]

        formulas = [line for line in output.split('\n') if '=' in line]
        cur_left_numbers = problem

        for f in formulas:
            ## make sure left numbers are good
            try:
                left_numbers = Game24Engine.get_left_numbers(f, cur_left_numbers)
            except Exception as e:
                #("error with expression: {} given left numbers: {}".format(f, cur_left_numbers))
                return False

            cur_left_numbers = left_numbers

            ## make sure math operation is correct
            try:
                f = f.split('(left')[0]
                lhs = f.split('=')[0].strip()
                rhs = f.split('=')[1].strip()
                if sympy.simplify(lhs) != sympy.simplify(rhs):
                    return False
            except Exception as e:
                #print("error with expression: ", f)
                return False
        
        try: 
            return sympy.simplify(cur_left_numbers) == sympy.simplify('24')
        except Exception as e:
            return False 


        