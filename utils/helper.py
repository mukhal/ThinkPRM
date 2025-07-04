
## main template used in the paper
def format_verification_cot_for_thinkprm(tokenizer, problem, solution, cot=None, long_cot=False, instruction=None):
    #TODO: pass template as argument
    _instruction = instruction if instruction is not None else "Review and critique each step in the proposed solution to determine whether each step is correct. If the solution is incomplete, only verify the provided steps." ## default instruction (models were trained with this)
    
    instruction_template = """You are given a math problem and a proposed step-by-step solution:

[Math Problem]

{problem}

[Solution]

{solution}

{_instruction}
""".strip()       
    if cot: # training
        if "Let's verify step by step:" not in cot:
            cot = "Let's verify step by step:\n" + cot
        
        s = tokenizer.apply_chat_template([
            {'role': "user", "content": instruction_template.replace('{problem}', problem).replace('{solution}', solution).replace('{_instruction}', _instruction)},
            {'role': "assistant", "content": cot}
        ], tokenize=False, add_generation_prompt=False)
        
        if "Let's verify step by step:" not in s:
            s = "Let's verify step by step:\n" + s
    else: # inference
        s = tokenizer.apply_chat_template([
            {'role': "user", "content": instruction_template.replace('{problem}', problem).replace('{solution}', solution).replace('{_instruction}', _instruction)}
        ], tokenize=False, add_generation_prompt=True) + "\nLet's verify step by step:"
                
    return s


## processing needed for training
def format_train_verification_cot_for_thinkprm(tokenizer, problem, solution, cot=None):
    instruction = """You are given a math problem and a proposed step-by-step solution:

[Math Problem]

{problem}

[Solution]

{solution}

Review and critique each step in the proposed solution to determine whether each step is correct. If the solution is incomplete, only verify the provided steps.
""".strip()
    if cot: # training
        # Save what was after </think>
        after_think = cot[cot.index('</think>')+len('</think>'):]
        # Remove </think>
        cot = cot[:cot.index('</think>')]
        
        if "Let's verify step by step:" not in cot:
            cot = cot.replace("<think>", "<think>\nLet's verify step by step:")
        
        s = tokenizer.apply_chat_template([
            {'role': "user", "content": instruction.replace('{problem}', problem).replace('{solution}', solution)},
            {'role': "assistant", "content": f"{cot}"}
        ], tokenize=False, add_generation_prompt=False).replace(tokenizer.eos_token, '')
     
        s += '</think>' + after_think
        s+= tokenizer.eos_token
        
    else: # inference
        s = tokenizer.apply_chat_template([
            {'role': "user", "content": instruction.replace('{problem}', problem).replace('{solution}', solution)},
        ], tokenize=False, add_generation_prompt=True) + "\nLet's verify step by step:"
        
    return s


## template for non-thinking verifiers (not used in the paper)
def format_verification_cot_no_think(tokenizer, problem, solution, cot=None):
    instruction = ("Given a math question and partial solution steps, analyze each step in the solution, then determine whether it is correct. Provide the analysis for each step first, then indicate with 'Yes' or 'No' whether it is correct.")
    try:
        if cot:
            return tokenizer.apply_chat_template([
                {'role': "user", "content": f"{instruction}\n\nQuestion: {problem}\n\n{solution}"},
                {'role': "assistant", "content": f"Analysis:\n{cot}"}
            ], tokenize=False)
        else:
            s = tokenizer.apply_chat_template([
                {'role': "user", "content": f"{instruction}\n\nQuestion: {problem}\n\n{solution}"},
                {'role': "assistant", "content": ""}
            ], tokenize=False, add_generation_prompt=False)
            return s.replace(tokenizer.eos_token, '')
    except Exception:
        if cot:
            return f"{instruction}\n\nQuestion: {problem}\n\n{solution}\n\nAnalysis:\n{cot}{tokenizer.eos_token}"
        else:
            return f"{instruction}\n\nQuestion: {problem}\n\n{solution}\n\n"


