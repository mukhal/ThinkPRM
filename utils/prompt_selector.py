
class PromptSelector:
    @staticmethod
    def select_prompt(task, model_name):
        match task:
            case 'game24':
                if 'wizardlm' in model_name.lower():
                    ### open prompts/game24/wizardlm.txt 
                    with open('prompts/game24/wizardlm.txt') as f:
                        prompt = f.read()

                if 'llama' in model_name.lower():
                    ### open prompts/game24/llama.txt 
                    with open('prompts/game24/llama.txt') as f:
                        prompt = f.read()

            case _:
                raise ValueError(f'Invalid task: {task}')
        
        return prompt
    