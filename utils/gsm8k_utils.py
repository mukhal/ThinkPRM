import re
from typing import Optional
from collections import Counter


def retrieve_answer(output: str) -> Optional[str]:
    match = re.match(r'.*The answer is .*?([ $.0-9,\-]+).*\..*', output)
    if match is None:
        return None
    answer = match[1].replace(',', '').replace('$', '').replace(' ', '')
    if '=' in answer:
        answer = answer[answer.rindex('=') + 1:]
    return answer


def retrieve_answer_from_dataset(answer: str) -> str:
    return re.match(r'[\S\s]*#### (.*)$', answer)[1]


def judge_answer(output: Optional[str], answer: str) -> bool:
    if output is None:
        return False
    try:
        output = int(output)
        answer = int(answer)
        return output == answer
    except Exception as e:
        pass
    try:
        output = float(output)
        answer = float(answer)
        return output == answer
    except Exception as e:
        pass
    return output == answer


def get_majoirty_answer(answers):
    if len(answers) == 0: # all solutions are invalid -- do not have a final answer
        return '[INVALID]'
    else:
        ## pick the majority answer
        voted_answer = Counter(answers).most_common(1)[0][0]
        ## pick the first solution that has the voted answer
        return voted_answer


def gsm8k_is_correct(model_output, gold_traj):
    return judge_answer(output=retrieve_answer(model_output), answer=retrieve_answer_from_dataset(gold_traj))