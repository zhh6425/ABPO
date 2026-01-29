from math_verify import parse, verify
import re

def normalize_answer(answer: str) -> str:
    s = answer
    s = s.replace("dfrac", "frac")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\s+", "", s).strip()
    s = s.replace("\\varnothing", "\\emptyset")
    for unit in ["cm", "cm²", "minutes", "meters", "a\\.m\\.", "p\\.m\\."]:
        s = re.sub(rf"(\d+)\s*{unit}", r"\1", s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*\\%", r"\1", s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"\1", s)
    s = re.sub(r'(\\degree|degrees|°)', r'^\\circ', s)
    return s

def do_verify(nsol, gt): 
    res = 0.0 
    try:
        if not "\\boxed" in nsol:
            nsol = f"\\boxed{{{nsol}}}"
        pre = parse(nsol)
        if not "\\boxed" in gt: 
            gt = f"\\boxed{{{gt}}}"
        gt = parse(gt)
        
        if len(gt)>1 and (gt[1] in 'ABCDEFGHIJK'):
            res = float(nsol[len("\\boxed{"):].startswith(gt[1]))
        else:
            if len(pre)==0: res = 0.0 
            else: res = float(verify(pre, gt))
    except: 
        res = 0.0
    return res 

def format_reward(predict_str: str) -> float:
    pattern = re.compile(r"^<think>.*?</think>\s*<answer>.*?</answer>$", re.DOTALL)
    # pattern = re.compile(r"^<think>.*?</think>.*?<answer>.*?</answer>$", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0

def compute_score(data_source, solution_str, ground_truth, extra_info=None, generate=False):
    
    ground_truth = normalize_answer(ground_truth)
    retval = 0.0
    try:
        answer_pattern = r'<answer>\s*([^<]+?)\s*</answer>'
        match = re.search(answer_pattern, solution_str)
        answer = normalize_answer(match.group(1)) if match else None
        
        if answer is not None:
            retval = float(answer == ground_truth) # try to compare directly
            if retval == 0.0: # if not equal, try to verify
                retval = do_verify(answer, ground_truth)
    except:
        # print(f"Error:\ngot answer: {solution_str}\nand gt {gt}")
        retval = 0.0

    if generate:
        return float(retval) * format_reward(solution_str)
    else: # training with format reward
        return {
            "score": float(retval) + 0.1 * format_reward(solution_str),
            "answer": answer,
        }
    
