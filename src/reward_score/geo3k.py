# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import re
import json

from mathruler.grader import extract_boxed_content, grade_answer

def _log_result(log_path: str, record: dict, concurrent_semaphore=None):
    """把单条记录以 JSON 行写入日志文件；如提供并发信号量则上锁写入。"""
    if not log_path:
        return
    line = json.dumps(record, ensure_ascii=False)
    if concurrent_semaphore is not None:
        try:
            concurrent_semaphore.acquire()
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        finally:
            concurrent_semaphore.release()
    else:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

def format_reward(predict_str: str) -> float:
    pattern = re.compile(r".*\\boxed\{.*\}.*", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0


def acc_reward(predict_str: str, ground_truth: str, use_boxed: bool = True) -> float:
    if use_boxed:
        answer = extract_boxed_content(predict_str)
    else:
        answer = predict_str
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(data_source, solution_str: str, ground_truth: str, extra_info, use_boxed: bool = True, format_score: float = 0.1) -> float:
    acc = (1.0 - format_score) * acc_reward(solution_str, ground_truth, use_boxed)
    format = format_score * format_reward(
        solution_str
    )
    score = acc + format
    # record = {
    #     "solution": solution_str,
    #     "ground_truth": ground_truth,
    #     "acc": acc,
    #     "format": format,
    #     "score": score,
    # }
    # log_path = "/vepfs-cnbj3fa964354bf4/jiamx/ymy/Open-TeleRL-0.5.x_freeze_only/logs/answers/geo3k_imcot.jsonl"
    # _log_result(log_path, record)

    return score
