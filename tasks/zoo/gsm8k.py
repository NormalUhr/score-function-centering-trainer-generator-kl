"""GSM8K as a verifiers environment: grade-school word problems with a 0/1
reward for RLVR. Same conventions as countdown.py (system prompt, <answer>
tags, load_environment()), so the trainer runs it with env.id=gsm8k and the
Countdown prompt format unchanged.
"""
from __future__ import annotations
import re

import verifiers as vf

SYSTEM_PROMPT = ("You are a helpful assistant. You first thinks about the "
                 "reasoning process in the mind and then provides the user "
                 "with the answer.")


def instruction(question):
    return (f" {question} Show your work in <think> </think> tags. And return "
            f"the final answer as a single number in <answer> </answer> tags, "
            f"for example <answer> 42 </answer>.")


def _messages(question):
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction(question)}]


def extract_answer(text):
    m = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return m[-1].strip() if m else None


def parse_number(text):
    """The last number in `text` with commas, currency and units dropped;
    None when there is none."""
    cleaned = text.replace(",", "").replace("$", "")
    numbers = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    return float(numbers[-1]) if numbers else None


def gold_answer(solution):
    """GSM8K stores the reference as a worked solution ending in '#### N'."""
    return parse_number(solution.split("####")[-1])


def compute_reward(completion, target):
    answer = extract_answer(completion)
    if answer is None:
        return 0.0
    value = parse_number(answer)
    return 1.0 if value is not None and abs(value - target) < 1e-6 else 0.0


def correct(completion, info):
    text = completion[-1].content if completion else None
    return compute_reward(text if isinstance(text, str) else "", info["target"])


def _rows(split):
    from datasets import load_dataset
    rows = []
    for example in load_dataset("openai/gsm8k", "main", split=split):
        target = gold_answer(example["answer"])
        rows.append({"prompt": _messages(example["question"]), "answer": str(target),
                     "info": {"target": target}})
    return rows


def load_environment(num_eval_examples=128):
    """Standard verifiers entry point: the 7,473 training problems, the first
    `num_eval_examples` test problems as the environment's evaluation set (the
    trainer holds out the head of the shuffled training set instead), and a
    parser whose extraction rate the trainer logs as frac_parsed."""
    from datasets import Dataset
    return vf.SingleTurnEnv(
        dataset=Dataset.from_list(_rows("train")),
        eval_dataset=Dataset.from_list(_rows("test")[:num_eval_examples]),
        parser=vf.Parser(extract_fn=extract_answer),
        rubric=vf.Rubric(funcs=[correct]))


if __name__ == "__main__":
    assert gold_answer("Natalia sold 48/2 = <<48/2=24>>24 clips.\n#### 72") == 72.0
    assert gold_answer("... total.\n#### 1,250") == 1250.0
    for name, completion, target, expected in (
            ("correct", "<think>72 total</think><answer> 72 </answer>", 72.0, 1.0),
            ("correct with units", "<answer>$72.00</answer>", 72.0, 1.0),
            ("wrong value", "<answer>71</answer>", 72.0, 0.0),
            ("no tag", "the answer is 72", 72.0, 0.0),
            ("empty tag", "<answer></answer>", 72.0, 0.0),
            ("last tag counts", "<answer>5</answer> no, <answer>72</answer>", 72.0, 1.0),
            ("negative", "<answer>-3</answer>", -3.0, 1.0)):
        got = compute_reward(completion, target)
        assert got == expected, (name, got, expected)
        print(f"{name:20s} -> {got}")
    env = load_environment(num_eval_examples=4)
    row = env.get_dataset()[0]
    good = [vf.AssistantMessage(content=f"<answer>{row['answer']}</answer>")]
    assert correct(good, row["info"]) == 1.0 and correct([], row["info"]) == 0.0
    print(f"ok: env of {len(env.get_dataset())}+{len(env.get_eval_dataset())} rows, "
          "rubric scores the recorded answers")
