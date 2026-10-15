"""Countdown as a verifiers environment: on-the-fly solvable problems and a
0/1 reward for RLVR. One file in the hub convention (a load_environment()
function), so tasks/rollout.py loads it by id like an installed hub environment.
"""
from __future__ import annotations
import random
import re
from collections import Counter
from fractions import Fraction

import verifiers as vf

OPERATIONS = ("+", "-", "*", "/")


def _build_random_expr(rng, numbers, operations):
    """Combine `numbers` pairwise in random order/ops -> (Fraction value, expr)."""
    items = [(Fraction(n), str(n)) for n in numbers]
    while len(items) > 1:
        i, j = rng.sample(range(len(items)), 2)
        av, ae = items[i]
        bv, be = items[j]
        op = rng.choice(operations)
        if op == "/" and bv == 0:
            op = "+"
        val = {"+": av + bv, "-": av - bv, "*": av * bv,
               "/": (av / bv if bv != 0 else av)}[op]
        items = [items[k] for k in range(len(items)) if k not in (i, j)]
        items.append((val, f"({ae} {op} {be})"))
    return items[0]


def generate(rng, num_operands=4, min_number=1, max_number=100,
             max_target=1000, operations=OPERATIONS, max_tries=200):
    """Generate ONE solvable instance: dict(numbers, target, solution).
    Re-rolls until the built expression has a positive-integer value in range."""
    for _ in range(max_tries):
        numbers = [rng.randint(min_number, max_number) for _ in range(num_operands)]
        val, exp = _build_random_expr(rng, numbers, list(operations))
        if val.denominator == 1 and 1 <= val <= max_target:
            return {"numbers": numbers, "target": int(val),
                    "solution": exp[1:-1] if exp[0] == "(" else exp}
    raise RuntimeError("no in-range integer target found in max_tries; widen "
                       "max_target or narrow the number range / operations")


SYSTEM_PROMPT = ("You are a helpful assistant. You first thinks about the "
                 "reasoning process in the mind and then provides the user "
                 "with the answer.")


def instruction(numbers, target):
    return (f" Using the numbers {numbers}, create an equation that equals "
            f"{target}. You can use basic arithmetic operations (+, -, *, /) "
            f"and each number can only be used once. Show your work in "
            f"<think> </think> tags. And return the final answer in "
            f"<answer> </answer> tags, for example <answer> (1 + 2) / 3 "
            f"</answer>.")


def _messages(problem):
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction(problem["numbers"], problem["target"])}]


_ALLOWED = re.compile(r"^[\d\s+\-*/().]+$")


def extract_answer(text):
    m = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return m[-1].strip() if m else None


def compute_reward(completion, numbers, target):
    """1.0 iff the <answer> is a valid +,-,*,/ expression that uses each
    provided number exactly once and evaluates to target; else 0.0."""
    expr = extract_answer(completion)
    if expr is None or not _ALLOWED.match(expr) or "**" in expr or "//" in expr:
        return 0.0  # ** and // slip past the char class but aren't allowed ops
    if Counter(int(x) for x in re.findall(r"\d+", expr)) != Counter(numbers):
        return 0.0  # must use each provided number once (also catches bare-target hack)
    try:
        return 1.0 if abs(eval(expr, {"__builtins__": {}}, {}) - target) < 1e-9 else 0.0
    except Exception:
        return 0.0


def correct(completion, info):
    text = completion[-1].content if completion else None
    return compute_reward(text if isinstance(text, str) else "",
                          info["numbers"], info["target"])


def _rows(rng, n, **gen_args):
    return [{"prompt": _messages(p), "answer": p["solution"],
             "info": {"numbers": p["numbers"], "target": p["target"]}}
            for p in (generate(rng, **gen_args) for _ in range(n))]


def load_environment(num_operands=4, min_number=1, max_number=100, max_target=1000,
                     dataset_size=20_000, num_eval_examples=128, seed=0):
    """Standard verifiers entry point. Fixed seeded datasets stand in for
    on-the-fly generation: training draws without replacement (~1s per 20k
    problems at import), eval reuses the same held-out problems every run."""
    from datasets import Dataset
    gen_args = dict(num_operands=num_operands, min_number=min_number,
                    max_number=max_number, max_target=max_target)
    return vf.SingleTurnEnv(
        dataset=Dataset.from_list(_rows(random.Random(seed), dataset_size, **gen_args)),
        eval_dataset=Dataset.from_list(
            _rows(random.Random(seed + 999), num_eval_examples, **gen_args)),
        rubric=vf.Rubric(funcs=[correct]))


if __name__ == "__main__":
    rng = random.Random(0)
    for _ in range(3):
        ex = generate(rng)
        print(ex)
        assert compute_reward(f"<answer>{ex['solution']}</answer>",
                              ex["numbers"], ex["target"]) == 1.0
    # reward edge cases
    nums, tgt = [3, 5, 7], 26
    for name, c in {
        "correct": "<answer>3 * 7 + 5</answer>",
        "wrong value": "<answer>3 + 5 + 7</answer>",
        "reuses number": "<answer>3 * 7 + 5 + 5</answer>",
        "unknown number": "<answer>3 * 7 + 9</answer>",
        "no tag": "the answer is 3*7+5",
        "bare target in answer": "<think>3*7+5</think><answer>26</answer>",
    }.items():
        print(f"{name:24s} -> {compute_reward(c, nums, tgt)}")
    env = load_environment(dataset_size=8, num_eval_examples=4)
    row = env.get_dataset()[0]
    good = [vf.AssistantMessage(content=f"<answer>{row['answer']}</answer>")]
    assert correct(good, row["info"]) == 1.0 and correct([], row["info"]) == 0.0
    print(f"ok: env of {len(env.get_dataset())}+{len(env.get_eval_dataset())} rows, "
          "rubric scores the recorded solutions")
