"""The collapse rule behind the "stopped at N (collapsed)" rows of the results
table. A run is stopped once it has logged at least 400 steps and its mean
training reward over the last 200 steps is below a floor: a sustained
collapse to near zero or to a quarter of what it had reached. Prints one line
per run, STOP or ok, with the numbers.

    python scripts/stop_rule.py results/runs Lst4-sc-s0 Lst2-sc-s0

Floors. Most settings: max(0.05, 0.25 x the best 200-step mean). The
quantized-sampler study (Q prefixes) trains a Base model whose heaviest
sampler keeps the training reward under 0.05 while the policy still learns
(the clean-sampler evaluation rises), so its absolute floor is 0.01. The
2-prompt Countdown setting (Lst2) dips deeply and recovers, and a collapse
there shows as a reward near zero with nearly every completion at the token
limit, so it uses an absolute floor of 0.02 only.
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
import at_a_glance  # noqa: E402


def floor_of(run, best):
    if run.startswith("Lst2-"):
        return 0.02
    if run.startswith("Q"):
        return max(0.01, 0.25 * best)
    return max(0.05, 0.25 * best)


def verdict(run, rewards):
    """(verdict, steps, last-200 mean, best 200-step mean, floor); the verdict
    is "ok" below 400 steps."""
    n = len(rewards)
    if n < 400:
        return "ok", n, None, None, None
    last = float(np.mean(rewards[-200:]))
    best = max(float(np.mean(rewards[i:i + 200])) for i in range(0, n - 199, 50))
    floor = floor_of(run, best)
    return ("STOP" if last < floor else "ok"), n, last, best, floor


def main():
    root, runs = Path(sys.argv[1]), sys.argv[2:]
    for run in runs:
        run_dir = root / run
        if (run_dir / "STOPPED").exists():
            print(f"{run} ok (already stopped)")
            continue
        path = at_a_glance.log_path(run_dir, "logs.jsonl")
        if path is None:
            print(f"{run} ok (no log)")
            continue
        rewards = [row["reward"] for row in at_a_glance.read_jsonl(path)]
        kind, n, last, best, floor = verdict(run, rewards)
        if last is None:
            print(f"{run} ok (steps {n} < 400)")
        else:
            print(f"{run} {kind} steps={n} last200={last:.3f} best200={best:.3f} floor={floor:.3f}")


if __name__ == "__main__":
    main()
