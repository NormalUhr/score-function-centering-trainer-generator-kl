"""Launch a wave of training runs packed onto GPU slices of one node.

Reads a wave manifest (experiments/*.yaml), expands each arm over the seeds,
skips runs whose logs already reach their final step, and keeps every GPU
slice busy until the queue is empty. Each launch is appended to
launched/<wave>.jsonl, and a status line with the last logged step of every
running arm is printed every poll.

    python scripts/wave_driver.py experiments/quantized_sampler.yaml \
        --group Qf8 --gpus <ids> --gpus-per-run <n>

The shared 0.6B Countdown protocol lives in PROTOCOL; an arm's overrides come
after it and win.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

PROTOCOL = [
    "model.source=Qwen/Qwen3-0.6B", "env.id=countdown", "env.prompt_format=chat",
    "log.project=score-centering", "log.wandb_mode=disabled",
    "rl.num_prompts=64", "rl.seq_len=512", "eval.num_prompts=64",
    "opt.optimizer=sgd", "opt.lr=0.01",
]
PROJECT = "score-centering"


@dataclass
class Run:
    name: str
    overrides: list
    steps: int


def _value(overrides, key, default=None):
    for item in reversed(overrides):
        if item.startswith(key + "="):
            return item.split("=", 1)[1]
    return default


def expand_arms(manifest, seeds, arm_filters=(), group=None):
    """One Run per (arm, seed). Arms are selected by their `group` field when
    a group is given and by name prefix when filters are given; both apply
    when both are given, and every arm qualifies when neither is. An arm's
    own `seeds` list overrides the command-line seeds. Longer runs come first
    so the wave's tail is short."""
    runs = []
    for arm in manifest["arms"]:
        if group is not None and arm.get("group") != group:
            continue
        if arm_filters and not any(arm["name"].startswith(f) for f in arm_filters):
            continue
        steps = _value(arm["overrides"].split(), "stop.steps")
        if steps is None:
            raise ValueError(f"arm {arm['name']} sets no stop.steps, so completion cannot be judged")
        for seed in arm.get("seeds", seeds):
            overrides = arm["overrides"].split() + [f"run.seed={seed}"]
            runs.append(Run(arm["name"].format(seed=seed), overrides, int(steps)))
    return sorted(runs, key=lambda run: -run.steps)


def gpu_slices(gpus, per_run):
    if "-" in gpus:
        first, last = map(int, gpus.split("-"))
        ids = list(range(first, last + 1))
    else:
        ids = [int(g) for g in gpus.split(",")]
    if len(ids) % per_run:
        raise ValueError(f"{len(ids)} GPUs do not split into slices of {per_run}")
    return [",".join(map(str, ids[i:i + per_run])) for i in range(0, len(ids), per_run)]


def last_step(root, name):
    """Last logged step, or None. A trailing line that the trainer is still
    writing does not parse and counts as no newer step."""
    log = Path(root) / PROJECT / name / "logs.jsonl"
    if not log.exists():
        return None
    last = None
    with open(log) as f:
        for line in f:
            try:
                last = json.loads(line)["step"]
            except (json.JSONDecodeError, KeyError):
                continue
    return last


STOPPED_MARKER = "STOPPED"  # placed by hand in a run directory that was stopped on purpose


def is_complete(root, run):
    """A run is complete when its log reaches the final step, or when it was
    stopped on purpose (a STOPPED file in its directory records why)."""
    if (Path(root) / PROJECT / run.name / STOPPED_MARKER).exists():
        return True
    step = last_step(root, run.name)
    return step is not None and step >= run.steps - 1


def clear_partial(root, run):
    """Move an incomplete run directory aside before a relaunch: the trainer
    appends to logs.jsonl, so relaunching on top of it would interleave two
    runs' steps, and deleting it would destroy evidence of the failed run."""
    run_dir = Path(root) / PROJECT / run.name
    if run_dir.exists():
        run_dir.rename(run_dir.with_name(f"{run.name}.partial-{time.strftime('%Y%m%dT%H%M%S')}"))


def train_command(run, root, code):
    return [sys.executable, f"{code}/train_rl.py", *PROTOCOL,  # the environment of the launch reaches every run process
            f"model.weights_dir={root}/weights", f"log.dir={root}/outputs",
            f"log.run_name={run.name}", *run.overrides]


def live_processes(run_name):
    """Pids of trainers launched for this run name and still alive, or an
    empty list; a driver killed with SIGKILL leaves its trainers behind."""
    result = subprocess.run(["pgrep", "-f", f"log.run_name={run_name}( |$)"],
                            capture_output=True, text=True)
    return [int(pid) for pid in result.stdout.split()]


def launch(run, gpus, root, code):
    log = open(f"{root}/logs/{run.name}.log", "w")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpus, "WANDB_MODE": "disabled",
           "HF_HOME": f"{root}/hf", "PYTHONUNBUFFERED": "1"}
    return subprocess.Popen(train_command(run, root, code), cwd=code, env=env,
                            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--gpus", default="0-7")
    parser.add_argument("--gpus-per-run", type=int, default=2)
    parser.add_argument("--filter", nargs="*", default=[], help="arm-name prefixes (any match)")
    parser.add_argument("--group", default=None, help="run only arms with this `group` field")
    parser.add_argument("--root", default=os.environ.get("SC_ASYNC_ROOT", os.getcwd()))
    parser.add_argument("--code", default=None)
    parser.add_argument("--poll", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    code = os.path.realpath(args.code or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    with open(args.manifest) as f:
        manifest = yaml.safe_load(f)
    outputs = f"{args.root}/outputs"
    selected = expand_arms(manifest, args.seeds, args.filter, args.group)
    pending = [run for run in selected if not is_complete(outputs, run)]
    skipped = len(selected) - len(pending)
    print(f"wave={manifest['wave']} pending={len(pending)} skipped_complete={skipped} "
          f"slices={gpu_slices(args.gpus, args.gpus_per_run)}", flush=True)
    if args.dry_run:
        for run in pending:
            print(" ".join(train_command(run, args.root, code)))
        return
    Path(f"{args.root}/launched").mkdir(parents=True, exist_ok=True)
    Path(f"{args.root}/logs").mkdir(parents=True, exist_ok=True)
    record = open(f"{args.root}/launched/{manifest['wave']}.jsonl", "a")
    free = gpu_slices(args.gpus, args.gpus_per_run)
    schedule(pending, free, {}, record, args.root, code, outputs, args.poll)
    print("WAVE_DONE", flush=True)


def schedule(pending, free, running, record, root, code, outputs, poll):
    """Launch pending runs into free GPU slices until every run has exited.
    A run that still has live processes (a driver killed with SIGKILL leaves
    its trainers behind) is left out untouched: launching over it would
    double the run and moving its directory aside would lose its log."""
    while pending or running:
        while pending and free:
            gpus = free.pop(0)
            run = pending.pop(0)
            live = live_processes(run.name)
            if live:
                print(f"{time.strftime('%H:%M:%S')} skipped {run.name}: live processes {live}", flush=True)
                free.insert(0, gpus)
                continue
            clear_partial(outputs, run)
            process = launch(run, gpus, root, code)
            running[gpus] = (run, process, time.time())
            record.write(json.dumps({"run": run.name, "gpus": gpus, "pid": process.pid,
                                     "start": time.time()}) + "\n")
            record.flush()
            print(f"{time.strftime('%H:%M:%S')} launched {run.name} on GPUs {gpus} pid {process.pid}", flush=True)
        time.sleep(poll)
        for gpus, (run, process, start) in list(running.items()):
            code_ = process.poll()
            if code_ is None:
                continue
            del running[gpus]
            free.append(gpus)
            record.write(json.dumps({"run": run.name, "gpus": gpus, "pid": process.pid,
                                     "end": time.time(), "returncode": code_,
                                     "last_step": last_step(outputs, run.name)}) + "\n")
            record.flush()
            print(f"{time.strftime('%H:%M:%S')} finished {run.name} rc={code_} "
                  f"last_step={last_step(outputs, run.name)} in {time.time() - start:.0f}s", flush=True)
        status = " ".join(f"{run.name}@{last_step(outputs, run.name)}" for run, _, _ in running.values())
        print(f"{time.strftime('%H:%M:%S')} running={len(running)} pending={len(pending)} {status}", flush=True)


if __name__ == "__main__":
    main()
