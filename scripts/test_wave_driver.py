"""Pure parts of the wave driver: arm expansion over seeds, completed-run
detection, and GPU slice construction."""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wave_driver


def skips_live_runs():
    """A run whose trainer is still alive is left alone: its directory is not
    moved aside, nothing is launched over it, and the wave goes on."""
    import types
    calls = {"clear": [], "launch": []}
    saved = (wave_driver.live_processes, wave_driver.clear_partial, wave_driver.launch, wave_driver.last_step)
    wave_driver.live_processes = lambda name: [4242] if name == "r1-pg-s0" else []
    wave_driver.clear_partial = lambda outputs, run: calls["clear"].append(run.name)
    wave_driver.launch = lambda run, gpus, root, code: (calls["launch"].append(run.name),
                                                       types.SimpleNamespace(pid=1, poll=lambda: 0, returncode=0))[1]
    wave_driver.last_step = lambda outputs, name: 2
    try:
        runs = wave_driver.expand_arms({"wave": "toy", "arms": [
            {"name": "r1-pg-s{seed}", "overrides": "stop.steps=3"},
            {"name": "r1-sc-s{seed}", "overrides": "stop.steps=3"}]}, seeds=[0], arm_filters=[])
        with tempfile.TemporaryDirectory() as root:
            record = open(f"{root}/record.jsonl", "a")
            wave_driver.schedule(list(runs), ["0,1"], {}, record, root, root, root, poll=0.0)
        assert calls["clear"] == ["r1-sc-s0"], calls
        assert calls["launch"] == ["r1-sc-s0"], calls
    finally:
        wave_driver.live_processes, wave_driver.clear_partial, wave_driver.launch, wave_driver.last_step = saved
    print("ok: a live run is skipped, not clobbered")


def main():
    skips_live_runs()
    manifest = {
        "wave": "toy",
        "arms": [
            {"name": "r1-pg-s{seed}", "overrides": "stop.steps=3 rl.is.level=null"},
            {"name": "r1-sc-s{seed}", "overrides": "stop.steps=3 rl.sc=true sampler.vocab_logprobs=128"},
        ],
    }
    runs = wave_driver.expand_arms(manifest, seeds=[0, 1], arm_filters=[])
    assert [run.name for run in runs] == ["r1-pg-s0", "r1-pg-s1", "r1-sc-s0", "r1-sc-s1"]
    assert runs[1].overrides[-1] == "run.seed=1"
    assert runs[1].steps == 3
    assert [run.name for run in wave_driver.expand_arms(manifest, [0], ["r1-sc"])] == ["r1-sc-s0"]
    # Several prefixes select the union of their arms.
    both = wave_driver.expand_arms(manifest, [0], ["r1-sc", "r1-pg"])
    assert [run.name for run in both] == ["r1-pg-s0", "r1-sc-s0"]
    # A group selects arms exactly, so a prefix like "r2-tis-s" can never
    # also catch "r2-tis-sc-s...".
    grouped = {"wave": "g", "arms": [
        {"name": "r2-tis-s{seed}", "group": "A", "overrides": "stop.steps=2"},
        {"name": "r2-tis-sc-s{seed}", "group": "B", "overrides": "stop.steps=2"},
    ]}
    assert [r.name for r in wave_driver.expand_arms(grouped, [0], [], group="A")] == ["r2-tis-s0"]
    assert [r.name for r in wave_driver.expand_arms(grouped, [0], [], group="B")] == ["r2-tis-sc-s0"]
    # An arm's own seed list overrides the command-line seeds, and longer
    # runs are queued first so the wave's tail is short.
    mixed = {"wave": "mixed", "arms": [
        {"name": "short-s{seed}", "overrides": "stop.steps=3", "seeds": [0, 1, 2]},
        {"name": "long-s{seed}", "overrides": "stop.steps=12", "seeds": [0, 1]},
    ]}
    names = [run.name for run in wave_driver.expand_arms(mixed, seeds=[7], arm_filters=[])]
    assert names == ["long-s0", "long-s1", "short-s0", "short-s1", "short-s2"], names

    slices = wave_driver.gpu_slices("0-7", per_run=2)
    assert slices == ["0,1", "2,3", "4,5", "6,7"]
    assert wave_driver.gpu_slices("2,3,4,5", per_run=4) == ["2,3,4,5"]
    assert wave_driver.gpu_slices("0-7", per_run=8) == ["0,1,2,3,4,5,6,7"]

    with tempfile.TemporaryDirectory() as root:
        done = Path(root) / "score-centering" / "r1-pg-s0"
        done.mkdir(parents=True)
        with open(done / "logs.jsonl", "w") as f:
            for step in range(3):
                f.write(json.dumps({"step": step, "reward": 0.1}) + "\n")
        partial = Path(root) / "score-centering" / "r1-pg-s1"
        partial.mkdir(parents=True)
        with open(partial / "logs.jsonl", "w") as f:
            f.write(json.dumps({"step": 0, "reward": 0.1}) + "\n")
        assert wave_driver.is_complete(root, runs[0]) is True
        assert wave_driver.is_complete(root, runs[1]) is False   # stopped early
        assert wave_driver.is_complete(root, runs[2]) is False   # never ran
        # A half-written trailing line (the trainer is appending) reads as
        # "no newer step" instead of ending the driver.
        with open(partial / "logs.jsonl", "a") as f:
            f.write('{"step": 1, "rew')
        assert wave_driver.last_step(root, runs[1].name) == 0
        # A relaunch starts from scratch: the trainer appends to logs.jsonl,
        # so a partial directory is moved aside (never deleted) first.
        wave_driver.clear_partial(root, runs[1])
        assert not partial.exists()
        assert len(list((Path(root) / "score-centering").glob("r1-pg-s1.partial-*"))) == 1
        wave_driver.clear_partial(root, runs[2])  # nothing to move is fine
        # A run stopped on purpose carries a STOPPED marker and counts as
        # complete, so a relaunched driver neither deletes nor repeats it.
        stopped = Path(root) / "score-centering" / "r1-sc-s0"
        stopped.mkdir(parents=True)
        with open(stopped / "logs.jsonl", "w") as f:
            f.write(json.dumps({"step": 0, "reward": 0.0}) + "\n")
        (stopped / "STOPPED").write_text("collapsed; stopped by hand\n")
        assert wave_driver.is_complete(root, runs[2]) is True
    # A manifest arm without stop.steps is rejected rather than treated as
    # never complete.
    try:
        wave_driver.expand_arms({"wave": "x", "arms": [{"name": "a-s{seed}", "overrides": "rl.sc=true"}]}, [0])
    except ValueError as error:
        assert "stop.steps" in str(error)
    else:
        raise AssertionError("arm without stop.steps accepted")

    command = wave_driver.train_command(runs[2], root="/work", code="/work/code")
    assert command[:2] == [sys.executable, "/work/code/train_rl.py"]  # the launching interpreter runs every run
    assert "log.run_name=r1-sc-s0" in command
    assert "rl.sc=true" in command and "run.seed=0" in command
    assert "log.dir=/work/outputs" in command
    # Protocol defaults come first so an arm override can replace them.
    assert command.index("rl.num_prompts=64") < command.index("stop.steps=3")
    print("ok: wave driver expansion, slices, completion detection, command")


if __name__ == "__main__":
    main()
