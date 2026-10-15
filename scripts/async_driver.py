"""Run an asynchronous wave on one node: for each arm, one trainer process and
several generator processes on disjoint GPU slices, fed through an exchange
directory (async_rl.py). The manifest format, the protocol, the completion
rule and the run records are those of wave_driver.py.

    python scripts/async_driver.py experiments/sfc_methods.yaml --group Lst4 \\
        --filter Lst4-sc-klr0.3 --gpus <ids> --trainer-gpus <n> --generator-gpus <n> --generators <n>

A slot is trainer_gpus + generators * generator_gpus GPUs; the GPU list is cut
into slots and one run occupies a slot until its trainer exits. Generators exit
on the trainer's DONE marker and are killed after a grace period otherwise.

A trainer can span several nodes: each node runs its own driver with --role
trainer, the same --trainer-hosts and --coordinator (host:port of the node with
--trainer-host-index 0) and its own --trainer-host-index; the trainer processes
join one JAX program over their GPUs. Host 0 owns the exchange, the run
directory and the logs, as a single-host trainer does.
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wave_driver import PROTOCOL, PROJECT, clear_partial, expand_arms, is_complete, last_step  # noqa: E402

GENERATOR_GRACE_SECONDS = 180


@dataclass(frozen=True)
class TrainerHosts:
    """The processes of a trainer that spans several nodes: how many, which
    one this node runs, and the address of host 0, which coordinates them."""
    count: int = 1
    index: int = 0
    coordinator: str = None

    def overrides(self):
        if self.count == 1:
            return []
        return [f"async.trainer_hosts={self.count}", f"async.trainer_host_index={self.index}",
                f"async.coordinator={self.coordinator}"]

    def log_suffix(self):
        """Distinguishes the hosts' trainer logs on a shared filesystem; host 0
        keeps the single-host name."""
        return "" if self.index == 0 else f"-host{self.index}"


def slots(gpus, trainer_gpus, generator_gpus, generators, role="both"):
    """Cut the GPU list into slots of (trainer slice, [generator slices]),
    `generators` being the generator processes this node starts. A
    trainer-only driver takes no generator slices and a generators-only
    driver no trainer slice: the other side runs on another node and shares
    the exchange directory through a shared filesystem."""
    if "-" in gpus:
        first, last = map(int, gpus.split("-"))
        ids = list(range(first, last + 1))
    else:
        ids = [int(g) for g in gpus.split(",")]
    trainer_gpus = trainer_gpus if role != "generators" else 0
    generators = generators if role != "trainer" else 0
    per_slot = trainer_gpus + generators * generator_gpus
    if len(ids) % per_slot:
        raise ValueError(f"{len(ids)} GPUs do not split into slots of {per_slot}")
    result = []
    for start in range(0, len(ids), per_slot):
        chunk = ids[start:start + per_slot]
        trainer = ",".join(map(str, chunk[:trainer_gpus]))
        gens = tuple(",".join(map(str, chunk[trainer_gpus + i * generator_gpus:
                                              trainer_gpus + (i + 1) * generator_gpus]))
                     for i in range(generators))
        result.append((trainer, gens))  # tuples: a slot is a dict key while it runs
    return result


def common_command(run, root, code, exchange, generators):
    return [sys.executable, f"{code}/async_rl.py", *PROTOCOL,  # the environment of the launch reaches every run process
            f"model.weights_dir={root}/weights", f"log.dir={root}/outputs",
            f"log.run_name={run.name}", f"async.exchange_dir={exchange}",
            f"async.generators={generators}", *run.overrides]


def live_processes(run_name, role="both", hosts=TrainerHosts()):
    """Pids of the processes this driver would start for the run name that
    are already running on this node (a driver killed with SIGKILL leaves them
    behind), or an empty list. Only this driver's processes count: a
    generators-only driver may share the node with the run's trainer, a
    trainer-only driver with the run's generators, and the trainer hosts of
    one run with each other (a test layout on one node)."""
    pattern = {"both": f"log.run_name={run_name}( |$)",
               "trainer": f"log.run_name={run_name} .*role=trainer",
               "generators": f"log.run_name={run_name} .*role=generator"}[role]
    if role == "trainer" and hosts.count > 1:
        pattern = f"log.run_name={run_name} .*async.trainer_host_index={hosts.index}( |$)"
    result = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return [int(pid) for pid in result.stdout.split()]


def start(command, gpus, log_path, code, root):
    log = open(log_path, "w")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpus, "WANDB_MODE": "disabled",
           "HF_HOME": f"{root}/hf", "PYTHONUNBUFFERED": "1"}
    return subprocess.Popen(command, cwd=code, env=env, stdout=log,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)


class AsyncRun:
    """One trainer and its generators, or one side of them when the trainer
    and the generators run on different nodes (`role`), or one host of a
    trainer that spans several nodes (`hosts`)."""

    def __init__(self, run, slot, root, code, exchange_root, generators, role="both",
                 hosts=TrainerHosts(), generator_offset=0):
        self.run, self.role, self.start_time = run, role, time.time()
        exchange = f"{exchange_root}/{run.name}"
        if role != "generators" and hosts.index == 0:  # trainer host 0 owns the exchange
            shutil.rmtree(exchange, ignore_errors=True)  # a stale DONE or queue would poison the run
        trainer_gpus, generator_slices = slot
        command = common_command(run, root, code, exchange, generators)  # `generators`: the run's total
        self.trainer = None
        if role != "generators":
            self.trainer = start(command + ["role=trainer", *hosts.overrides()], trainer_gpus,
                                 f"{root}/logs/{run.name}{hosts.log_suffix()}.log", code, root)
        # Generator indices are global to the run, so several nodes can each
        # start a share of the run's generators with different offsets.
        self.generators = [
            start(command + ["role=generator", f"async.generator_index={generator_offset + index}"], gpus,
                  f"{root}/logs/{run.name}-gen{generator_offset + index}.log", code, root)
            for index, gpus in enumerate(generator_slices)]
        self.exchange = exchange
        self.pids = [process.pid for process in self.processes()]

    def processes(self):
        return ([self.trainer] if self.trainer is not None else []) + self.generators

    def returncode(self):
        return self.trainer.returncode if self.trainer is not None else None

    def kill(self):
        for process in self.processes():
            if process.poll() is None:
                process.kill()
                process.wait()

    def finished(self):
        """True once the trainer has exited and every generator has followed
        (or was killed after the grace period). Without a local trainer the
        generators follow the DONE marker of the trainer on the other node.
        The exchange is removed by the side that runs the generators, since
        remote generators still read it after the trainer exits."""
        if self.trainer is not None and self.trainer.poll() is None:
            return False
        trainer_done = self.trainer is not None or Path(self.exchange, "DONE").exists()
        if trainer_done and not hasattr(self, "trainer_end"):
            self.trainer_end = time.time()
        for generator in self.generators:
            if generator.poll() is None:
                if hasattr(self, "trainer_end") and time.time() - self.trainer_end > GENERATOR_GRACE_SECONDS:
                    generator.kill()
                    generator.wait()
                else:
                    return False
        if self.role != "trainer":
            shutil.rmtree(self.exchange, ignore_errors=True)
        return True


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--gpus", default="0-7")
    parser.add_argument("--trainer-gpus", type=int, default=2)
    parser.add_argument("--generator-gpus", type=int, default=2)
    parser.add_argument("--generators", type=int, default=3, help="the run's generator processes in total")
    parser.add_argument("--local-generators", type=int, default=None,
                        help="generator processes started on this node (default: all of them)")
    parser.add_argument("--generator-index-offset", type=int, default=0,
                        help="index of this node's first generator when the run's generators span nodes")
    parser.add_argument("--filter", nargs="*", default=[], help="arm-name prefixes (any match)")
    parser.add_argument("--group", default=None, help="run only arms with this `group` field")
    parser.add_argument("--root", default=os.environ.get("SC_ASYNC_ROOT", os.getcwd()))
    parser.add_argument("--code", default=None)
    parser.add_argument("--exchange-root", default="/dev/shm/sc-async-exchange",
                        help="directory for weights and rollouts; node-local and RAM-backed by "
                             "default, on a shared filesystem when the roles run on two nodes")
    parser.add_argument("--role", choices=("both", "trainer", "generators"), default="both",
                        help="start both sides here, or only the trainer or only the generators "
                             "(the other side runs on another node with the same exchange root)")
    parser.add_argument("--trainer-hosts", type=int, default=1,
                        help="nodes the trainer spans; each runs a driver with --role trainer")
    parser.add_argument("--trainer-host-index", type=int, default=0, help="this node's index among them")
    parser.add_argument("--coordinator", default=None,
                        help="host:port of the node with --trainer-host-index 0 (required with several hosts)")
    parser.add_argument("--poll", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    hosts = TrainerHosts(args.trainer_hosts, args.trainer_host_index, args.coordinator)
    code = os.path.realpath(args.code or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    with open(args.manifest) as f:
        manifest = yaml.safe_load(f)
    outputs = f"{args.root}/outputs"
    selected = expand_arms(manifest, args.seeds, args.filter, args.group)
    pending = [run for run in selected if not is_complete(outputs, run)]
    local_generators = args.generators if args.local_generators is None else args.local_generators
    free = slots(args.gpus, args.trainer_gpus, args.generator_gpus, local_generators, args.role)
    print(f"wave={manifest['wave']} pending={len(pending)} "
          f"skipped_complete={len(selected) - len(pending)} slots={free}", flush=True)
    if args.dry_run:
        for run in pending:
            print(" ".join(common_command(run, args.root, code, f"{args.exchange_root}/{run.name}",
                                          args.generators) + hosts.overrides()))
        return
    Path(f"{args.root}/launched").mkdir(parents=True, exist_ok=True)
    Path(f"{args.root}/logs").mkdir(parents=True, exist_ok=True)
    record = open(f"{args.root}/launched/{manifest['wave']}.jsonl", "a")
    running = {}  # slot -> AsyncRun
    # A driver that dies must not leave trainers and generators behind: a
    # later launch of the same run would share their GPUs and exchange.
    # SIGTERM (kill without -9) becomes SystemExit so the cleanup below runs.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        schedule(pending, free, running, record, args, code, outputs, hosts)
    finally:
        for live in running.values():
            live.kill()
    print("WAVE_DONE", flush=True)


def schedule(pending, free, running, record, args, code, outputs, hosts=TrainerHosts()):
    while pending or running:
        while pending and free:
            slot = free.pop(0)
            run = pending.pop(0)
            stale = live_processes(run.name, args.role, hosts)
            if stale:
                # Launching over them would double the run; moving its
                # directory aside would lose its log. Leave it out.
                print(f"{time.strftime('%H:%M:%S')} skipped {run.name}: live processes {stale}", flush=True)
                free.insert(0, slot)
                continue
            if args.role != "generators" and hosts.index == 0:  # trainer host 0 owns outputs/<run>
                clear_partial(outputs, run)
            live = AsyncRun(run, slot, args.root, code, args.exchange_root, args.generators, args.role, hosts,
                            generator_offset=args.generator_index_offset)
            running[slot] = live
            record.write(json.dumps({"run": run.name, "trainer_gpus": slot[0],
                                     "generator_gpus": list(slot[1]), "pids": live.pids,
                                     "trainer_host": hosts.index, "start": time.time()}) + "\n")
            record.flush()
            print(f"{time.strftime('%H:%M:%S')} launched {run.name}: trainer on {slot[0]}, "
                  f"generators on {slot[1]}, pids {live.pids}", flush=True)
        time.sleep(args.poll)
        for slot, live in list(running.items()):
            if not live.finished():
                continue
            del running[slot]
            free.append(slot)
            record.write(json.dumps({"run": live.run.name, "end": time.time(),
                                     "returncode": live.returncode(),
                                     "generator_returncodes": [g.returncode for g in live.generators],
                                     "last_step": last_step(outputs, live.run.name)}) + "\n")
            record.flush()
            print(f"{time.strftime('%H:%M:%S')} finished {live.run.name} rc={live.returncode()} "
                  f"generators={[g.returncode for g in live.generators]} "
                  f"last_step={last_step(outputs, live.run.name)} in {time.time() - live.start_time:.0f}s",
                  flush=True)
        status = " ".join(f"{live.run.name}@{last_step(outputs, live.run.name)}" for live in running.values())
        print(f"{time.strftime('%H:%M:%S')} running={len(running)} pending={len(pending)} {status}", flush=True)


if __name__ == "__main__":
    main()
