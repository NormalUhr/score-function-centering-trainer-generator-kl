"""Tests of the asynchronous driver's split roles: a trainer-only driver on
one machine and a generators-only driver on another share an exchange
directory on a shared filesystem."""
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(__file__))
import async_driver  # noqa: E402


def test_slots_per_role():
    assert async_driver.slots("0-7", 8, 4, 2, role="trainer") == [("0,1,2,3,4,5,6,7", ())]
    assert async_driver.slots("0-7", 8, 4, 2, role="generators") == [("", ("0,1,2,3", "4,5,6,7"))]
    assert async_driver.slots("0-7", 2, 1, 6, role="both") == [("0,1", ("2", "3", "4", "5", "6", "7"))]


class FakeProcess:
    def __init__(self, command, gpus, log_path):
        self.command, self.gpus, self.log_path = command, gpus, log_path
        self.pid, self.returncode = 1, None

    def poll(self):
        return self.returncode

    def kill(self):
        self.returncode = -9

    def wait(self):
        return self.returncode


def fake_start(started, returncode=None):
    def start(command, gpus, log_path, code, root):
        process = FakeProcess(command, gpus, log_path)
        process.returncode = returncode
        started.append(process)
        return process
    return start


def run_with(monkeypatch, tmp, role, slot, hosts=async_driver.TrainerHosts(), generators=2, generator_offset=0):
    started = []
    monkeypatch.setattr(async_driver, "start", fake_start(started))
    run = SimpleNamespace(name="p30ba-sc-klr-s0", overrides=["stop.steps=3"])
    live = async_driver.AsyncRun(run, slot, tmp, tmp, f"{tmp}/exchange", generators, role=role, hosts=hosts,
                                 generator_offset=generator_offset)
    return live, started


COORDINATOR = "127.0.0.1:29400"


def test_trainer_host_arguments():
    """The driver takes the trainer's host count, this machine's index and the
    coordinator address, and forwards them to the trainer as overrides; a
    single host adds nothing to the command."""
    args = async_driver.build_parser().parse_args(["m.yaml"])
    assert (args.trainer_hosts, args.trainer_host_index, args.coordinator) == (1, 0, None)
    args = async_driver.build_parser().parse_args(
        ["m.yaml", "--trainer-hosts", "2", "--trainer-host-index", "1", "--coordinator", COORDINATOR])
    assert (args.trainer_hosts, args.trainer_host_index, args.coordinator) == (2, 1, COORDINATOR)
    assert async_driver.TrainerHosts(2, 1, COORDINATOR).overrides() == [
        "async.trainer_hosts=2", "async.trainer_host_index=1", f"async.coordinator={COORDINATOR}"]
    assert async_driver.TrainerHosts().overrides() == []


def test_first_trainer_host_owns_the_exchange(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        exchange = Path(tmp) / "exchange" / "p30ba-sc-klr-s0" / "weights"
        exchange.mkdir(parents=True)
        (exchange / "LATEST").write_text("0")  # left by an earlier run
        live, started = run_with(monkeypatch, tmp, "trainer", ("0,1,2,3", ()),
                                 hosts=async_driver.TrainerHosts(2, 0, COORDINATOR))
        assert not (exchange / "LATEST").exists(), "trainer host 0 clears the exchange"
        assert started[0].command[-4:] == ["role=trainer", "async.trainer_hosts=2",
                                           "async.trainer_host_index=0", f"async.coordinator={COORDINATOR}"]
        assert started[0].log_path == f"{tmp}/logs/p30ba-sc-klr-s0.log"


def test_second_trainer_host_leaves_the_exchange_and_logs_apart(monkeypatch):
    """Another trainer host joins host 0's program: it must not clear the
    exchange that host 0 owns, and its log must not collide with host 0's on
    the shared filesystem."""
    with tempfile.TemporaryDirectory() as tmp:
        exchange = Path(tmp) / "exchange" / "p30ba-sc-klr-s0" / "weights"
        exchange.mkdir(parents=True)
        (exchange / "LATEST").write_text("0")  # published by host 0
        live, started = run_with(monkeypatch, tmp, "trainer", ("4,5,6,7", ()),
                                 hosts=async_driver.TrainerHosts(2, 1, COORDINATOR))
        assert (exchange / "LATEST").exists()
        assert started[0].command[-3:] == ["async.trainer_hosts=2", "async.trainer_host_index=1",
                                           f"async.coordinator={COORDINATOR}"]
        assert started[0].log_path == f"{tmp}/logs/p30ba-sc-klr-s0-host1.log"
        assert live.generators == []
        started[0].returncode = 0
        assert live.finished()
        assert Path(live.exchange).exists()


def test_trainer_role_starts_only_the_trainer(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        live, started = run_with(monkeypatch, tmp, "trainer", ("0,1,2,3,4,5,6,7", ()))
        assert [p.gpus for p in started] == ["0,1,2,3,4,5,6,7"]
        assert "role=trainer" in started[0].command
        assert "async.generators=2" in started[0].command  # the remote generators' count
        assert live.generators == []
        Path(live.exchange).mkdir(parents=True)  # written by the trainer process in a real run
        started[0].returncode = 0
        assert live.finished()
        # The remote generators still read the exchange after the trainer exits.
        assert Path(live.exchange).exists()


def test_generators_role_starts_only_generators_and_keeps_the_exchange(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        exchange = Path(tmp) / "exchange" / "p30ba-sc-klr-s0" / "weights"
        exchange.mkdir(parents=True)
        (exchange / "LATEST").write_text("0")  # published by the trainer on the other machine
        live, started = run_with(monkeypatch, tmp, "generators", ("", ("0,1,2,3", "4,5,6,7")))
        assert (exchange / "LATEST").exists(), "a generators-only driver must not clear the exchange"
        assert [p.gpus for p in started] == ["0,1,2,3", "4,5,6,7"]
        assert all("role=generator" in p.command for p in started)
        assert live.trainer is None
        assert not live.finished()
        for p in started:
            p.returncode = 0
        assert live.finished()
        assert not Path(live.exchange).exists(), "the generators' side removes the exchange at the end"


def test_both_role_starts_the_trainer_and_the_generators(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        live, started = run_with(monkeypatch, tmp, "both", ("0,1", ("2", "3")))
        assert [p.gpus for p in started] == ["0,1", "2", "3"]
        assert "role=trainer" in started[0].command


def test_live_processes_looks_only_for_this_drivers_roles():
    """A generators-only driver may share its machine with the run's trainer (and
    a trainer-only driver with the generators), so the check for processes
    left behind by an earlier driver must look for this driver's roles only."""
    import subprocess
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)",
                                "log.run_name=live-check-run", "async.exchange_dir=/tmp/x", "role=trainer",
                                "async.trainer_hosts=2", "async.trainer_host_index=1", f"async.coordinator={COORDINATOR}"])
    try:
        for _ in range(50):  # until the process is listed
            if async_driver.live_processes("live-check-run") == [process.pid]:
                break
            time.sleep(0.1)
        assert async_driver.live_processes("live-check-run") == [process.pid]
        assert async_driver.live_processes("live-check-run", role="trainer") == [process.pid]
        assert async_driver.live_processes("live-check-run", role="generators") == []
        assert async_driver.live_processes("live-check-run", role="both") == [process.pid]
        # Trainer hosts of one run may share a machine (a test layout): each looks for its own index.
        two = async_driver.TrainerHosts(2, 1, COORDINATOR)
        assert async_driver.live_processes("live-check-run", role="trainer", hosts=two) == [process.pid]
        assert async_driver.live_processes("live-check-run", role="trainer",
                                           hosts=async_driver.TrainerHosts(2, 0, COORDINATOR)) == []
    finally:
        process.kill()
        process.wait()


def test_only_trainer_host_0_clears_the_run_directory(monkeypatch):
    """Trainer host 0 owns outputs/<run>; a generators-only driver or another
    trainer host that started after it must not move the fresh run directory
    aside."""
    cleared = []
    monkeypatch.setattr(async_driver, "clear_partial", lambda outputs, run: cleared.append(run.name))
    monkeypatch.setattr(async_driver, "live_processes", lambda name, role, hosts: [])
    monkeypatch.setattr(async_driver, "last_step", lambda outputs, name: 0)
    started = []
    monkeypatch.setattr(async_driver, "start", fake_start(started, returncode=0))  # processes exit at once
    run = SimpleNamespace(name="p30ba-sc-klr-s0", overrides=[])
    with tempfile.TemporaryDirectory() as tmp:
        for role, hosts in (("generators", async_driver.TrainerHosts()),
                            ("trainer", async_driver.TrainerHosts(2, 1, COORDINATOR)),
                            ("trainer", async_driver.TrainerHosts(2, 0, COORDINATOR))):
            args = SimpleNamespace(root=tmp, exchange_root=f"{tmp}/exchange", generators=2, role=role, generator_index_offset=0, poll=0.0)
            slot = ("", ("0,1,2,3", "4,5,6,7")) if role == "generators" else ("0,1,2,3", ())
            with open(f"{tmp}/record.jsonl", "a") as record:
                async_driver.schedule([run], [slot], {}, record, args, tmp, tmp, hosts)
    assert cleared == ["p30ba-sc-klr-s0"], cleared


def test_generators_over_two_machines_get_global_indices(monkeypatch):
    """A machine that starts one of a run's two generators with an index offset
    passes the run's total to the processes and numbers its generator and its
    log by the global index."""
    with tempfile.TemporaryDirectory() as tmp:
        exchange = Path(tmp) / "exchange" / "p30ba-sc-klr-s0" / "weights"
        exchange.mkdir(parents=True)
        live, started = run_with(monkeypatch, tmp, "generators", ("", ("0,1,2,3,4,5,6,7",)),
                                 generators=2, generator_offset=1)
        assert len(started) == 1
        assert "async.generators=2" in started[0].command and "async.generator_index=1" in started[0].command
        assert started[0].log_path.endswith("p30ba-sc-klr-s0-gen1.log")
        assert live.trainer is None


def test_parser_local_generators_and_offset():
    args = async_driver.build_parser().parse_args(
        ["m.yaml", "--generators", "2", "--local-generators", "1", "--generator-index-offset", "1"])
    assert (args.generators, args.local_generators, args.generator_index_offset) == (2, 1, 1)
    assert async_driver.build_parser().parse_args(["m.yaml"]).local_generators is None


def test_run_processes_use_the_drivers_interpreter():
    """Every run process starts with the interpreter the driver runs under,
    so a driver started from a virtual environment starts the trainer and
    the generators in it too."""
    run = SimpleNamespace(name="demo-s0", overrides=["stop.steps=3"])
    command = async_driver.common_command(run, "/work", "/work/code", "/tmp/x", 6)
    assert command[0] == sys.executable, command[0]
    assert command[1] == "/work/code/async_rl.py"
