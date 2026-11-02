"""Stopping the broker: signals reach Scheduler.stop(), and run() always ends in shutdown()."""

import json
import signal
import threading
import time

import pytest

from gpuworker import cli

from helpers import cpu_jt, make_scheduler, registry


@pytest.fixture
def restore_signals():
    saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    yield
    for s, h in saved.items():
        signal.signal(s, h)


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_stop_signal_ends_the_loop_through_shutdown(tmp_path, restore_signals, sig):
    # Under systemd a stop is SIGTERM. It must run the same path as Ctrl-C: stop the loop,
    # then shutdown(), which tears warm serve containers down so they release VRAM.
    sched = make_scheduler(tmp_path, registry(cpu_jt()))
    cli.install_stop_handlers(sched)
    t = threading.Thread(target=sched.run, kwargs={"interval_s": 30.0}, daemon=True)
    t.start()
    time.sleep(0.2)
    signal.getsignal(sig)(sig, None)  # what the interpreter does on delivery
    t.join(5.0)
    assert not t.is_alive()  # the 30 s tick wait wakes at once
    assert sched._draining  # shutdown() ran


def test_run_publishes_limits_once_started(tmp_path):
    sched = make_scheduler(tmp_path, registry(cpu_jt()))
    t = threading.Thread(target=sched.run, kwargs={"interval_s": 0.05}, daemon=True)
    t.start()
    time.sleep(0.2)
    sched.stop()
    t.join(5.0)
    assert (tmp_path / "Q" / "broker_limits.json").exists()


def test_a_second_broker_on_the_same_state_dir_refuses_to_start(tmp_path, capsys):
    """Startup clears batches/ and serve/ and reaps serve containers, so a second broker
    started by mistake destroyed the running one's state."""
    import os

    from gpuworker.lock import BrokerRunning

    first = make_scheduler(tmp_path, registry(cpu_jt()))
    t = threading.Thread(target=first.run, kwargs={"interval_s": 0.05}, daemon=True)
    t.start()
    try:
        for _ in range(200):
            if (tmp_path / "state" / "broker.lock").exists():
                break
            time.sleep(0.01)
        live = tmp_path / "state" / "batches" / "live-batch"
        live.mkdir(parents=True)
        second = make_scheduler(tmp_path, registry(cpu_jt()))
        with pytest.raises(BrokerRunning, match=f"pid {os.getpid()}"):
            second.run(interval_s=0.05)
        assert live.is_dir()  # the running broker's state is untouched
        cfg = tmp_path / "gpuworker.toml"
        cfg.write_text(
            'container_runtime = "local"\nregistry_dir = "modules"\n'
            f"state_dir = {json.dumps(str(tmp_path / 'state'))}\n"
            f"queue_roots = [{json.dumps(str(tmp_path / 'Q2'))}]\n"
            f"status_path = {json.dumps(str(tmp_path / 'status2.json'))}\n"
            f"footprints_path = {json.dumps(str(tmp_path / 'fp2.json'))}\n"
        )
        assert cli.main(["--config", str(cfg), "--mock-gpu", "serve"]) == 1
        assert "another broker" in capsys.readouterr().err
    finally:
        first.stop()
        t.join(5.0)
    again = make_scheduler(tmp_path, registry(cpu_jt()))  # released on exit
    t = threading.Thread(target=again.run, kwargs={"interval_s": 0.05}, daemon=True)
    t.start()
    time.sleep(0.2)
    again.stop()
    t.join(5.0)
    assert not t.is_alive()


def test_loop_errors_are_logged_with_a_traceback_backed_off_and_shown_in_status(tmp_path):
    from gpuworker.obs import JobLogger

    log = tmp_path / "events.jsonl"
    sched = make_scheduler(tmp_path, registry(cpu_jt()))
    sched.log = JobLogger.to_path(str(log))

    def broken_tick():
        raise RuntimeError("disk on fire")

    sched.tick = broken_tick
    t = threading.Thread(target=sched.run, kwargs={"interval_s": 0.01}, daemon=True)
    t.start()
    time.sleep(0.6)
    sched.stop()
    t.join(5.0)
    sched.log.close()
    errors = [
        e for e in map(json.loads, log.read_text().splitlines()) if e["event"] == "tick_error"
    ]
    assert errors and "broken_tick" in errors[0]["traceback"]
    assert [e["consecutive"] for e in errors] == list(range(1, len(errors) + 1))
    assert len(errors) <= 8  # backed off: 0.6 s at 0.01 s per tick would be about 60
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["last_loop_error"]["error"] == "RuntimeError: disk on fire"


def test_a_clean_tick_clears_the_last_loop_error(tmp_path):
    """last_loop_error was never reset, so status.json showed an old error forever."""
    sched = make_scheduler(tmp_path, registry(cpu_jt()))
    real_tick, calls = sched.tick, []

    def tick_that_fails_once():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("share hiccup")
        real_tick()

    sched.tick = tick_that_fails_once
    t = threading.Thread(target=sched.run, kwargs={"interval_s": 0.01}, daemon=True)
    t.start()
    deadline = time.monotonic() + 5.0
    while len(calls) < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    sched.stop()
    t.join(5.0)
    assert len(calls) >= 3
    assert sched.last_loop_error is None
    assert json.loads((tmp_path / "status.json").read_text())["last_loop_error"] is None


class _Reaper:
    """A launcher stand-in that records the startup reaps."""

    def __init__(self):
        self.reaped = []

    def reap_serve_orphans(self, prefix=""):
        pass

    def reap_jobs(self, prefix="", exited_only=True):
        self.reaped.append((prefix, exited_only))


def test_a_restarted_broker_removes_the_oneshot_containers_it_left_running(tmp_path):
    """Only exited oneshot containers were reaped at start, so one a crashed broker left
    running kept its VRAM. The broker's id is kept in state_dir, so the next broker on that
    state_dir knows the names; another broker's and run-once's have other ids."""
    from gpuworker.launcher import JOB_NAME_PREFIX

    prefixes = []
    for _ in range(2):
        sched = make_scheduler(tmp_path, registry(cpu_jt()))
        reaper = _Reaper()
        sched.launcher = reaper
        t = threading.Thread(target=sched.run, kwargs={"interval_s": 0.01}, daemon=True)
        t.start()
        time.sleep(0.2)
        sched.stop()
        t.join(5.0)
        assert reaper.reaped == [(sched.job_name_prefix, False), (JOB_NAME_PREFIX, True)]
        assert sched._worker_ctx(cpu_jt()).job_name_prefix == sched.job_name_prefix
        prefixes.append(sched.job_name_prefix)
    assert prefixes[0] == prefixes[1] != JOB_NAME_PREFIX
    other = make_scheduler(tmp_path / "other", registry(cpu_jt()))
    assert other.job_name_prefix != prefixes[0]  # a run-once or another state_dir


def test_start_logs_the_bundles_left_in_processing_and_when_they_run_again(tmp_path):
    """After a crash a bundle waits reclaim_after_s (two hours by default) in processing/
    with nothing in the log to say so."""
    import datetime
    import os

    from gpuworker.obs import JobLogger

    log = tmp_path / "events.jsonl"
    sched = make_scheduler(tmp_path, registry(cpu_jt()))
    sched.log = JobLogger.to_path(str(log))
    q = sched.queues[0].ensure()
    for name in ("left.zip", "done.zip"):
        with open(os.path.join(q.dirs["processing"], name), "w") as fh:
            fh.write("x")
    with open(os.path.join(q.dirs["outbox"], "done.zip"), "w") as fh:
        fh.write("result")  # archived, not run again
    claimed = os.path.getmtime(os.path.join(q.dirs["processing"], "left.zip"))
    t = threading.Thread(target=sched.run, kwargs={"interval_s": 0.01}, daemon=True)
    t.start()
    time.sleep(0.2)
    sched.stop()
    t.join(5.0)
    sched.log.close()
    events = list(map(json.loads, log.read_text().splitlines()))
    left = [e for e in events if e["event"] == "left_in_processing"]
    assert [e["job_id"] for e in left] == ["left.zip"]
    when = datetime.datetime.fromisoformat(left[0]["runs_again_after"]).timestamp()
    assert abs(when - (claimed + sched.cfg.reclaim_after_s)) < 2
