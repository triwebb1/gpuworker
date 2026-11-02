"""End-to-end: submit an echo bundle, run the scheduler with the local launcher,
collect a terminal result. Proves transport + scheduler + execution + reliability
(the README quick start) without a container or GPU."""

import os
import time

from gpuworker import gpu
from gpuworker.client import submit, wait_result
from gpuworker.launcher import make_launcher
from gpuworker.manifest import JobManifest
from gpuworker.module import scan_registry
from gpuworker.scheduler import Scheduler

from helpers import make_config


def _make_scheduler(tmp_path, queue_root):
    cfg = make_config(
        tmp_path,
        queue_roots=[queue_root],
        registry_dir="modules",
        container_runtime="local",
        weights_dir=str(tmp_path / "weights"),
    )
    reg = scan_registry("modules")
    return Scheduler(cfg, reg, make_launcher("local"), gpu.MockGpu())


def _tick_until(sched, queue_root, filename, timeout_s=30):
    deadline = time.monotonic() + timeout_s
    out = os.path.join(queue_root, "outbox", filename)
    while time.monotonic() < deadline:
        sched.tick()
        if os.path.exists(out):
            # let a final tick flush completion bookkeeping
            sched.tick()
            return
        time.sleep(0.05)
    raise AssertionError("scheduler did not finish the batch in time")


def test_echo_job_end_to_end(tmp_path):
    queue_root = str(tmp_path / "Echo")
    m = JobManifest.from_dict(
        {
            "job_id": "echo1",
            "job_type": "echo.echo",
            "outputs": ["out/echo1.txt"],
            "options": {"text": "hello-broker"},
        }
    )
    fn = submit(queue_root, [m])
    sched = _make_scheduler(tmp_path, queue_root)
    _tick_until(sched, queue_root, fn)

    results = wait_result(queue_root, fn, timeout_s=5)
    assert len(results) == 1
    r = results[0]
    assert r.status == "done"
    assert r.result and r.result["outputs"] == ["out/echo1.txt"]


def test_unknown_job_type_fails_whole_batch(tmp_path):
    queue_root = str(tmp_path / "Echo2")
    m = JobManifest.from_dict({"job_id": "bad1", "job_type": "nope.nope"})
    fn = submit(queue_root, [m])
    sched = _make_scheduler(tmp_path, queue_root)
    # intake should fail the batch immediately (unknown job_type)
    sched.tick()
    results = wait_result(queue_root, fn, timeout_s=5)
    assert results[0].status == "failed"
    assert "nope.nope" in results[0].message or "batch failed" in results[0].message


def test_multiple_jobs_one_batch(tmp_path):
    queue_root = str(tmp_path / "Echo3")
    mans = [
        JobManifest.from_dict(
            {"job_id": f"e{i}", "job_type": "echo.echo", "outputs": [f"out/e{i}.txt"]}
        )
        for i in range(3)
    ]
    fn = submit(queue_root, mans)
    sched = _make_scheduler(tmp_path, queue_root)
    _tick_until(sched, queue_root, fn)
    results = wait_result(queue_root, fn, timeout_s=5)
    assert len(results) == 3
    assert all(r.status == "done" for r in results)
