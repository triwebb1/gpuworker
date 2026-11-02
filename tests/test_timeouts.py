"""Which timeout bound a job runs under, and how overruns are reported."""

import json
import os
import time

import pytest

from gpuworker.client import submit
from gpuworker.manifest import JobManifest
from gpuworker.oneshot import STREAM_HEAD_BYTES, STREAM_TAIL_BYTES, await_proc
from gpuworker.scheduler import ONESHOT_WATCHDOG_SLACK_S

from helpers import cpu_jt, inflight, make_scheduler, registry, serve_jt


def test_oneshot_watchdog_uses_the_type_timeout(tmp_path):
    # The watchdog must arm from the type's timeout, not just manifest.timeout_s or
    # default_timeout_s, or a type allowed longer than the default is killed early.
    jt = cpu_jt(code="import time; time.sleep(2)", oneshot_job_timeout_s=5000)
    sched = make_scheduler(tmp_path, registry(jt), default_timeout_s=1800)
    clk = [1000.0]
    sched.clock = lambda: clk[0]
    submit(str(tmp_path / "Q"), [JobManifest.from_dict({"job_id": "long", "job_type": jt.id})])
    for _ in range(50):
        sched.tick()
        if inflight(sched, "long"):
            break
        time.sleep(0.01)
    _pj, deadline = inflight(sched, "long")
    assert deadline == pytest.approx(1000.0 + 5000 + ONESHOT_WATCHDOG_SLACK_S)


def test_broker_limits_report_the_mode_each_type_runs_in(tmp_path):
    oneshot = cpu_jt(id="ex.oneshot")  # inherits the default
    long_oneshot = cpu_jt(id="ex.long", oneshot_job_timeout_s=900)
    served = serve_jt(id="ex.served", job_timeout=120)  # vram + serve
    sched = make_scheduler(
        tmp_path, registry(oneshot, long_oneshot, served), default_timeout_s=1800
    )
    limits_path = os.path.join(str(tmp_path / "Q"), "broker_limits.json")
    # Constructing a Scheduler (as `gpuworker status` does) must not write into client
    # queue trees; run() publishes the limits.
    assert not os.path.exists(limits_path)
    sched._publish_limits()
    with open(limits_path, encoding="utf-8") as fh:
        limits = json.load(fh)["job_type_timeouts_s"]
    assert limits == {"ex.oneshot": 1800, "ex.long": 900, "ex.served": 120}


def test_vram_sampling_drains_pipes_so_a_verbose_worker_finishes(tmp_path):
    # While sampling NVML the supervisor used to poll() for exit without reading the pipes,
    # so a worker that wrote more than a pipe buffer blocked forever and timed out.
    import subprocess
    import sys

    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.write('x' * 2_000_000); "
            "sys.stderr.write('y' * 300_000); import time; time.sleep(0.6)",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    sched = make_scheduler(tmp_path, registry(cpu_jt()))
    peak = {"mb": None}
    started = time.monotonic()
    out, err = await_proc(proc, None, 20, sched._worker_ctx(cpu_jt()), None, peak, {})
    assert time.monotonic() - started < 15
    assert err == b"y" * 300_000  # under the cap: kept whole
    assert peak["mb"] is not None  # it was sampling throughout
    # Over the cap: the head and the tail survive, with a note of what was dropped.
    assert out.startswith(b"x" * STREAM_HEAD_BYTES) and out.endswith(b"x" * STREAM_TAIL_BYTES)
    omitted = 2_000_000 - STREAM_HEAD_BYTES - STREAM_TAIL_BYTES
    assert f"[gpuworker: {omitted} bytes of output omitted]".encode() in out


def test_captured_output_is_bounded_however_much_a_worker_writes(tmp_path):
    import subprocess
    import sys

    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys; [sys.stdout.write('z' * 100_000) for _ in range(200)]; "
            "sys.stdout.write('THE END')",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    sched = make_scheduler(tmp_path, registry(cpu_jt()))
    streams: dict[str, bytes] = {}
    out, _ = await_proc(proc, None, 30, sched._worker_ctx(cpu_jt()), None, None, streams)
    assert len(out) < STREAM_HEAD_BYTES + STREAM_TAIL_BYTES + 100
    assert out.endswith(b"THE END") and streams["stdout"] == out


def test_non_timeout_error_while_waiting_is_not_reported_as_a_timeout(tmp_path):
    class BrokenProc:
        stdout = stderr = None

        def wait(self, timeout=None):
            raise OSError("pipe broke")

    class NoopLauncher:
        terminated = 0

        def terminate(self, proc, spec):
            NoopLauncher.terminated += 1

    sched = make_scheduler(tmp_path, registry(cpu_jt()))
    ctx = sched._worker_ctx(cpu_jt())
    with pytest.raises(OSError, match="pipe broke"):
        await_proc(BrokenProc(), None, 5, ctx, NoopLauncher(), None, {})
    assert NoopLauncher.terminated == 1


def test_a_manifest_cannot_raise_its_timeout_past_the_type_bound(tmp_path):
    """timeout_s up to 10,000,000 s used to replace the admin's bound outright."""
    from gpuworker.oneshot import oneshot_timeout
    from gpuworker.serve_manager import serve_timeout

    jt = cpu_jt(oneshot_job_timeout_s=600)
    sjt = serve_jt(job_timeout=120)
    sched = make_scheduler(tmp_path, registry(jt), default_timeout_s=1800)
    long = JobManifest.from_dict({"job_id": "a", "job_type": jt.id, "timeout_s": 10_000_000})
    short = JobManifest.from_dict({"job_id": "b", "job_type": jt.id, "timeout_s": 60})
    assert oneshot_timeout(long, jt, sched.cfg) == 600
    assert oneshot_timeout(short, jt, sched.cfg) == 60
    assert serve_timeout(long, sjt) == 120
    assert serve_timeout(short, sjt) == 60


def test_a_capped_timeout_is_logged_at_intake(tmp_path):
    from gpuworker.obs import JobLogger

    log = tmp_path / "events.jsonl"
    jt = cpu_jt(oneshot_job_timeout_s=600)
    sched = make_scheduler(tmp_path, registry(jt))
    sched.log = JobLogger.to_path(str(log))
    submit(
        str(tmp_path / "Q"),
        [JobManifest.from_dict({"job_id": "a", "job_type": jt.id, "timeout_s": 99999})],
    )
    sched.intake()
    sched.log.close()
    events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    (capped,) = [e for e in events if e["event"] == "timeout_capped"]
    assert capped["requested"] == 99999 and capped["cap"] == 600
