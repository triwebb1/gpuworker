"""Infrastructure-only retry and quarantine (SPEC 10), plus the timeout plumbing that
decides which bound a job runs under.

The rule under test: the broker re-runs a job only when its infrastructure failed (launch
error, OOM-kill, serve load/death, reclaim-gate timeout), at most ``max_retries`` times,
and quarantines a job that fails that way on every attempt. A job that failed on its own
(bad exit, failed result, timeout) is never re-run.
"""

import json
import os
import time

import pytest

from gpuworker import gpu, serve
from gpuworker.client import submit, wait_result
from gpuworker.config import Config
from gpuworker.jobs import Completion, PendingJob
from gpuworker.launcher import LocalLauncher
from gpuworker.manifest import JobManifest
from gpuworker.obs import JobLogger
from gpuworker.scheduler import Scheduler

from helpers import FakeServe, cpu_jt, fake_connect, make_config, make_scheduler, registry, serve_jt


def _sched(tmp_path, reg, launcher=None, max_retries=1, **cfg):
    return make_scheduler(tmp_path, reg, launcher, max_retries=max_retries, **cfg)


class FlakyLocal(LocalLauncher):
    """Host-process launcher whose first *fail_first* launches raise, like a container
    runtime that is briefly unavailable."""

    def __init__(self, fail_first):
        self.fail_first, self.runs = fail_first, 0

    def run(self, spec):
        self.runs += 1
        if self.runs <= self.fail_first:
            raise OSError("container runtime unavailable")
        return super().run(spec)


class CountingLocal(LocalLauncher):
    def __init__(self):
        self.runs = 0

    def run(self, spec):
        self.runs += 1
        return super().run(spec)


def _run_one(tmp_path, sched, jt, job_id="j1"):
    qr = str(tmp_path / "Q")
    fn = submit(qr, [JobManifest.from_dict({"job_id": job_id, "job_type": jt.id})])
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        sched.tick()
        if os.path.exists(os.path.join(qr, "outbox", fn)):
            break
        time.sleep(0.01)
    return wait_result(qr, fn, timeout_s=5)[0]


def test_launch_error_is_retried_then_succeeds(tmp_path):
    jt = cpu_jt()
    sched = _sched(tmp_path, registry(jt))
    sched._local = FlakyLocal(fail_first=1)
    res = _run_one(tmp_path, sched, jt)
    assert res.status == "done"
    assert sched._local.runs == 2


def test_persistent_launch_error_is_quarantined(tmp_path):
    jt = cpu_jt()
    sched = _sched(tmp_path, registry(jt), max_retries=1)
    sched._local = FlakyLocal(fail_first=99)
    res = _run_one(tmp_path, sched, jt)
    assert res.status == "failed"
    assert "launch error" in res.message
    assert "quarantined" in res.message
    assert sched._local.runs == 2  # the first attempt + exactly one retry


def test_max_retries_zero_disables_retry(tmp_path):
    jt = cpu_jt()
    sched = _sched(tmp_path, registry(jt), max_retries=0)
    sched._local = FlakyLocal(fail_first=1)
    res = _run_one(tmp_path, sched, jt)
    assert res.status == "failed"
    assert "quarantined" not in res.message  # nothing was retried, so nothing quarantined
    assert sched._local.runs == 1


def test_oom_kill_exit_is_retried(tmp_path):
    jt = cpu_jt(code="import sys; sys.exit(137)")
    sched = _sched(tmp_path, registry(jt))
    sched._local = CountingLocal()
    res = _run_one(tmp_path, sched, jt)
    assert res.status == "failed"
    assert "OOM-kill" in res.message and "quarantined" in res.message
    assert sched._local.runs == 2


def test_worker_failure_is_never_retried(tmp_path):
    # A non-zero exit is the job failing, not the broker: re-running it would only repeat
    # the failure (or worse, silently produce a second, different output).
    jt = cpu_jt(code="import sys; sys.exit(3)")
    sched = _sched(tmp_path, registry(jt), max_retries=3)
    sched._local = CountingLocal()
    res = _run_one(tmp_path, sched, jt)
    assert res.status == "failed"
    assert "worker exited 3" in res.message
    assert sched._local.runs == 1


def test_retry_events_are_logged(tmp_path):
    jt = cpu_jt()
    log_path = tmp_path / "events.jsonl"
    cfg = make_config(
        tmp_path,
        queue_roots=[str(tmp_path / "Q")],
        registry_dir="modules",
        container_runtime="local",
        weights_dir=str(tmp_path / "w"),
        max_retries=1,
    )
    logger = JobLogger.to_path(str(log_path))
    sched = Scheduler(cfg, registry(jt), LocalLauncher(), gpu.MockGpu(), logger=logger)
    sched._local = FlakyLocal(fail_first=99)
    _run_one(tmp_path, sched, jt)
    logger.close()
    events = [
        json.loads(line)["event"] for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert events.count("retry") == 1
    assert "quarantined" in events


def test_serve_timeout_is_not_retried(tmp_path, monkeypatch):
    # A serve job that overran tears its container down like a crash would, but it is the
    # job that misbehaved, so it must fail once and stay failed.
    launcher = FakeServe(hang_on={"slow"})
    monkeypatch.setattr(serve, "connect", fake_connect(launcher))
    jt = serve_jt(job_timeout=0.3)
    cfg = make_config(
        tmp_path,
        queue_roots=[str(tmp_path / "Q")],
        registry_dir="modules",
        container_runtime="docker",
        weights_dir=str(tmp_path / "w"),
        sandbox_image="gpuworker/sandbox:cuda12.8",
        allow_unpinned_images=True,
        max_retries=2,
    )
    sched = Scheduler(cfg, registry(jt), launcher, gpu.MockGpu())
    res = _run_one(tmp_path, sched, jt, job_id="slow")
    assert res.status == "failed"
    assert "timed out" in res.message
    assert "quarantined" not in res.message
    assert launcher.runs == 1


def test_serve_crash_backlog_is_retried_on_a_fresh_container(tmp_path, monkeypatch):
    # The job that crashed the container is retried and crashes again (quarantined); a
    # job queued behind it never started, so its retry runs on the fresh container.
    launcher = FakeServe(crash_on={"boom"})
    monkeypatch.setattr(serve, "connect", fake_connect(launcher))
    jt = serve_jt()
    cfg = make_config(
        tmp_path,
        queue_roots=[str(tmp_path / "Q")],
        registry_dir="modules",
        container_runtime="docker",
        weights_dir=str(tmp_path / "w"),
        sandbox_image="gpuworker/sandbox:cuda12.8",
        allow_unpinned_images=True,
    )
    sched = Scheduler(cfg, registry(jt), launcher, gpu.MockGpu())
    qr = str(tmp_path / "Q")
    fn = submit(
        qr,
        [
            JobManifest.from_dict(
                {
                    "job_id": j,
                    "job_type": jt.id,
                    "outputs": [f"out/{j}.txt"],
                    "options": {"text": j},
                }
            )
            for j in ("boom", "after")
        ],
    )
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and not os.path.exists(os.path.join(qr, "outbox", fn)):
        sched.tick()
        time.sleep(0.01)
    res = {m.job_id: m for m in wait_result(qr, fn, timeout_s=5)}
    assert res["boom"].status == "failed" and "quarantined" in res["boom"].message
    assert res["after"].status == "done", res["after"].message


# ---- timeouts ---------------------------------------------------------------------


def test_a_dying_serve_container_takes_no_new_work(tmp_path, monkeypatch):
    # Once a manager reports an infrastructure failure it is tearing down. A retried job
    # must not be handed straight back to it (burning its retry on a container that is
    # already gone); it waits for serve_gone and then gets a fresh container.
    launcher = FakeServe()
    monkeypatch.setattr(serve, "connect", fake_connect(launcher))
    jt = serve_jt()
    cfg = Config(
        queue_roots=[str(tmp_path / "Q")],
        registry_dir="modules",
        container_runtime="docker",
        weights_dir=str(tmp_path / "w"),
        footprints_path=str(tmp_path / "fp.json"),
        state_dir=str(tmp_path / "state"),
        sandbox_image="gpuworker/sandbox:cuda12.8",
        allow_unpinned_images=True,
    )
    sched = Scheduler(cfg, registry(jt), launcher, gpu.MockGpu())
    qr = str(tmp_path / "Q")
    fn = submit(
        qr,
        [
            JobManifest.from_dict(
                {
                    "job_id": "w",
                    "job_type": jt.id,
                    "outputs": ["out/w.txt"],
                    "options": {"text": "w"},
                }
            )
        ],
    )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not os.path.exists(os.path.join(qr, "outbox", fn)):
        sched.tick()
        time.sleep(0.01)
    sc = sched._serve[jt.id]
    assert sched._warm_serve(jt) is sc
    # Report an infrastructure failure from the live container, as its manager would.
    probe = PendingJob(
        manifest=JobManifest.from_dict({"job_id": "p", "job_type": jt.id}), jt=jt, batch=None
    )
    sched.ledger.start_serve_job(jt.id, sched.clock())  # the failed job's running slot
    sched._on_complete(
        Completion(
            probe, "failed", "serve container error: boom", [], serve_epoch=sc.epoch, infra=True
        )
    )
    assert sc.dying
    assert sched._warm_serve(jt) is None
    assert not sched._admittable(
        PendingJob(
            manifest=JobManifest.from_dict({"job_id": "q", "job_type": jt.id}), jt=jt, batch=None
        )
    )
    sched.shutdown()


def test_shutdown_fails_a_retryable_job_instead_of_requeueing_it(tmp_path):
    # During shutdown nothing will run a re-queued job, so an infrastructure failure
    # drained then must end the job (with a reason), not leave it queued forever.
    jt = cpu_jt()
    sched = _sched(tmp_path, registry(jt), max_retries=3)
    qr = str(tmp_path / "Q")
    fn = submit(qr, [JobManifest.from_dict({"job_id": "s1", "job_type": jt.id})])
    sched.intake()
    [pj] = sched.pending
    sched.pending.remove(pj)  # it was dispatched: a reporting job is no longer pending
    sched.ledger.admit_cpu()  # the slot the failed attempt held
    sched._draining = True
    sched._on_complete(Completion(pj, "failed", "launch error: runtime gone", [], infra=True))
    assert pj not in sched.pending
    res = wait_result(qr, fn, timeout_s=5)[0]
    assert res.status == "failed"
    assert "shutting down" in res.message


def test_serve_config_error_is_final_not_retried(tmp_path, monkeypatch):
    # An unsealed required weight cache fails the same way on every attempt: it is a
    # configuration error, not an infrastructure failure, for serve exactly as for oneshot.
    launcher = FakeServe()
    monkeypatch.setattr(serve, "connect", fake_connect(launcher))
    jt = serve_jt()
    jt.weights_required = True
    sched = make_scheduler(
        tmp_path, registry(jt), launcher, container_runtime="docker", max_retries=2
    )
    res = _run_one(tmp_path, sched, jt, job_id="cfg")
    assert res.status == "failed"
    assert "serve config error" in res.message and "weight cache" in res.message
    assert "quarantined" not in res.message
    assert launcher.runs == 0  # never launched


def test_a_container_torn_down_after_a_timeout_takes_no_new_work(tmp_path, monkeypatch):
    launcher = FakeServe()
    monkeypatch.setattr(serve, "connect", fake_connect(launcher))
    jt = serve_jt()
    sched = make_scheduler(tmp_path, registry(jt), launcher, container_runtime="docker")
    _run_one(tmp_path, sched, jt, job_id="warm")
    sc = sched._serve[jt.id]
    from gpuworker.jobs import Batch
    from gpuworker.share import QueueTree

    (tmp_path / "wd").mkdir()
    batch = Batch(
        queue=QueueTree(str(tmp_path / "Q")),
        filename="t.zip",
        work_dir=str(tmp_path / "wd"),
        out_dir=str(tmp_path / "wd"),
    )
    probe = PendingJob(
        manifest=JobManifest.from_dict({"job_id": "t", "job_type": jt.id}), jt=jt, batch=batch
    )
    batch.jobs.append(probe)
    sched.ledger.start_serve_job(jt.id, sched.clock())
    # A timeout is the job's own failure (not retried) but the container is still gone.
    sched._on_complete(
        Completion(
            probe,
            "failed",
            "serve job t timed out after 1s",
            [],
            serve_epoch=sc.epoch,
            container_gone=True,
        )
    )
    assert sc.dying and sched._warm_serve(jt) is None
    sched.shutdown()


@pytest.mark.parametrize("env,infra", [("container", True), ("host", False)])
def test_exit_125_is_a_runtime_error_only_for_container_jobs(tmp_path, env, infra):
    # Docker/Podman exit 125 when the runtime could not create the container; a host
    # process exiting 125 is just the job's own exit code.
    import queue as _q

    from gpuworker.jobs import Batch, WorkerContext
    from gpuworker.oneshot import run_oneshot
    from gpuworker.share import QueueTree

    jt = cpu_jt(code="import sys; sys.exit(125)")
    jt.env = env
    cfg = make_config(tmp_path)
    ctx = WorkerContext(
        cfg=cfg,
        launcher=LocalLauncher(),
        monitor=gpu.MockGpu(),
        completions=_q.Queue(),
        log=JobLogger.null(),
        weights_for=lambda jt: None,
        write_worker_log=lambda *a: None,
    )
    (tmp_path / "wd").mkdir()
    batch = Batch(
        queue=QueueTree(str(tmp_path / "Q")),
        filename="b.zip",
        work_dir=str(tmp_path / "wd"),
        out_dir=str(tmp_path / "wd"),
    )
    pj = PendingJob(
        manifest=JobManifest.from_dict({"job_id": "e", "job_type": jt.id}), jt=jt, batch=batch
    )
    run_oneshot(pj, ctx, sample_vram=False)
    c = ctx.completions.get_nowait()
    assert c.status == "failed" and c.infra is infra


def test_a_oneshot_container_is_named_from_its_own_attempt(tmp_path):
    """The name came from pj.attempts, read after the weight check, and cleanup removes by
    that name: had the job moved on to a retry meanwhile, the attempt would have taken the
    retry's name."""
    import queue as _q

    from gpuworker.jobs import Attempt, Batch, WorkerContext
    from gpuworker.oneshot import run_oneshot
    from gpuworker.share import QueueTree

    class Done:
        returncode = 0
        stdout = stderr = None

        def wait(self, timeout=None):
            return 0

    class Recording(LocalLauncher):
        def __init__(self):
            self.names, self.cleaned = [], []

        def run(self, spec):
            self.names.append(spec.name)
            return Done()

        def cleanup(self, spec):
            self.cleaned.append(spec.name)

    jt = cpu_jt()
    jt.env = "container"
    launcher = Recording()
    ctx = WorkerContext(
        cfg=make_config(tmp_path),
        launcher=launcher,
        monitor=gpu.MockGpu(),
        completions=_q.Queue(),
        log=JobLogger.null(),
        weights_for=lambda jt: None,
        write_worker_log=lambda *a: None,
        job_name_prefix="gpuworker-job-b0b0-",
    )
    (tmp_path / "wd").mkdir()
    batch = Batch(
        queue=QueueTree(str(tmp_path / "Q")),
        filename="b.zip",
        work_dir=str(tmp_path / "wd"),
        out_dir=str(tmp_path / "wd"),
    )
    pj = PendingJob(
        manifest=JobManifest.from_dict({"job_id": "e", "job_type": jt.id}), jt=jt, batch=batch
    )
    attempt = Attempt(pj.attempts)
    pj.attempts += 1  # a retry was scheduled before this attempt launched
    run_oneshot(pj, ctx, sample_vram=False, attempt=attempt)
    assert launcher.names == [f"gpuworker-job-b0b0-{pj.uid}-{attempt.number}"]
    assert launcher.cleaned == launcher.names
