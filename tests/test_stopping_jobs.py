"""Stopping a running oneshot job from outside it: the watchdog and a broker shutdown end
its process, and only then report the job and free its ledger slot."""

import sys
import threading
import time

from gpuworker import gpu, oneshot
from gpuworker.client import collect, submit, wait_result
from gpuworker.launcher import LocalLauncher
from gpuworker.manifest import JobManifest
from gpuworker.module import JobType
from gpuworker.scheduler import Scheduler

from helpers import cpu_jt, drive, make_config, registry

SLEEPER = "import time; time.sleep(60)"


class _Recording(LocalLauncher):
    """A host launcher that keeps every process it starts. While *gate* is clear, a
    terminate blocks before it kills, as a slow `podman rm -f` would; with *ignore* it
    never kills at all."""

    def __init__(self, ignore=False):
        self.procs = []
        self.gate = threading.Event()
        self.gate.set()
        self.ignore = ignore

    def run(self, spec):
        proc = super().run(spec)
        self.procs.append(proc)
        return proc

    def terminate(self, proc, spec):
        self.gate.wait(10)
        if not self.ignore:
            super().terminate(proc, spec)

    def kill_all(self):
        for p in self.procs:
            p.kill()
            p.wait(10)


def _sched(tmp_path, jt, launcher, monitor=None, **kw):
    sched = Scheduler(make_config(tmp_path), registry(jt), launcher, monitor or gpu.MockGpu(), **kw)
    sched._local = launcher  # host job types run through it
    return sched


def _started(sched, launcher, n=1):
    for _ in range(400):
        sched.tick()
        if len(launcher.procs) >= n:
            return
        time.sleep(0.005)
    raise AssertionError("the worker never started")


def _job(jt, job_id="hung"):
    return JobManifest.from_dict({"job_id": job_id, "job_type": jt.id})


def test_the_watchdog_stops_a_hung_oneshot_before_freeing_its_slot(tmp_path):
    """The watchdog only reported the job failed. The slot was released while the worker
    kept running, so the next model could be admitted against memory still in use."""
    launcher = _Recording()
    jt = cpu_jt(code=SLEEPER)
    now = [1000.0]
    sched = _sched(tmp_path, jt, launcher, clock=lambda: now[0])
    qr = str(tmp_path / "Q")
    fn = submit(qr, [_job(jt)])
    _started(sched, launcher)
    (proc,) = launcher.procs
    assert sched.ledger.cpu_active == 1
    try:
        launcher.gate.clear()  # the stop is slow
        now[0] += 1e6  # past the watchdog deadline
        for _ in range(20):
            sched.tick()
            time.sleep(0.01)
        assert proc.poll() is None
        assert sched.ledger.cpu_active == 1  # held while the worker still runs
        launcher.gate.set()
        assert drive(sched, qr, fn)
        assert proc.poll() is not None  # the worker was stopped
        assert sched.ledger.cpu_active == 0
        (res,) = wait_result(qr, fn, timeout_s=5)
        assert res.status == "failed" and "watchdog" in res.message
        assert "worker stopped" in res.message
    finally:
        launcher.gate.set()
        launcher.kill_all()


def test_a_worker_that_will_not_stop_keeps_vram_admission_held(tmp_path, monkeypatch):
    """The job still gets its result, but its memory may still be in use, so the GPU is
    held degraded until the card shows it free."""
    monkeypatch.setattr(oneshot, "STOP_WAIT_S", 0.3)
    launcher = _Recording(ignore=True)
    jt = JobType(
        id="ex.vram",
        resource_class="vram",
        vram_mb=4000,
        env="host",
        entrypoint=[sys.executable, "-c", SLEEPER],
        sandbox="none",
    )
    now = [1000.0]
    g = gpu.MockGpu(used_mb=800)
    sched = _sched(tmp_path, jt, launcher, g, clock=lambda: now[0])
    qr = str(tmp_path / "Q")
    fn = submit(qr, [_job(jt)])
    try:
        _started(sched, launcher)
        g.used = 800 + 4000  # what the worker holds
        now[0] += 1e6
        assert drive(sched, qr, fn)
        (res,) = wait_result(qr, fn, timeout_s=5)
        assert res.status == "failed" and "did not exit when stopped" in res.message
        assert not sched.ledger.residents  # the job is over
        assert sched._gpu_degraded
        sched.tick()
        assert sched._gpu_degraded  # its memory is still in use
        g.used = 800
        sched.tick()
        assert not sched._gpu_degraded
    finally:
        launcher.kill_all()


def test_a_worker_that_outlives_its_timeout_keeps_vram_admission_held(tmp_path, monkeypatch):
    """At its own timeout the supervisor terminated the worker, waited only for its output
    and reported the job, so the slot was freed whether or not the worker had exited. Both
    terminate implementations swallow their errors."""
    monkeypatch.setattr(oneshot, "STOP_WAIT_S", 0.3)
    launcher = _Recording(ignore=True)
    jt = JobType(
        id="ex.vram",
        resource_class="vram",
        vram_mb=4000,
        env="host",
        entrypoint=[sys.executable, "-c", SLEEPER],
        sandbox="none",
        oneshot_job_timeout_s=1,
    )
    g = gpu.MockGpu(used_mb=800)
    sched = _sched(tmp_path, jt, launcher, g)
    qr = str(tmp_path / "Q")
    fn = submit(qr, [_job(jt)])
    try:
        _started(sched, launcher)
        g.used = 800 + 4000
        assert drive(sched, qr, fn, ticks=2000)
        (res,) = wait_result(qr, fn, timeout_s=5)
        assert res.status == "failed" and "exceeded timeout" in res.message
        assert "did not exit when stopped" in res.message
        assert sched._gpu_degraded
        g.used = 800
        sched.tick()
        assert not sched._gpu_degraded
    finally:
        launcher.kill_all()


class _Exited:
    def __init__(self, hangs=False):
        self.hangs = hangs

    def wait(self, timeout=None):
        if self.hangs:
            raise oneshot.subprocess.TimeoutExpired("worker", timeout)
        return 0


class _Runtime:
    def __init__(self, names):
        self.names = set(names)

    def container_exists(self, name):
        return name in self.names


def test_a_stopped_worker_counts_as_gone_only_once_its_container_is_gone():
    """`podman rm -f` failing quietly left the container, and its memory, behind a client
    that had exited."""
    from gpuworker.launcher import LaunchSpec

    spec = LaunchSpec(job_type=cpu_jt(), bundle_dir="", out_dir="", name="gpuworker-job-1-0")
    assert not oneshot.worker_gone(_Exited(), spec, _Runtime({spec.name}), 0.1)
    assert oneshot.worker_gone(_Exited(), spec, _Runtime(set()), 0.1)
    assert not oneshot.worker_gone(_Exited(hangs=True), spec, _Runtime(set()), 0.1)
    host = LaunchSpec(job_type=cpu_jt(), bundle_dir="", out_dir="")
    assert oneshot.worker_gone(_Exited(), host, LocalLauncher(), 0.1)


def test_shutdown_stops_a_running_oneshot_and_reports_it(tmp_path):
    """Oneshot supervisors are daemon threads: a stop left the job without a result and
    its worker running after the broker had exited."""
    launcher = _Recording()
    jt = cpu_jt(code=SLEEPER)
    sched = _sched(tmp_path, jt, launcher)
    qr = str(tmp_path / "Q")
    fn = submit(qr, [_job(jt)])
    try:
        _started(sched, launcher)
        (proc,) = launcher.procs
        sched.shutdown(timeout_s=0.2)
        assert proc.poll() is not None
        (res,) = collect(qr, fn, str(tmp_path / "got"), timeout_s=5)
        assert res.status == "failed" and "broker shut down" in res.message
        assert not sched._attempts
        log = (tmp_path / "got" / "_gpuworker" / "hung.log").read_text(encoding="utf-8")
        assert "broker shut down" in log  # the worker log ships with a stopped job too
    finally:
        launcher.kill_all()


def test_shutdown_waits_for_a_oneshot_that_finishes_in_time(tmp_path):
    launcher = _Recording()
    jt = cpu_jt(code="import time; time.sleep(0.5)")
    sched = _sched(tmp_path, jt, launcher)
    qr = str(tmp_path / "Q")
    fn = submit(qr, [_job(jt, "quick")])
    _started(sched, launcher)
    sched.shutdown(timeout_s=10.0)
    (res,) = wait_result(qr, fn, timeout_s=5)
    assert res.status == "done", res.message


def test_a_stop_before_launch_starts_nothing(tmp_path):
    """A watchdog that fires while the supervisor is still preparing (a slow weight check)
    must not leave it to start the worker afterwards."""
    from gpuworker.jobs import Attempt

    attempt = Attempt(0)
    assert attempt.stop() == (None, None)
    assert attempt.launch(lambda: (_ for _ in ()).throw(AssertionError("started")), None) is None
    assert not attempt.claim_report()
