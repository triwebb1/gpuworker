"""Serve-mode lifecycle: one warm container per resident serve job type, jobs streamed
over the socket, kept warm + evicted via the ledger. Containerless: a fake runtime pairs
a socketpair per `run()`, runs an echo serve worker on one end, and hands the other to the
broker via a monkeypatched serve.connect (socketpair works on Windows + Linux)."""

import os
import threading
import time

import pytest

from gpuworker import gpu, serve
from gpuworker.client import submit, wait_result
from gpuworker.manifest import JobManifest
from gpuworker.scheduler import Scheduler

from helpers import (
    FakeServe,
    drive,
    fake_connect,
    inflight,
    make_config,
    parked_ids,
    registry,
    serve_jt,
)


def _sched(tmp_path, reg, launcher, monitor=None):
    cfg = make_config(
        tmp_path,
        queue_roots=[str(tmp_path / "Q")],
        registry_dir="modules",
        container_runtime="docker",
        weights_dir=str(tmp_path / "w"),
        sandbox_image="gpuworker/sandbox:cuda12.8",
        allow_unpinned_images=True,
    )
    cfg.serve_connect_timeout_s = 5.0
    cfg.serve_ready_timeout_s = 5.0
    return Scheduler(cfg, reg, launcher, monitor or gpu.MockGpu())


# ---- tests ------------------------------------------------------------------


def test_serve_cold_load_then_run(tmp_path, monkeypatch):
    launcher = FakeServe()
    monkeypatch.setattr(serve, "connect", fake_connect(launcher))
    jt = serve_jt()
    sched = _sched(tmp_path, registry(jt), launcher)
    qr = str(tmp_path / "Q")
    m = JobManifest.from_dict(
        {"job_id": "s1", "job_type": jt.id, "outputs": ["out/o.txt"], "options": {"text": "HELLO"}}
    )
    fn = submit(qr, [m])
    assert drive(sched, qr, fn)
    res = wait_result(qr, fn, timeout_s=5)
    assert res[0].status == "done"
    assert launcher.runs == 1
    assert sched.ledger.is_resident(jt.id)  # stays warm after the job


def test_serve_streams_many_jobs_to_one_container(tmp_path, monkeypatch):
    launcher = FakeServe()
    monkeypatch.setattr(serve, "connect", fake_connect(launcher))
    jt = serve_jt()
    sched = _sched(tmp_path, registry(jt), launcher)
    qr = str(tmp_path / "Q")
    fns = []
    for i in range(4):
        m = JobManifest.from_dict(
            {
                "job_id": f"j{i}",
                "job_type": jt.id,
                "outputs": [f"out/{i}.txt"],
                "options": {"text": f"v{i}"},
            }
        )
        fns.append(submit(qr, [m]))
    for fn in fns:
        assert drive(sched, qr, fn)
        assert wait_result(qr, fn, timeout_s=5)[0].status == "done"
    assert launcher.runs == 1  # one container served all four
    assert sched.ledger.resident_vram_mb <= sched.ledger.admit_budget_mb


def test_serve_multi_job_batch(tmp_path, monkeypatch):
    # One bundle with two serve jobs: both stream to one warm container, both outputs are
    # collected, and the batch's work/out dirs aren't rmtree'd until both finish.
    import zipfile

    launcher = FakeServe()
    monkeypatch.setattr(serve, "connect", fake_connect(launcher))
    jt = serve_jt()
    sched = _sched(tmp_path, registry(jt), launcher)
    qr = str(tmp_path / "Q")
    m1 = JobManifest.from_dict(
        {"job_id": "x", "job_type": jt.id, "outputs": ["out/x.txt"], "options": {"text": "X"}}
    )
    m2 = JobManifest.from_dict(
        {"job_id": "y", "job_type": jt.id, "outputs": ["out/y.txt"], "options": {"text": "Y"}}
    )
    fn = submit(qr, [m1, m2])  # one bundle, two jobs
    assert drive(sched, qr, fn)
    res = wait_result(qr, fn, timeout_s=5)
    assert {r.job_id: r.status for r in res} == {"x": "done", "y": "done"}
    assert launcher.runs == 1
    with zipfile.ZipFile(os.path.join(qr, "outbox", fn)) as zf:
        names = zf.namelist()
    assert "out/x.txt" in names and "out/y.txt" in names


def test_serve_idle_evicts_after_keep_warm(tmp_path, monkeypatch):
    launcher = FakeServe()
    monkeypatch.setattr(serve, "connect", fake_connect(launcher))
    jt = serve_jt(keep_warm=0.0)  # evict as soon as idle
    sched = _sched(tmp_path, registry(jt), launcher)
    qr = str(tmp_path / "Q")
    fn = submit(
        qr, [JobManifest.from_dict({"job_id": "s1", "job_type": jt.id, "outputs": ["out/o.txt"]})]
    )
    assert drive(sched, qr, fn)
    assert wait_result(qr, fn, timeout_s=5)[0].status == "done"
    # idle -> evict_expired tears the warm container down (async); wait for the manager
    # to actually terminate it (ledger frees on the loop; teardown on the manager thread).
    for _ in range(200):
        sched.tick()
        if jt.id not in sched._serve and launcher.terminated >= 1:
            break
        time.sleep(0.005)
    assert not sched.ledger.is_resident(jt.id)
    assert jt.id not in sched._serve
    assert launcher.terminated >= 1


def test_serve_evicts_first_type_for_second(tmp_path, monkeypatch):
    launcher = FakeServe()
    monkeypatch.setattr(serve, "connect", fake_connect(launcher))
    a = serve_jt(id="ex.a", vram=10000, keep_warm=100.0)
    b = serve_jt(id="ex.b", vram=10000, keep_warm=100.0)  # both can't co-reside (20G > budget)
    g = gpu.MockGpu(total_mb=16000, used_mb=200)
    sched = _sched(tmp_path, registry(a, b), launcher, monitor=g)
    qr = str(tmp_path / "Q")
    fa = submit(
        qr, [JobManifest.from_dict({"job_id": "a1", "job_type": "ex.a", "outputs": ["out/a.txt"]})]
    )
    assert drive(sched, qr, fa)
    assert wait_result(qr, fa, timeout_s=5)[0].status == "done"
    assert sched.ledger.is_resident("ex.a")
    # b arrives: a is idle -> evicted; its container frees VRAM -> b loads. Mock frees on
    # eviction (ledger), and MockGpu.used tracks via the test setting it after teardown.
    g.used = 200  # a's container "freed" the card
    fb = submit(
        qr, [JobManifest.from_dict({"job_id": "b1", "job_type": "ex.b", "outputs": ["out/b.txt"]})]
    )
    assert drive(sched, qr, fb)
    assert wait_result(qr, fb, timeout_s=5)[0].status == "done"
    assert sched.ledger.is_resident("ex.b")
    assert not sched.ledger.is_resident("ex.a")  # evicted
    assert sched.ledger.resident_vram_mb <= sched.ledger.admit_budget_mb
    assert launcher.runs == 2  # one container each, sequentially


def test_two_cold_serve_jobs_stream_into_one_warm_no_degrade(tmp_path, monkeypatch):
    # Two jobs of a not-yet-resident serve type arrive together (the "warm a model, send a
    # batch" pattern). After the first loads the model the card fills, so the second's
    # NVML free is below its target, but it needs no new VRAM. It must stream into the
    # warm container, not time out the reclaim gate and latch _gpu_degraded.
    launcher = FakeServe()
    monkeypatch.setattr(serve, "connect", fake_connect(launcher))
    jt = serve_jt(vram=5000, keep_warm=100.0)
    g = gpu.MockGpu(total_mb=16000, used_mb=200)
    sched = _sched(tmp_path, registry(jt), launcher, monitor=g)
    sched.cfg.reclaim_timeout_s = 0.3  # without the warm bypass, job 2 would degrade fast
    qr = str(tmp_path / "Q")
    f0 = submit(
        qr,
        [
            JobManifest.from_dict(
                {
                    "job_id": "a",
                    "job_type": jt.id,
                    "outputs": ["out/a.txt"],
                    "options": {"text": "A"},
                }
            )
        ],
    )
    f1 = submit(
        qr,
        [
            JobManifest.from_dict(
                {
                    "job_id": "b",
                    "job_type": jt.id,
                    "outputs": ["out/b.txt"],
                    "options": {"text": "B"},
                }
            )
        ],
    )
    loaded = False
    out0 = os.path.join(qr, "outbox", f0)
    out1 = os.path.join(qr, "outbox", f1)
    for _ in range(400):
        sched.tick()
        if jt.id in sched._serve and not loaded:
            g.used = 12000  # model loaded: free ~4000 < target 6024
            loaded = True
        if os.path.exists(out0) and os.path.exists(out1):
            sched.tick()
            break
        time.sleep(0.005)
    assert wait_result(qr, f0, timeout_s=5)[0].status == "done"
    assert wait_result(qr, f1, timeout_s=5)[0].status == "done"  # streamed in, not failed
    assert not sched._gpu_degraded  # gate not falsely tripped
    assert launcher.runs == 1


def test_serve_respawn_periodic_max_jobs(tmp_path, monkeypatch):
    """serve_max_jobs=1: the broker respawns a fresh container before each job after the
    first (periodic recycle for a model that accumulates state). Both jobs complete; job 2
    gets a new container (runs==2, terminated>=1); the resident slot stays (no serve_gone);
    running->0."""
    launcher = FakeServe()
    monkeypatch.setattr(serve, "connect", fake_connect(launcher))
    jt = serve_jt(id="example.large", max_jobs=1)
    sched = _sched(tmp_path, registry(jt), launcher)
    qr = str(tmp_path / "Q")
    f0 = submit(
        qr,
        [
            JobManifest.from_dict(
                {
                    "job_id": "a",
                    "job_type": jt.id,
                    "outputs": ["out/a.txt"],
                    "options": {"text": "A"},
                }
            )
        ],
    )
    assert drive(sched, qr, f0)
    assert wait_result(qr, f0, timeout_s=5)[0].status == "done"
    assert launcher.runs == 1  # first job on the initial container
    f1 = submit(
        qr,
        [
            JobManifest.from_dict(
                {
                    "job_id": "b",
                    "job_type": jt.id,
                    "outputs": ["out/b.txt"],
                    "options": {"text": "B"},
                }
            )
        ],
    )
    assert drive(sched, qr, f1)
    assert wait_result(qr, f1, timeout_s=5)[0].status == "done"
    assert launcher.runs == 2  # job 2 on a fresh container (the respawn)
    assert launcher.terminated >= 1  # old container torn down before relaunch
    assert sched.ledger.is_resident(jt.id)  # same slot, no serve_gone
    assert not sched.ledger.residents[jt.id].busy  # running-count returned to 0


def test_serve_respawn_on_demand_reset_serve(tmp_path, monkeypatch):
    """options.reset_serve on a job forces a fresh-container respawn before it (a client's
    retry hook), driven per-job with no periodic cap."""
    launcher = FakeServe()
    monkeypatch.setattr(serve, "connect", fake_connect(launcher))
    jt = serve_jt()
    sched = _sched(tmp_path, registry(jt), launcher)
    qr = str(tmp_path / "Q")
    f0 = submit(
        qr,
        [
            JobManifest.from_dict(
                {
                    "job_id": "a",
                    "job_type": jt.id,
                    "outputs": ["out/a.txt"],
                    "options": {"text": "A"},
                }
            )
        ],
    )
    assert drive(sched, qr, f0)
    assert wait_result(qr, f0, timeout_s=5)[0].status == "done"
    assert launcher.runs == 1
    f1 = submit(
        qr,
        [
            JobManifest.from_dict(
                {
                    "job_id": "b",
                    "job_type": jt.id,
                    "outputs": ["out/b.txt"],
                    "options": {"text": "B", "reset_serve": 1},
                }
            )
        ],
    )
    assert drive(sched, qr, f1)
    assert wait_result(qr, f1, timeout_s=5)[0].status == "done"
    assert launcher.runs == 2  # reset_serve forced a fresh container
    assert launcher.terminated >= 1
    assert sched.ledger.is_resident(jt.id)


def test_serve_respawn_gate_fails_when_vram_not_reclaimed(tmp_path, monkeypatch):
    """If the old model's VRAM has not returned when a respawn is due, the reclaim gate
    times out: the container is torn down (not a silent OOM-hang), the GPU is marked
    degraded exactly as for the admission gate, and the job waits, re-queued, until the
    memory returns."""
    launcher = FakeServe()
    monkeypatch.setattr(serve, "connect", fake_connect(launcher))
    jt = serve_jt(id="example.large", vram=10000, max_jobs=1)
    g = gpu.MockGpu(total_mb=16000, used_mb=200)
    sched = _sched(tmp_path, registry(jt), launcher, monitor=g)
    sched.cfg.reclaim_timeout_s = 0.2
    qr = str(tmp_path / "Q")
    f0 = submit(
        qr,
        [
            JobManifest.from_dict(
                {
                    "job_id": "a",
                    "job_type": jt.id,
                    "outputs": ["out/a.txt"],
                    "options": {"text": "A"},
                }
            )
        ],
    )
    assert drive(sched, qr, f0)
    assert wait_result(qr, f0, timeout_s=5)[0].status == "done"
    g.used = 15000  # old model's VRAM did not return (free 1000 < 10000)
    f1 = submit(
        qr,
        [
            JobManifest.from_dict(
                {
                    "job_id": "b",
                    "job_type": jt.id,
                    "outputs": ["out/b.txt"],
                    "options": {"text": "B"},
                }
            )
        ],
    )
    for _ in range(400):
        sched.tick()
        if sched._gpu_degraded and not sched.ledger.is_resident(jt.id):
            break
        time.sleep(0.005)
    assert sched._gpu_degraded  # same response as an admission-gate timeout
    assert not sched.ledger.is_resident(jt.id)  # slot freed, not left pinned
    (b,) = [pj for pj in sched.pending if pj.manifest.job_id == "b"]
    assert "respawn" in b.manifest.message  # re-queued with the reason
    g.used = 200  # the memory comes back: degraded mode clears and b runs
    assert drive(sched, qr, f1)
    assert wait_result(qr, f1, timeout_s=5)[0].status == "done"
    assert not sched._gpu_degraded


def test_serve_no_recycle_without_serve_max_jobs(tmp_path, monkeypatch):
    """Periodic recycle is a per-job-type opt-in. A serve type that leaves
    serve_max_jobs at 0 must not respawn between jobs (that would defeat its warm-serve
    design and pay a needless reload per job)."""
    launcher = FakeServe()
    monkeypatch.setattr(serve, "connect", fake_connect(launcher))
    jt = serve_jt(id="example.model")  # serve_max_jobs defaults to 0
    sched = _sched(tmp_path, registry(jt), launcher)
    qr = str(tmp_path / "Q")
    for i in range(2):
        fn = submit(
            qr,
            [
                JobManifest.from_dict(
                    {
                        "job_id": f"c{i}",
                        "job_type": jt.id,
                        "outputs": [f"out/{i}.txt"],
                        "options": {"text": f"v{i}"},
                    }
                )
            ],
        )
        assert drive(sched, qr, fn)
        assert wait_result(qr, fn, timeout_s=5)[0].status == "done"
    assert launcher.runs == 1  # one warm container served both (no respawn)
    assert launcher.terminated == 0


def _hang_then(tmp_path, monkeypatch, behind=()):
    """A warm container whose job "hang" never answers, with *behind* queued after it, and
    a clock that has just jumped past the watchdog deadline."""
    launcher = FakeServe(hang_on={"hang"})
    monkeypatch.setattr(serve, "connect", fake_connect(launcher))
    # Large keep_warm + job_timeout: neither idle-evict nor the manager's own recv-timeout may
    # fire first, so the broker watchdog (driven by the injected clock) fails the job.
    jt = serve_jt(keep_warm=1e6, job_timeout=1e6)  # never self-fire; must fit C timeval
    sched = _sched(tmp_path, registry(jt), launcher)
    clk = [1000.0]
    sched.clock = lambda: clk[0]
    qr = str(tmp_path / "Q")
    fn = submit(
        qr, [JobManifest.from_dict({"job_id": "hang", "job_type": jt.id, "outputs": ["out/o.txt"]})]
    )
    for _ in range(400):  # until warm and watchdog armed
        sched.tick()
        if inflight(sched, "hang") and sched.ledger.is_resident(jt.id):
            break
        time.sleep(0.005)
    assert inflight(sched, "hang")
    assert sched.ledger.residents[jt.id].busy  # running == 1 (in-flight)
    later = []
    for jid in behind:
        m = JobManifest.from_dict(
            {"job_id": jid, "job_type": jt.id, "outputs": ["out/o.txt"], "options": {"text": jid}}
        )
        later.append(submit(qr, [m]))
    for _ in range(50):  # dispatched into the hung container's queue, not yet started
        sched.tick()
        if not any(p.manifest.job_id in behind for p in sched.pending):
            break
        time.sleep(0.005)
    sc = sched._serve[jt.id]
    clk[0] += 1e12  # jump past the watchdog deadline
    return launcher, sched, jt, qr, fn, later, sc


def test_watchdog_retires_the_hung_container(tmp_path, monkeypatch):
    """A serve job that never reports is failed by the watchdog, and its container is
    retired: a manager stuck in one job would otherwise hold every later job of the type
    with no deadline. The slot is freed rather than left busy, and the container is
    force-removed by name."""
    launcher, sched, jt, qr, fn, _, sc = _hang_then(tmp_path, monkeypatch)
    for _ in range(200):
        sched.tick()
        if os.path.exists(os.path.join(qr, "outbox", fn)):
            sched.tick()
            break
        time.sleep(0.005)
    res = wait_result(qr, fn, timeout_s=5)
    assert res[0].status == "failed"
    assert "watchdog" in res[0].message
    assert not sched.ledger.is_resident(jt.id)  # slot freed, not pinned busy
    assert jt.id not in sched._serve and sc.dying and sc.stop.is_set()
    for _ in range(200):
        if sc.name in launcher.killed_by_name:
            break
        time.sleep(0.01)
    assert sc.name in launcher.killed_by_name


def test_next_job_behind_a_permanent_hang_still_runs(tmp_path, monkeypatch):
    """The job queued behind a hung one never started, so it is re-queued without spending
    a retry (none are allowed here) and runs on a fresh container."""
    launcher, sched, jt, qr, fn, (f2,), sc = _hang_then(tmp_path, monkeypatch, behind=("next",))
    sched.cfg.max_retries = 0
    assert drive(sched, qr, f2, ticks=800)
    (res,) = wait_result(qr, f2, timeout_s=5)
    assert res.status == "done", res.message
    assert wait_result(qr, fn, timeout_s=5)[0].status == "failed"
    assert launcher.runs == 2  # a new container, not the hung one
    assert sched._serve[jt.id].epoch != sc.epoch


def test_serve_gone_drains_late_queued_job(tmp_path, monkeypatch):
    """When a serve container dies, a job the loop already dispatched into its jobq (that
    the exited manager will never read) is taken back on serve_gone, or else the batch wedges
    forever and a --wait submitter hangs. It never started, so it is re-queued without
    spending a retry (none are allowed here) and runs on a fresh container. Also exercises
    the sc.thread wiring."""
    launcher = FakeServe()
    monkeypatch.setattr(serve, "connect", fake_connect(launcher))
    jt = serve_jt()
    sched = _sched(tmp_path, registry(jt), launcher)
    sched.cfg.max_retries = 0
    qr = str(tmp_path / "Q")
    f0 = submit(
        qr,
        [
            JobManifest.from_dict(
                {
                    "job_id": "warm",
                    "job_type": jt.id,
                    "outputs": ["out/w.txt"],
                    "options": {"text": "W"},
                }
            )
        ],
    )
    assert drive(sched, qr, f0)
    assert wait_result(qr, f0, timeout_s=5)[0].status == "done"
    sc = sched._serve[jt.id]
    # Force the manager to exit (as a crash/stop would) and join it via sc.thread: the
    # container is now dead + has posted its serve_gone, but the loop has not processed it yet.
    sc.stop.set()
    assert sc.thread is not None
    sc.thread.join(5.0)
    assert not sc.thread.is_alive()
    # Dispatch a new job straight into the dead container's jobq: the window under test.
    from gpuworker.scheduler import Batch, PendingJob
    from gpuworker.share import QueueTree

    q = QueueTree(qr).ensure()
    wd, od = tmp_path / "wd", tmp_path / "od"
    os.makedirs(wd, exist_ok=True)
    os.makedirs(od, exist_ok=True)
    b = Batch(queue=q, filename="late.zip", work_dir=str(wd), out_dir=str(od))
    pj = PendingJob(
        manifest=JobManifest.from_dict({"job_id": "late", "job_type": jt.id}), jt=jt, batch=b
    )
    b.jobs.append(pj)
    sched.batches.append(b)
    sc.jobq.put(pj)
    runs_before = launcher.runs
    for _ in range(400):  # serve_gone drains the jobq, re-queues it, a new container runs it
        sched.tick()
        if pj.manifest.is_terminal:
            break
        time.sleep(0.005)
    assert pj.manifest.status == "done", pj.manifest.message
    assert pj.requeues == 1  # re-queued as never started, so no retry was spent
    assert launcher.runs == runs_before + 1  # on a fresh container
    assert wait_result(qr, "late.zip", timeout_s=5)[0].status == "done"  # shipped: no --wait hang


def test_serve_container_name_has_epoch():
    """The serve container name carries an epoch suffix so an evicted container's trailing
    async `rm -f <name>` can't remove a freshly-spawned same-type container."""
    from gpuworker.launcher import SERVE_NAME_PREFIX, serve_container_name

    n1 = serve_container_name("example.large", 1)
    n2 = serve_container_name("example.large", 2)
    assert n1 != n2
    assert n1.startswith(SERVE_NAME_PREFIX) and n2.startswith(SERVE_NAME_PREFIX)
    assert n1.endswith("-1") and n2.endswith("-2")
    assert "/" not in n1  # job_type id sep is sanitized


def test_serve_container_death_fails_job_and_frees(tmp_path, monkeypatch):
    launcher = FakeServe(crash_on={"die"})
    monkeypatch.setattr(serve, "connect", fake_connect(launcher))
    jt = serve_jt()
    sched = _sched(tmp_path, registry(jt), launcher)
    qr = str(tmp_path / "Q")
    fn = submit(
        qr, [JobManifest.from_dict({"job_id": "die", "job_type": jt.id, "outputs": ["out/o.txt"]})]
    )
    assert drive(sched, qr, fn)
    res = wait_result(qr, fn, timeout_s=5)
    assert res[0].status == "failed"
    # the dead container's resident is removed + VRAM freed (serve_gone)
    for _ in range(200):
        sched.tick()
        if jt.id not in sched._serve and not sched.ledger.is_resident(jt.id):
            break
        time.sleep(0.005)
    assert jt.id not in sched._serve
    assert not sched.ledger.is_resident(jt.id)


def test_manager_threads_never_read_footprint_state(tmp_path, monkeypatch):
    """The single-writer rule: footprint state is loop-owned. A recycle's reclaim target
    used to be computed by calling Scheduler._eff from the manager thread; now the loop
    hands it over as a value."""
    launcher = FakeServe()
    monkeypatch.setattr(serve, "connect", fake_connect(launcher))
    jt = serve_jt(id="example.large", max_jobs=1)
    sched = _sched(tmp_path, registry(jt), launcher)
    loop_thread = threading.current_thread()
    callers = []
    real_eff = sched._eff

    def eff(t):
        callers.append(threading.current_thread())
        return real_eff(t)

    monkeypatch.setattr(sched, "_eff", eff)
    qr = str(tmp_path / "Q")
    for jid in ("a", "b"):  # max_jobs=1, so job b runs after a recycle
        fn = submit(
            qr, [JobManifest.from_dict({"job_id": jid, "job_type": jt.id, "outputs": ["o.txt"]})]
        )
        assert drive(sched, qr, fn)
        assert wait_result(qr, fn, timeout_s=5)[0].status == "done"
    assert launcher.runs == 2  # the recycle happened
    assert callers and all(t is loop_thread for t in callers)
    assert sched._serve[jt.id].respawn_free_target_mb == jt.vram_mb + sched.cfg.headroom_mb


class _NeverReady(FakeServe):
    """A serve container that connects but never sends `ready`, as a model stalling in its
    load would. Its socket closes when the broker removes it by name."""

    def run(self, spec):
        import socket as _socket

        self.runs += 1
        broker, worker = _socket.socketpair()
        self._broker_ends[spec.serve_socket_host] = broker
        self._by_name[spec.name] = (broker, worker)
        return _Held((broker, worker))


class _Held:
    def __init__(self, ends):
        self.ends = ends
        self.returncode = None

    def poll(self):
        return None

    def wait(self, timeout=None):
        return 0


def test_a_container_that_never_loads_is_retired_by_the_loop(tmp_path, monkeypatch):
    """No job watchdog is armed until the first job starts, and a loading container holds
    its VRAM slot. With the manager's own ready timeout set far off, the loop's load
    deadline is what retires it; the job spends its retry rather than loop forever."""
    launcher = _NeverReady()
    monkeypatch.setattr(serve, "connect", fake_connect(launcher))
    jt = serve_jt()
    now = [1000.0]
    cfg = make_config(tmp_path, container_runtime="docker", max_retries=0)
    cfg.serve_ready_timeout_s = 3600.0
    sched = Scheduler(cfg, registry(jt), launcher, gpu.MockGpu(), clock=lambda: now[0])
    qr = str(tmp_path / "Q")
    fn = submit(qr, [JobManifest.from_dict({"job_id": "a", "job_type": jt.id})])
    # Wait for the manager to launch the container, not just for the loop to create its
    # record: a retire before the launch has nothing to remove by name.
    for _ in range(400):
        sched.tick()
        if launcher._by_name:
            break
        time.sleep(0.005)
    sc = sched._serve[jt.id]
    now[0] += sched._serve_load_window() - 1
    sched.tick()
    assert jt.id in sched._serve  # not yet
    now[0] += 2
    assert drive(sched, qr, fn, ticks=200)
    (res,) = wait_result(qr, fn, timeout_s=5)
    assert res.status == "failed" and "did not load within" in res.message
    assert not sched.ledger.is_resident(jt.id)
    for _ in range(200):
        if sc.name in launcher.killed_by_name:
            break
        time.sleep(0.01)
    assert sc.name in launcher.killed_by_name
    assert sc.thread is not None
    sc.thread.join(5)
    assert not sc.thread.is_alive()  # the manager's read ended when its socket closed


class _StopsReading(FakeServe):
    """A serve worker that sends `ready` and then never reads its socket again, as a model
    wedged mid-job would. Small socket buffers make a large run frame fill them, so every
    later send from the broker blocks."""

    def run(self, spec):
        import socket as _socket

        self.runs += 1
        broker, worker = _socket.socketpair()
        for end in (broker, worker):
            end.setsockopt(_socket.SOL_SOCKET, _socket.SO_SNDBUF, 4096)
            end.setsockopt(_socket.SOL_SOCKET, _socket.SO_RCVBUF, 4096)
        serve.send_frame(worker, serve.ready_msg())
        self._broker_ends[spec.serve_socket_host] = broker
        self._by_name[spec.name] = (broker, worker)
        return _Held((broker, worker))


def test_a_worker_that_stops_reading_is_still_torn_down(tmp_path, monkeypatch):
    """The job times out with its run frame stuck in a full socket buffer. The unload
    frame after it cannot be sent either; that send gives up, and the container is still
    terminated and its slot released."""
    launcher = _StopsReading()
    monkeypatch.setattr(serve, "connect", fake_connect(launcher))
    monkeypatch.setattr(serve, "UNLOAD_TIMEOUT_S", 0.2)
    jt = serve_jt(job_timeout=0.5)
    cfg = make_config(tmp_path, container_runtime="docker", max_retries=0)
    sched = Scheduler(cfg, registry(jt), launcher, gpu.MockGpu())
    qr = str(tmp_path / "Q")
    big = {"job_id": "a", "job_type": jt.id, "options": {"pad": "x" * 600_000}}
    fn = submit(qr, [JobManifest.from_dict(big)])
    for _ in range(200):
        sched.tick()
        if jt.id in sched._serve:
            break
        time.sleep(0.005)
    sc = sched._serve[jt.id]
    assert drive(sched, qr, fn, ticks=800)
    (res,) = wait_result(qr, fn, timeout_s=5)
    assert res.status == "failed" and "timed out" in res.message
    assert sc.thread is not None
    sc.thread.join(5)
    assert not sc.thread.is_alive()
    assert launcher.terminated == 1
    for _ in range(200):
        sched.tick()
        if jt.id not in sched._serve:
            break
        time.sleep(0.005)
    assert jt.id not in sched._serve and not sched.ledger.is_resident(jt.id)


def test_a_serve_dir_that_cannot_be_created_fails_the_job_and_reserves_nothing(
    tmp_path, monkeypatch
):
    """The job was taken off pending, marked running and given a VRAM slot before its
    directory was made. A full state disk then escaped the tick, and the job and its slot
    were stranded with no container and no deadline."""
    import errno
    import tempfile

    from gpuworker import scheduler as scheduler_mod

    launcher = FakeServe()
    monkeypatch.setattr(serve, "connect", fake_connect(launcher))
    jt = serve_jt()
    sched = Scheduler(
        make_config(tmp_path, container_runtime="docker", max_retries=0),
        registry(jt),
        launcher,
        gpu.MockGpu(),
    )
    real = tempfile.mkdtemp

    def mkdtemp(prefix=None, dir=None):
        if dir and os.path.basename(dir) == "serve":
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(prefix=prefix, dir=dir)

    monkeypatch.setattr(scheduler_mod.tempfile, "mkdtemp", mkdtemp)
    qr = str(tmp_path / "Q")
    fn = submit(qr, [JobManifest.from_dict({"job_id": "a", "job_type": jt.id})])
    assert drive(sched, qr, fn)
    (res,) = wait_result(qr, fn, timeout_s=5)
    assert res.status == "failed" and "No space left" in res.message
    assert not sched.ledger.residents and not sched._serve and not sched.pending
    assert launcher.runs == 0


class _Ordered(FakeServe):
    """An echo serve worker that holds each job until *gate* is set and records the order
    it ran them in, by client job_id."""

    def __init__(self):
        super().__init__()
        self.order = []
        self.gate = threading.Event()

    def run(self, spec):
        import socket as _socket

        from helpers import FakeProc

        self.runs += 1
        broker, worker = _socket.socketpair()
        out_root = spec.out_dir

        def go():
            serve.send_frame(worker, serve.ready_msg())
            while True:
                try:
                    msg = serve.recv_frame(worker)
                except (serve.ProtocolError, OSError):
                    return
                if msg.get("type") != "run":
                    return
                self.gate.wait(10)
                key, man = msg["job_id"], msg["manifest"]
                self.order.append(man["job_id"])
                for rel in man["outputs"]:
                    p = os.path.join(out_root, key, rel)
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    with open(p, "w", encoding="utf-8") as fh:
                        fh.write("x\n")
                serve.send_frame(worker, serve.result_msg(key, "done", man["outputs"], ""))

        t = threading.Thread(target=go, daemon=True)
        t.start()
        self._broker_ends[spec.serve_socket_host] = broker
        self._by_name[spec.name] = (broker, worker)
        return FakeProc(t, (broker, worker))


def test_a_later_higher_priority_job_passes_queued_serve_work(tmp_path, monkeypatch):
    """Every pending job of a warm type was handed to its container at once, in a FIFO,
    so a high-priority job submitted a moment later ran after all of them."""
    launcher = _Ordered()
    monkeypatch.setattr(serve, "connect", fake_connect(launcher))
    jt = serve_jt()
    sched = Scheduler(
        make_config(tmp_path, container_runtime="docker"), registry(jt), launcher, gpu.MockGpu()
    )
    qr = str(tmp_path / "Q")

    def job(job_id, priority=0):
        return JobManifest.from_dict(
            {"job_id": job_id, "job_type": jt.id, "outputs": [f"o/{job_id}"], "priority": priority}
        )

    low = submit(qr, [job(f"l{i}") for i in range(1, 6)])
    for _ in range(400):
        sched.tick()
        if launcher.runs:
            break
        time.sleep(0.005)
    for _ in range(20):
        sched.tick()
        time.sleep(0.005)
    high = submit(qr, [job("h", priority=10)])
    for _ in range(5):
        sched.tick()
    launcher.gate.set()
    assert drive(sched, qr, low) and drive(sched, qr, high)
    # l1 was running and l2 queued behind it; h goes before the rest.
    assert launcher.order == ["l1", "l2", "h", "l3", "l4", "l5"]
    sched.shutdown()


@pytest.mark.skipif(
    os.name == "nt", reason="Windows emulates append mode in the C runtime; serve runs on Linux"
)
def test_the_serve_log_is_capped_between_jobs_and_the_container_keeps_writing(tmp_path):
    """serve.log grew for as long as a warm container lived."""
    import sys

    from gpuworker.launcher import DockerLauncher, LaunchSpec
    from gpuworker.serve_manager import ServeContainer

    sc = ServeContainer(
        jt=serve_jt(), epoch=1, name="n", root=str(tmp_path), jobq=None, stop=threading.Event()
    )
    go = tmp_path / "go"
    child = (
        "import os, sys, time\n"
        "sys.stdout.write('A' * 5000); sys.stdout.flush()\n"
        f"while not os.path.exists({str(go)!r}): time.sleep(0.01)\n"
        "sys.stdout.write('after'); sys.stdout.flush()\n"
    )
    launcher = DockerLauncher()
    launcher.build_argv = lambda spec: [sys.executable, "-c", child]
    spec = LaunchSpec(
        job_type=serve_jt(), bundle_dir="", out_dir="", log_path=sc.log_path, name="n"
    )
    proc = launcher.run(spec)
    try:
        for _ in range(500):
            if sc.log_size() >= 5000:
                break
            time.sleep(0.01)
        sc.cap_log(max_bytes=10_000)
        assert sc.log_size() == 5000  # under the cap: kept
        sc.cap_log(max_bytes=1000)
        assert sc.log_size() == 0
        go.write_text("")
        proc.wait(10)
    finally:
        if proc.poll() is None:
            proc.kill()
    # Appended at the new end, not after a hole of NUL bytes where the old output was.
    assert open(sc.log_path, "rb").read() == b"after"


def test_the_reclaim_deadline_waits_for_an_evicted_container_to_go(tmp_path, monkeypatch):
    """The gate's deadline started when the job parked, in the same step that began
    evicting a warm container. An unload slower than reclaim_timeout_s then marked a
    healthy card degraded."""
    release = threading.Event()

    class SlowTeardown(FakeServe):
        def cleanup(self, spec):
            if "ex.a" in spec.name:
                release.wait(10)  # the evicted container is still unloading
            super().cleanup(spec)

    launcher = SlowTeardown()
    monkeypatch.setattr(serve, "connect", fake_connect(launcher))
    a = serve_jt(id="ex.a", vram=10000)
    b = serve_jt(id="ex.b", vram=10000)
    g = gpu.MockGpu(total_mb=16000, used_mb=200)
    sched = _sched(tmp_path, registry(a, b), launcher, monitor=g)
    sched.cfg.reclaim_timeout_s = 0.2
    qr = str(tmp_path / "Q")
    fa = submit(
        qr, [JobManifest.from_dict({"job_id": "a1", "job_type": "ex.a", "outputs": ["out/a.txt"]})]
    )
    assert drive(sched, qr, fa)
    g.used = 10200  # a's model, still in memory
    fb = submit(
        qr, [JobManifest.from_dict({"job_id": "b1", "job_type": "ex.b", "outputs": ["out/b.txt"]})]
    )
    end = time.monotonic() + 1.0
    while time.monotonic() < end:
        sched.tick()
        time.sleep(0.01)
    assert "b1" in parked_ids(sched)
    assert not sched._gpu_degraded
    g.used = 200
    release.set()
    assert drive(sched, qr, fb)
    assert wait_result(qr, fb, timeout_s=5)[0].status == "done"
    assert not sched._gpu_degraded
