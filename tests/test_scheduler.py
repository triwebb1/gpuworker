"""Scheduler correctness: intake validation, false-success checks, weight verification,
the reclaim gate, admission limits, stale reclaim, and the watchdog."""

import os
import time

import pytest

from gpuworker import gpu, weights
from gpuworker.client import submit, wait_result
from gpuworker.launcher import make_launcher
from gpuworker.manifest import JobManifest
from gpuworker.module import JobType, Registry
from gpuworker.scheduler import Scheduler

from helpers import make_config, parked_ids


def _reg_with(jt: JobType) -> Registry:
    r = Registry()
    r.job_types[jt.id] = jt
    return r


def _vram_jt(id="ex.model", vram=5000, co=True, modes=("oneshot",)):
    return JobType(
        id=id,
        resource_class="vram",
        vram_mb=vram,
        env="host",
        entrypoint=["python", "-c", "pass"],
        modes=tuple(modes),
        co_resident=co,
        sandbox="standard",
    )


def _sched(tmp_path, reg, monitor=None, runtime="local"):
    cfg = make_config(
        tmp_path,
        queue_roots=[str(tmp_path / "Q")],
        registry_dir="modules",
        container_runtime=runtime,
        weights_dir=str(tmp_path / "w"),
    )
    return Scheduler(cfg, reg, make_launcher(runtime), monitor or gpu.MockGpu())


@pytest.mark.filterwarnings("ignore:Duplicate name:UserWarning")  # the duplicate is the point
def test_duplicate_job_id_fails_batch(tmp_path):
    reg = _reg_with(_vram_jt())
    queue_root = str(tmp_path / "Q")
    # build two manifests with the same id into one bundle by packing directly
    from gpuworker import bundle
    from gpuworker.share import QueueTree

    q = QueueTree(queue_root).ensure()
    bundle.pack(
        os.path.join(q.dirs["inbox"], "dup.zip"),
        [{"job_id": "x", "job_type": "ex.model"}, {"job_id": "x", "job_type": "ex.model"}],
        root=str(tmp_path),
    )
    sched = _sched(tmp_path, reg)
    sched.tick()
    res = wait_result(queue_root, "dup.zip", timeout_s=5)
    assert all(r.status == "failed" for r in res)
    assert any("two bundle members name the same path" in r.message for r in res)


def test_no_outputs_produced_is_failure(tmp_path):
    # entrypoint exits 0 but writes nothing; job declares an output -> must fail.
    reg = _reg_with(
        JobType(
            id="cpu.noop",
            resource_class="cpu",
            vram_mb=0,
            env="host",
            entrypoint=["python", "-c", "pass"],
            modes=("oneshot",),
            sandbox="none",
        )
    )
    queue_root = str(tmp_path / "Q")
    m = JobManifest.from_dict(
        {"job_id": "n1", "job_type": "cpu.noop", "outputs": ["out/missing.txt"]}
    )
    fn = submit(queue_root, [m])
    sched = _sched(tmp_path, reg)
    for _ in range(50):
        sched.tick()
        if os.path.exists(os.path.join(queue_root, "outbox", fn)):
            sched.tick()
            break
        time.sleep(0.02)
    res = wait_result(queue_root, fn, timeout_s=5)
    assert res[0].status == "failed"
    assert "did not produce" in res[0].message


def test_a_partial_set_of_outputs_is_a_failure(tmp_path):
    """Only a job with no outputs at all used to fail; one that wrote some of its declared
    outputs and exited 0 counted as done."""
    from helpers import cpu_jt, registry

    code = "import os; open(os.path.join(os.environ['GPUWORKER_OUT'], 'a.txt'), 'w').write('x')"
    sched = _sched(tmp_path, registry(cpu_jt(code=code)))
    qr = str(tmp_path / "Q")
    m = JobManifest.from_dict({"job_id": "p", "job_type": "ex.cpu", "outputs": ["a.txt", "b.txt"]})
    fn = submit(qr, [m])
    for _ in range(100):
        sched.tick()
        if os.path.exists(os.path.join(qr, "outbox", fn)):
            break
        time.sleep(0.02)
    (res,) = wait_result(qr, fn, timeout_s=5)
    assert res.status == "failed" and "'b.txt'" in res.message and "'a.txt'" not in res.message


def test_bad_weight_cache_fails_job(tmp_path):
    # A staged weight cache that fails its hash manifest must fail the job at launch.
    reg = _reg_with(
        JobType(
            id="cpu.noop",
            resource_class="cpu",
            vram_mb=0,
            env="host",
            entrypoint=["python", "-c", "pass"],
            modes=("oneshot",),
            sandbox="none",
        )
    )
    queue_root = str(tmp_path / "Q")
    cache = tmp_path / "w" / "cpu.noop"
    cache.mkdir(parents=True)
    (cache / "model.bin").write_text("real")
    (cache / "weights.sha256").write_text("0" * 64 + "  model.bin\n")  # wrong hash
    m = JobManifest.from_dict({"job_id": "w1", "job_type": "cpu.noop", "outputs": ["out/w1.txt"]})
    fn = submit(queue_root, [m])
    sched = _sched(tmp_path, reg)
    out = os.path.join(queue_root, "outbox", fn)
    for _ in range(100):
        sched.tick()
        if os.path.exists(out):
            sched.tick()
            break
        time.sleep(0.02)
    res = wait_result(queue_root, fn, timeout_s=5)
    assert res[0].status == "failed"
    assert "weight cache verification failed" in res[0].message


def test_required_cpu_weight_cache_must_exist_and_be_sealed(tmp_path):
    jt = JobType(
        id="example.cpu_model",
        resource_class="cpu",
        vram_mb=0,
        env="container",
        entrypoint=["python", "-m", "example.cpu_worker"],
        modes=("oneshot",),
        sandbox="standard",
        weights_required=True,
    )
    sched = _sched(tmp_path, _reg_with(jt))
    with pytest.raises(RuntimeError, match="required weight cache missing"):
        sched._weights_for(jt)

    cache = tmp_path / "w" / jt.id
    cache.mkdir(parents=True)
    (cache / "model.safetensors").write_text("model")
    with pytest.raises(RuntimeError, match="unsealed"):
        sched._weights_for(jt)

    weights.write_manifest(str(cache))
    assert sched._weights_for(jt) == str(cache)


def test_required_weight_launch_injects_fixed_identity_and_type_timeout(tmp_path, monkeypatch):
    class Done:
        returncode = 0
        stdout = stderr = None

        def wait(self, timeout=None):
            return 0

    class RecordingLauncher:
        def __init__(self):
            self.spec = None

        def run(self, spec):
            self.spec = spec
            return Done()

        def terminate(self, proc, spec):
            pass

        def terminate_by_name(self, name):
            pass

        def reap_serve_orphans(self):
            pass

        def cleanup(self, spec):
            pass

    revision = "f" * 40
    jt = JobType(
        id="example.cpu_model",
        resource_class="cpu",
        vram_mb=0,
        env="container",
        entrypoint=["python", "-m", "example.cpu_worker", "--job", "cpu_model"],
        image="sandbox@sha256:" + "a" * 64,
        modes=("oneshot",),
        sandbox="standard",
        weights_required=True,
        oneshot_job_timeout_s=900,
        fixed_env={"EXAMPLE_SOURCE_REVISION": revision},
    )
    reg = _reg_with(jt)
    queue_root = str(tmp_path / "Q")
    cfg = make_config(
        tmp_path,
        queue_roots=[queue_root],
        registry_dir="modules",
        container_runtime="docker",
        weights_dir=str(tmp_path / "w"),
    )
    cache = tmp_path / "w" / jt.id
    cache.mkdir(parents=True)
    (cache / "model.safetensors").write_text("model")
    weights.write_manifest(str(cache))
    launcher = RecordingLauncher()
    sched = Scheduler(cfg, reg, launcher, gpu.MockGpu())
    from gpuworker import oneshot

    timeouts = []
    real_await = oneshot.await_proc

    def await_proc(proc, spec, timeout, *a, **kw):
        timeouts.append(timeout)
        return real_await(proc, spec, timeout, *a, **kw)

    monkeypatch.setattr(oneshot, "await_proc", await_proc)

    fn = submit(queue_root, [JobManifest.from_dict({"job_id": "p1", "job_type": jt.id})])
    out = os.path.join(queue_root, "outbox", fn)
    for _ in range(100):
        sched.tick()
        if os.path.exists(out):
            sched.tick()
            break
        time.sleep(0.02)

    result = wait_result(queue_root, fn, timeout_s=5)
    assert result[0].status == "done"
    assert timeouts == [900]
    assert launcher.spec.weights_dir == str(cache)
    assert launcher.spec.extra_env == {
        "EXAMPLE_SOURCE_REVISION": revision,
        "GPUWORKER_JOB_ID": "p1",
    }


def test_reclaim_gate_blocks_admission_until_free(tmp_path):
    # A heavy non-resident vram job must wait for NVML to show room even with an
    # empty eviction plan (prior-context-draining case).
    reg = _reg_with(_vram_jt(id="ex.heavy", vram=14000, co=False))
    g = gpu.MockGpu(total_mb=16000, used_mb=15000)  # only 1000 free
    sched = _sched(tmp_path, reg, monitor=g)
    sched.cfg.reclaim_timeout_s = 0.0  # deadline immediate -> fails on the next service
    sched.cfg.max_retries = 0  # this test is about the gate, not retry (test_retry.py)
    queue_root = str(tmp_path / "Q")
    m = JobManifest.from_dict({"job_id": "h1", "job_type": "ex.heavy"})
    fn = submit(queue_root, [m])
    # tick-spanning gate: tick 1 parks the non-resident vram job; a later tick's
    # _service_reclaim sees free<target + deadline passed -> fail + degrade.
    for _ in range(3):
        sched.tick()
    res = wait_result(queue_root, fn, timeout_s=5)
    assert res[0].status == "failed"
    assert "reclaim gate" in res[0].message
    assert sched._gpu_degraded


def test_reclaim_gate_admits_when_memory_frees_without_blocking(tmp_path):
    # The gate is tick-spanning: a non-resident vram job parks (loop not blocked),
    # then _service_reclaim admits it once NVML free rises above target.
    jt = JobType(
        id="ex.gated",
        resource_class="vram",
        vram_mb=10000,
        co_resident=False,
        env="host",
        entrypoint=["python", "modules/echo/src/worker.py"],
        modes=("oneshot",),
        sandbox="standard",
    )
    reg = _reg_with(jt)
    g = gpu.MockGpu(total_mb=16000, used_mb=15000)  # 1000 free < 10000+headroom
    sched = _sched(tmp_path, reg, monitor=g)
    queue_root = str(tmp_path / "Q")
    m = JobManifest.from_dict(
        {
            "job_id": "g1",
            "job_type": "ex.gated",
            "outputs": ["out/g1.txt"],
            "options": {"text": "gated-ok"},
        }
    )
    fn = submit(queue_root, [m])
    sched.tick()
    assert "g1" in parked_ids(sched)  # parked, not blocked, not admitted
    g.used = 1000  # free 15000 >= 10000+1024
    out = os.path.join(queue_root, "outbox", fn)
    for _ in range(100):
        sched.tick()
        if os.path.exists(out):
            sched.tick()
            break
        time.sleep(0.02)
    res = wait_result(queue_root, fn, timeout_s=5)
    assert res[0].status == "done"
    assert not sched._pending_reclaim


def test_two_parked_vram_jobs_do_not_overcommit(tmp_path):
    # Two heavy non-co-resident vram jobs both park; NVML free stays high, but the
    # ledger (not just lagging NVML) gates the second, so they never both admit and
    # the resident footprint never exceeds the budget.
    def heavy(jid):
        return JobType(
            id=jid,
            resource_class="vram",
            vram_mb=10000,
            co_resident=False,
            env="host",
            entrypoint=["python", "-c", "import time; time.sleep(1.0)"],
            modes=("oneshot",),
            sandbox="standard",
        )

    reg = Registry()
    reg.job_types["ex.a"] = heavy("ex.a")
    reg.job_types["ex.b"] = heavy("ex.b")
    g = gpu.MockGpu(total_mb=16000, used_mb=500)  # free ~15500, always high
    sched = _sched(tmp_path, reg, monitor=g)
    qr = str(tmp_path / "Q")
    submit(qr, [JobManifest.from_dict({"job_id": "a", "job_type": "ex.a"})])
    submit(qr, [JobManifest.from_dict({"job_id": "b", "job_type": "ex.b"})])
    sched.tick()  # intake + park both
    for _ in range(5):  # fast ticks; first job still sleeping
        sched.tick()
        assert sched.ledger.resident_vram_mb <= sched.ledger.admit_budget_mb
    # exactly one admitted (10000); the other gated by the ledger, not over-committed
    assert sched.ledger.resident_vram_mb == 10000


def test_same_type_oneshot_jobs_serialize_at_reclaim_gate(tmp_path):
    # A batch of same-type oneshot vram jobs must not all park and load at once: each
    # loads its own model copy, so N concurrent loads can OOM. vram here is small enough
    # that the budget alone would admit ~7, so only the same-type serialization guard
    # (hold the rest while one is parked at the gate) prevents it.
    jt = JobType(
        id="ex.other",
        resource_class="vram",
        vram_mb=2000,
        co_resident=False,
        env="host",
        entrypoint=["python", "-c", "import time; time.sleep(1.0)"],
        modes=("oneshot",),
        sandbox="standard",
    )
    reg = _reg_with(jt)
    g = gpu.MockGpu(total_mb=16000, used_mb=500)  # free ~15500: budget fits ~7
    sched = _sched(tmp_path, reg, monitor=g)
    qr = str(tmp_path / "Q")
    submit(
        qr, [JobManifest.from_dict({"job_id": f"m{i}", "job_type": "ex.other"}) for i in range(6)]
    )
    sched.tick()  # intake + admit pass
    parked = sum(1 for (p, _, _) in sched._pending_reclaim.values() if p.jt.id == "ex.other")
    resident = sum(1 for r in sched.ledger.residents.values() if r.jt.id == "ex.other")
    assert parked + resident <= 1, f"expected <=1 same-type in flight, got {parked}+{resident}"
    for _ in range(5):  # fast ticks; job still sleeping
        sched.tick()
        assert sched.ledger.resident_vram_mb <= sched.ledger.admit_budget_mb
        assert (
            sum(1 for r in sched.ledger.residents.values() if r.jt.id == "ex.other") <= 1
        )  # never >1 concurrent same-type


def test_three_exclusive_cold_types_do_not_degrade_a_healthy_card(tmp_path):
    """Different exclusive types arriving together used to all park in one tick. The
    first was admitted, the ledger then held the other two back, and when the first ran
    past reclaim_timeout_s they timed out at the gate: the GPU was marked degraded and a
    job quarantined, on a card that was fine."""
    reg = Registry()
    for jid in ("ex.speech", "ex.check", "ex.audio"):
        reg.job_types[jid] = JobType(
            id=jid,
            resource_class="vram",
            vram_mb=9000,
            co_resident=False,
            env="host",
            entrypoint=["python", "-c", "import time; time.sleep(0.6)"],
            modes=("oneshot",),
            sandbox="standard",
        )
    g = gpu.MockGpu(total_mb=16000, used_mb=500)  # the card always has room
    sched = _sched(tmp_path, reg, monitor=g)
    sched.cfg.reclaim_timeout_s = 0.1  # far shorter than each job runs
    qr = str(tmp_path / "Q")
    fn = submit(
        qr,
        [
            JobManifest.from_dict({"job_id": jid.split(".")[1], "job_type": jid})
            for jid in reg.job_types
        ],
    )
    for _ in range(600):
        sched.tick()
        assert len(sched._pending_reclaim) <= 1
        if os.path.exists(os.path.join(qr, "outbox", fn)):
            break
        time.sleep(0.01)
    assert not sched._gpu_degraded
    assert [r.status for r in wait_result(qr, fn, timeout_s=5)] == ["done"] * 3


def test_over_budget_vram_job_fails_cleanly(tmp_path):
    # A footprint exceeding the budget even on an empty card must fail (not strand).
    reg = _reg_with(_vram_jt(id="ex.huge", vram=999999, co=False))
    queue_root = str(tmp_path / "Q")
    m = JobManifest.from_dict({"job_id": "hb", "job_type": "ex.huge"})
    fn = submit(queue_root, [m])
    sched = _sched(tmp_path, reg)
    for _ in range(3):
        sched.tick()
    res = wait_result(queue_root, fn, timeout_s=5)
    assert res[0].status == "failed"
    assert "exceeds VRAM budget" in res[0].message


def test_reclaim_stale_skips_active_bundle(tmp_path):
    from gpuworker.share import QueueTree

    q = QueueTree(str(tmp_path / "Q")).ensure()
    proc = os.path.join(q.dirs["processing"], "live.zip")
    with open(proc, "w") as fh:
        fh.write("x")
    old = os.path.getmtime(proc) - 10000
    os.utime(proc, (old, old))
    # skipped -> not reclaimed (stays in processing)
    assert q.reclaim_stale(3600, skip={"live.zip"}) == []
    assert os.path.exists(proc)
    # not skipped -> reclaimed back to inbox
    assert q.reclaim_stale(3600) == ["live.zip"]
    assert os.path.exists(os.path.join(q.dirs["inbox"], "live.zip"))


def test_degraded_recovers_when_memory_frees(tmp_path):
    reg = _reg_with(_vram_jt(id="ex.heavy", vram=14000, co=False))
    g = gpu.MockGpu(total_mb=16000, used_mb=15000)
    sched = _sched(tmp_path, reg, monitor=g)
    sched._gpu_degraded = True
    sched.tick()
    assert sched._gpu_degraded  # still tight
    g.used = 500  # memory frees
    sched.tick()
    assert not sched._gpu_degraded


def test_watchdog_fails_silent_job(tmp_path):
    reg = _reg_with(
        JobType(
            id="cpu.noop",
            resource_class="cpu",
            vram_mb=0,
            env="host",
            entrypoint=["python", "-c", "pass"],
            modes=("oneshot",),
            sandbox="none",
        )
    )
    sched = _sched(tmp_path, reg)
    # Inject a fake inflight entry with a past deadline and no completion.
    from gpuworker.scheduler import Batch, PendingJob
    from gpuworker.share import QueueTree

    q = QueueTree(str(tmp_path / "Q")).ensure()
    b = Batch(
        queue=q, filename="z.zip", work_dir=str(tmp_path / "wd"), out_dir=str(tmp_path / "od")
    )
    pj = PendingJob(
        manifest=JobManifest.from_dict({"job_id": "ghost", "job_type": "cpu.noop"}),
        jt=reg.get("cpu.noop"),
        batch=b,
    )
    b.jobs.append(pj)
    sched.batches.append(b)
    sched._inflight[pj.uid] = (pj, sched.clock() - 1.0)  # already overdue
    sched.tick()
    assert pj.manifest.status == "failed"
    assert "watchdog" in pj.manifest.message


def test_a_failed_archive_is_retried_by_the_next_reclaim(tmp_path, monkeypatch):
    from gpuworker.share import QueueTree

    reg = _reg_with(
        JobType(
            id="cpu.ok",
            resource_class="cpu",
            vram_mb=0,
            env="host",
            entrypoint=["python", "-c", "pass"],
            sandbox="none",
        )
    )
    sched = _sched(tmp_path, reg)
    qr = str(tmp_path / "Q")
    fn = submit(qr, [JobManifest.from_dict({"job_id": "a", "job_type": "cpu.ok"})])
    calls = []

    def flaky(self, name):
        calls.append(name)
        raise OSError("share went away")

    monkeypatch.setattr(QueueTree, "archive", flaky)
    for _ in range(400):
        sched.tick()
        if calls:
            break
        time.sleep(0.005)
    assert calls
    sched.tick()  # the reclaim step of this tick or the next archives it
    assert not os.path.exists(os.path.join(qr, "processing", fn))
    assert os.path.exists(os.path.join(qr, "archive", fn))


def test_host_job_types_need_an_opt_in_beside_a_container_runtime(tmp_path):
    """echo.echo is trusted code with no sandbox, on the host. A broker with a container
    runtime (podman or docker) runs it only if the operator asks."""
    reg = _reg_with(
        JobType(
            id="cpu.host",
            resource_class="cpu",
            vram_mb=0,
            env="host",
            entrypoint=["python", "-c", "pass"],
            sandbox="none",
        )
    )
    for allow, status in ((False, "failed"), (True, "done")):
        sched = _sched(tmp_path / str(allow), reg, runtime="docker")
        qr = str(tmp_path / str(allow) / "Q")
        sched.cfg.allow_host_jobs = allow
        fn = submit(qr, [JobManifest.from_dict({"job_id": "a", "job_type": "cpu.host"})])
        for _ in range(400):
            sched.tick()
            if os.path.exists(os.path.join(qr, "outbox", fn)):
                break
            time.sleep(0.005)
        (res,) = wait_result(qr, fn, timeout_s=5)
        assert res.status == status, res.message
        if not allow:
            assert "allow_host_jobs" in res.message


def test_the_idle_baseline_waits_for_running_cpu_jobs(tmp_path):
    """A host cpu job can touch the GPU, so its memory must not become the idle baseline."""
    sched = _sched(tmp_path, _reg_with(_vram_jt()))
    sched.ledger.admit_cpu()
    sched.tick()
    assert sched._used_baseline is None
    sched.ledger.finish_cpu()
    sched.tick()
    assert sched._used_baseline is not None
