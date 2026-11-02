"""Footprint reconciliation + load-cost-aware ordering in the scheduler."""

import os

from gpuworker import gpu
from gpuworker.config import Config
from gpuworker.launcher import make_launcher
from gpuworker.manifest import JobManifest
from gpuworker.module import JobType, Registry
from gpuworker.scheduler import Batch, PendingJob, Scheduler
from gpuworker.share import QueueTree


def _vram_jt(id, vram=5000, load_cost=10.0, co=True, profile="preallocated"):
    return JobType(
        id=id,
        resource_class="vram",
        vram_mb=vram,
        env="host",
        entrypoint=["python", "-c", "pass"],
        modes=("oneshot",),
        co_resident=co,
        vram_profile=profile,
        load_cost_s=load_cost,
        sandbox="standard",
    )


def _sched(tmp_path, jts, monitor=None):
    reg = Registry()
    for jt in jts:
        reg.job_types[jt.id] = jt
    cfg = Config(
        queue_roots=[],
        footprints_path=str(tmp_path / "fp.json"),
        state_dir=str(tmp_path / "state"),
    )
    return Scheduler(cfg, reg, make_launcher("local"), monitor or gpu.MockGpu(), queues=[])


def _pending(sched, jt, job_id, priority=0):
    q = QueueTree(str(jt.id))  # not used for I/O here
    # Under the scheduler's tmp state dir: a relative path would write into the cwd.
    wd, od = os.path.join(sched.state_dir, "wd"), os.path.join(sched.state_dir, "od")
    b = Batch(queue=q, filename="b.zip", work_dir=wd, out_dir=od)
    pj = PendingJob(
        manifest=JobManifest.from_dict({"job_id": job_id, "job_type": jt.id, "priority": priority}),
        jt=jt,
        batch=b,
    )
    sched.pending.append(pj)
    return pj


def test_reconcile_ratchets_effective_footprint(tmp_path):
    jt = _vram_jt("ex.m", vram=5000)
    sched = _sched(tmp_path, [jt])
    assert sched._eff(jt).vram_mb == 5000
    # Observe a model that actually peaked higher than declared.
    sched._reconcile_footprint(jt, baseline=1000, peak_used=8000)
    assert sched.footprints.observed("ex.m") == 7000
    assert sched._eff(jt).vram_mb == 7000  # admission now uses reality


def test_reconcile_noop_without_baseline(tmp_path):
    jt = _vram_jt("ex.m", vram=5000)
    sched = _sched(tmp_path, [jt])
    sched._reconcile_footprint(jt, baseline=None, peak_used=8000)
    assert sched.footprints.observed("ex.m") is None


def test_effective_footprint_changes_admission(tmp_path):
    # Two co-resident 5000 models fit by declared (10000) in a 12000 budget; once one
    # is observed at 9000, they no longer co-fit.
    a = _vram_jt("a.m", vram=5000)
    b = _vram_jt("b.m", vram=5000)
    sched = _sched(tmp_path, [a, b])
    sched.ledger.admit_budget_mb = 12000
    sched.ledger.admit_vram(sched._eff(a), now=1.0)
    assert sched.ledger.can_fit(sched._eff(b))  # 5000 + 5000 <= 12000
    sched.ledger.finish_vram("a.m", now=1.5, keep_warm_s=100)
    sched.ledger.evict("a.m")
    # now observe b huge
    sched.footprints.record("b.m", 9000)
    sched.ledger.admit_vram(sched._eff(a), now=2.0)
    assert not sched.ledger.can_fit(sched._eff(b))  # 5000 + 9000 > 12000


def test_pick_next_prefers_resident_model_at_equal_priority(tmp_path):
    warm = _vram_jt("warm.m", vram=4000)
    cold = _vram_jt("cold.m", vram=4000)
    sched = _sched(tmp_path, [warm, cold])
    sched.ledger.admit_vram(sched._eff(warm), now=1.0)
    sched.ledger.finish_vram("warm.m", now=1.5, keep_warm_s=100)  # resident, idle
    _pending(sched, cold, "c1", priority=0)
    pj_warm = _pending(sched, warm, "w1", priority=0)
    assert sched.pick_next() is pj_warm  # equal priority -> use loaded model


def test_priority_overrides_resident_model(tmp_path):
    # A higher-priority job for a different model preempts the resident model's queue,
    # forcing the unload/reload the submitter asked for.
    warm = _vram_jt("warm.m", vram=4000)
    cold = _vram_jt("cold.m", vram=4000)
    sched = _sched(tmp_path, [warm, cold])
    sched.ledger.admit_vram(sched._eff(warm), now=1.0)
    sched.ledger.finish_vram("warm.m", now=1.5, keep_warm_s=100)  # resident, idle
    pj_cold = _pending(sched, cold, "c1", priority=9)  # higher priority, cold
    _pending(sched, warm, "w1", priority=0)
    assert sched.pick_next() is pj_cold  # priority wins over batching


def test_pick_next_amortizes_costliest_cold_load(tmp_path):
    # Two cold models, equal priority; prefer the one whose (depth * load_cost) is
    # higher so the expensive load is spread over the most jobs.
    cheap = _vram_jt("cheap.m", vram=4000, load_cost=5.0)
    pricey = _vram_jt("pricey.m", vram=4000, load_cost=60.0)
    sched = _sched(tmp_path, [cheap, pricey])
    _pending(sched, cheap, "c1")
    _pending(sched, cheap, "c2")
    pj_pricey = _pending(sched, pricey, "p1")  # depth 1 * 60 = 60 > 2 * 5 = 10
    assert sched.pick_next() is pj_pricey


def test_observation_above_the_budget_is_rejected_not_recorded(tmp_path):
    # One contaminated sample larger than the whole budget would otherwise make the type
    # unschedulable for good (_unschedulable).
    jt = _vram_jt("ex.m", vram=5000)
    sched = _sched(tmp_path, [jt])
    sched._reconcile_footprint(jt, baseline=0, peak_used=sched.ledger.admit_budget_mb + 1)
    assert sched.footprints.observed("ex.m") is None
    assert sched._unschedulable(jt) is None


def test_no_baseline_while_another_lane_is_using_the_gpu(tmp_path):
    # A host-process cpu job can hold GPU memory, so a peak read while one runs
    # cannot be attributed to the vram job.
    jt = _vram_jt("ex.m")
    sched = _sched(tmp_path, [jt])
    sched.ledger.admit_cpu()
    assert not sched._attribution_clean()
    sched.ledger.finish_cpu()
    assert sched._attribution_clean()


def test_a_new_job_on_the_gpu_spoils_a_pending_baseline(tmp_path):
    jt = _vram_jt("ex.m")
    cpu = JobType(
        id="ex.cpu",
        resource_class="cpu",
        vram_mb=0,
        env="host",
        entrypoint=["python", "-c", "pass"],
        sandbox="none",
    )
    sched = _sched(tmp_path, [jt, cpu])
    sched._baseline["some-uid"] = 300  # a vram job loaded alone and is being measured
    pj = _pending(sched, cpu, "c")
    sched._admit_and_run(pj)
    assert sched._baseline == {}  # its later peak would include the cpu job's memory


def test_footprints_reset_command(tmp_path, capsys):
    from gpuworker import cli
    from gpuworker.footprints import Footprints

    fp_path = tmp_path / "fp.json"
    Footprints(str(fp_path)).record("ex.m", 9000, "img")
    cfg_path = tmp_path / "gpuworker.toml"
    cfg_path.write_text(f"footprints_path = {str(fp_path)!r}\n", encoding="utf-8")
    assert cli.main(["--config", str(cfg_path), "footprints", "reset", "ex.m"]) == 0
    assert Footprints(str(fp_path)).observed("ex.m") is None
    assert "reset ex.m" in capsys.readouterr().out


def test_a_cuda_oom_attempt_does_not_raise_the_footprint(tmp_path):
    """At an OOM the peak is what the card had, not what the model needs. Recording it
    made the gate wait for memory the card could never free."""
    from gpuworker.jobs import Completion

    jt = _vram_jt("ex.m", vram=5000)
    sched = _sched(tmp_path, [jt])
    pj = _pending(sched, jt, "j")
    sched.pending.remove(pj)
    sched.ledger.admit_vram(sched._eff(jt), now=1.0)
    sched._baseline[pj.uid] = 1000
    sched._on_complete(
        Completion(pj, "failed", "oom", [], infra=True, cuda_oom=True, peak_used_mb=15000)
    )
    assert sched.footprints.observed("ex.m") is None
    assert sched._eff(jt).vram_mb == 5000


def test_a_job_larger_than_the_idle_card_is_unschedulable_not_degraded(tmp_path):
    """The budget can exceed what the card really has free (another process, a desktop on
    WSL2). Such a job used to park, time out at the gate and mark the GPU degraded."""
    jt = _vram_jt("ex.big", vram=12000, co=False)
    g = gpu.MockGpu(total_mb=16384, used_mb=5000)  # 11384 free when idle
    sched = _sched(tmp_path, [jt], monitor=g)
    sched._used_baseline = 5000
    assert "free when idle" in sched._unschedulable(jt)
    pj = _pending(sched, jt, "j")
    assert sched.pick_next() is pj  # surfaced so admission fails it
    sched._admit_and_run(pj)
    assert pj.manifest.status == "failed" and "unschedulable" in pj.manifest.message
    assert not sched._gpu_degraded and not sched._pending_reclaim


def test_lower_priority_work_does_not_keep_a_busy_model_loaded(tmp_path):
    """A busy warm serve type kept taking lower-priority jobs, so a higher-priority job
    for another model waited behind all of them."""
    import types

    warm = _vram_jt("warm.m", vram=6000, co=False)
    warm = JobType(**{**warm.__dict__, "modes": ("serve",), "serve_socket": "/run/s.sock"})
    cold = _vram_jt("cold.m", vram=6000, co=False)
    sched = _sched(tmp_path, [warm, cold])
    sched.ledger.admit_budget_mb = 10000
    sched.ledger.admit_vram(sched._eff(warm), now=1.0)  # loaded and busy
    sched._serve[warm.id] = types.SimpleNamespace(dying=False)
    _pending(sched, warm, "w1", priority=0)
    pj_cold = _pending(sched, cold, "c1", priority=5)
    assert sched.pick_next() is None  # w1 waits; c1 cannot fit until warm.m is idle
    sched.ledger.finish_serve_job(warm.id, now=2.0, keep_warm_s=100)  # now idle
    assert sched.pick_next() is pj_cold  # evicts the idle model


def test_startup_reports_a_budget_larger_than_the_card(tmp_path):
    import json

    from gpuworker.obs import JobLogger

    log = tmp_path / "events.jsonl"
    cfg = Config(
        queue_roots=[],
        gpu_budget_mb=24000,
        state_dir=str(tmp_path / "state"),
        footprints_path=str(tmp_path / "fp.json"),
    )
    logger = JobLogger.to_path(str(log))
    sched = Scheduler(
        cfg,
        Registry(),
        make_launcher("local"),
        gpu.MockGpu(total_mb=16384),
        logger=logger,
        queues=[],
    )
    sched.stop()
    sched.run(interval_s=0.01)
    logger.close()
    events = [json.loads(line) for line in log.read_text().splitlines()]
    (e,) = [e for e in events if e["event"] == "gpu_budget_exceeds_card"]
    assert e["card_total_mb"] == 16384 and e["gpu_budget_mb"] == 24000
