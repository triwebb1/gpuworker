import json

from gpuworker import gpu
from gpuworker.config import Config
from gpuworker.launcher import make_launcher
from gpuworker.module import JobType, Registry
from gpuworker.obs import write_status
from gpuworker.scheduler import Scheduler


def _sched(tmp_path):
    jt = JobType(
        id="ex.m",
        resource_class="vram",
        vram_mb=5000,
        env="container",
        entrypoint=["x"],
        image="i@sha256:" + "a" * 64,
        vram_profile="growing",
        sandbox="standard",
    )
    reg = Registry()
    reg.job_types[jt.id] = jt
    cfg = Config(queue_roots=[], footprints_path=str(tmp_path / "fp.json"))
    return Scheduler(
        cfg, reg, make_launcher("local"), gpu.MockGpu(total_mb=16384, used_mb=800), queues=[]
    ), jt


def test_status_reports_gpu_and_modules(tmp_path):
    sched, jt = _sched(tmp_path)
    out = str(tmp_path / "status.json")
    write_status(out, sched)
    snap = json.loads(open(out, encoding="utf-8").read())
    assert snap["gpu"] == {"total_mb": 16384, "used_mb": 800, "free_mb": 15584}
    assert snap["ledger"]["admit_budget_mb"] == sched.cfg.admit_budget_mb()
    m = {x["job_type"]: x for x in snap["modules"]}["ex.m"]
    assert m["declared_mb"] == 5000
    assert m["observed_mb"] is None  # nothing observed yet
    assert m["effective_mb"] == 5000
    assert m["vram_profile"] == "growing" and m["packable"] is False


def test_status_reflects_observed_footprint(tmp_path):
    sched, jt = _sched(tmp_path)
    # An observation under an OLD image no longer applies, exactly as for admission.
    sched.footprints.record("ex.m", 9900, image="old@sha256:" + "b" * 64)
    sched.footprints.record("ex.m", 7200, image=sched._image_sig(jt))  # reality exceeded spec
    out = str(tmp_path / "status.json")
    write_status(out, sched)
    snap = json.loads(open(out, encoding="utf-8").read())
    m = {x["job_type"]: x for x in snap["modules"]}["ex.m"]
    assert m["observed_mb"] == 7200
    assert m["effective_mb"] == 7200 == sched._eff(jt).vram_mb  # what admission uses
