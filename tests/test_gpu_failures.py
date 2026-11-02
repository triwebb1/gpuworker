"""GPU-side failures the broker exists to handle: a CUDA out-of-memory error, a CUDA error
that leaves a serve process unusable, and a growing model whose peak arrives on a later
job. Also the per-output size cap. All without a GPU: the workers report through the
same exit code and serve error kinds a real worker uses (MODULE_GUIDE 3.6, 3.2)."""

import json

import pytest

from gpuworker import bundle, gpu, serve
from gpuworker.client import submit, wait_result
from gpuworker.launcher import LocalLauncher
from gpuworker.manifest import JobManifest
from gpuworker.obs import JobLogger
from gpuworker.oneshot import CUDA_ERROR_RETURNCODE, CUDA_OOM_RETURNCODE
from gpuworker.scheduler import Scheduler
from gpuworker.serve_manager import ServeContainer

from helpers import (
    FakeServe,
    cpu_jt,
    drive,
    fake_connect,
    make_config,
    make_scheduler,
    registry,
    serve_jt,
)


def _events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _serve_sched(tmp_path, jt, launcher, monitor=None, logger=None, **cfg):
    cfg = make_config(tmp_path, container_runtime="docker", **cfg)
    cfg.serve_connect_timeout_s = cfg.serve_ready_timeout_s = 5.0
    return Scheduler(cfg, registry(jt), launcher, monitor or gpu.MockGpu(), logger=logger)


def _job(job_id, jt, text="x"):
    return JobManifest.from_dict(
        {
            "job_id": job_id,
            "job_type": jt.id,
            "outputs": [f"out/{job_id}.txt"],
            "options": {"text": text},
        }
    )


def test_a_cuda_oom_exit_is_retried_as_infrastructure_and_logged(tmp_path):
    jt = cpu_jt(
        code=f"import sys; print('CUDA out of memory', file=sys.stderr); "
        f"sys.exit({CUDA_OOM_RETURNCODE})"
    )
    log = tmp_path / "events.jsonl"
    logger = JobLogger.to_path(str(log))
    cfg = make_config(tmp_path, max_retries=1)
    sched = Scheduler(cfg, registry(jt), LocalLauncher(), gpu.MockGpu(), logger=logger)
    qr = str(tmp_path / "Q")
    fn = submit(qr, [JobManifest.from_dict({"job_id": "big", "job_type": jt.id})])
    assert drive(sched, qr, fn, ticks=2000)
    logger.close()
    (res,) = wait_result(qr, fn, timeout_s=5)
    assert res.status == "failed"
    assert "ran out of CUDA memory" in res.message and "quarantined" in res.message
    names = [e["event"] for e in _events(log)]
    assert names.count("cuda_oom") == 2 and names.count("retry") == 1


def test_a_oneshot_cuda_error_exit_is_retried_as_infrastructure(tmp_path):
    """A serve worker's CUDA error was retried but a oneshot worker's was final: both
    modules exited 1 on it."""
    marker = tmp_path / "ran-once"
    jt = cpu_jt(
        code=f"import os, sys\np = {str(marker)!r}\n"
        f"if not os.path.exists(p): open(p, 'w').close(); sys.exit({CUDA_ERROR_RETURNCODE})"
    )
    sched = Scheduler(
        make_config(tmp_path, max_retries=1), registry(jt), LocalLauncher(), gpu.MockGpu()
    )
    qr = str(tmp_path / "Q")
    fn = submit(qr, [JobManifest.from_dict({"job_id": "e", "job_type": jt.id})])
    assert drive(sched, qr, fn, ticks=2000)
    (res,) = wait_result(qr, fn, timeout_s=5)
    assert res.status == "done"  # the retry, in a fresh process, succeeded


def test_a_serve_cuda_oom_keeps_the_container_and_retries_on_it(tmp_path, monkeypatch):
    """A worker that ran out of memory frees the job's memory and stays up, so the retry
    runs on the same warm container. Tearing it down cost a full model load."""
    launcher = FakeServe(error_once={"big": serve.ERROR_CUDA_OOM})
    monkeypatch.setattr(serve, "connect", fake_connect(launcher))
    jt = serve_jt()
    log = tmp_path / "events.jsonl"
    logger = JobLogger.to_path(str(log))
    g = gpu.MockGpu(used_mb=800)
    sched = _serve_sched(tmp_path, jt, launcher, monitor=g, logger=logger, max_retries=1)
    sched.tick()  # the idle baseline: 800
    g.used = 800 + 3000  # memory the ledger does not know about, so a retry may fit
    qr = str(tmp_path / "Q")
    fn = submit(qr, [_job("big", jt, "BIG"), _job("next", jt)])
    assert drive(sched, qr, fn, ticks=2000)
    logger.close()
    res = {m.job_id: m for m in wait_result(qr, fn, timeout_s=5)}
    assert res["big"].status == "done", res["big"].message  # the retry ran
    assert res["next"].status == "done"
    assert launcher.runs == 1  # no reload
    assert [e["event"] for e in _events(log)].count("cuda_oom") == 1


def test_a_cuda_oom_with_the_gpu_to_itself_is_not_retried(tmp_path, monkeypatch):
    """A retry meets the same free memory, so an OOM the job had the card to itself for
    is the job's own: retrying it spent a retry and failed the same way."""
    launcher = FakeServe(error_once={"big": serve.ERROR_CUDA_OOM})
    monkeypatch.setattr(serve, "connect", fake_connect(launcher))
    jt = serve_jt()
    log = tmp_path / "events.jsonl"
    logger = JobLogger.to_path(str(log))
    sched = _serve_sched(tmp_path, jt, launcher, logger=logger, max_retries=1)
    qr = str(tmp_path / "Q")
    fn = submit(qr, [_job("big", jt, "BIG")])
    assert drive(sched, qr, fn, ticks=2000)
    logger.close()
    (res,) = wait_result(qr, fn, timeout_s=5)
    assert res.status == "failed" and "not retried: the job had the GPU to itself" in res.message
    events = _events(log)
    assert [e["retried"] for e in events if e["event"] == "cuda_oom"] == [False]
    assert not any(e["event"] == "retry" for e in events)


def _cuda_error_once(tmp_path):
    marker = tmp_path / "ran-once"
    return cpu_jt(
        code=f"import os, sys\np = {str(marker)!r}\n"
        f"if not os.path.exists(p): open(p, 'w').close(); sys.exit({CUDA_ERROR_RETURNCODE})"
    )


def test_a_cuda_error_with_a_hardware_fault_degrades_the_gpu_and_is_not_retried(tmp_path):
    jt = _cuda_error_once(tmp_path)
    g = gpu.MockGpu()
    g.faults = ["Xid 79"]
    sched = Scheduler(make_config(tmp_path, max_retries=1), registry(jt), LocalLauncher(), g)
    qr = str(tmp_path / "Q")
    fn = submit(qr, [JobManifest.from_dict({"job_id": "e", "job_type": jt.id})])
    assert drive(sched, qr, fn, ticks=2000)
    (res,) = wait_result(qr, fn, timeout_s=5)
    assert res.status == "failed" and "not retried: GPU fault: Xid 79" in res.message
    assert sched._gpu_degraded and sched._gpu_fault == "Xid 79"
    g.faults = []
    sched.tick()
    assert sched._gpu_degraded  # free memory does not clear a hardware fault


def test_a_gpu_that_stops_answering_after_a_failure_is_degraded_not_faulted(tmp_path):
    """Any monitor error after a failed job latched gpu_fault until a restart, so one slow
    nvidia-smi after a hung job took the card out of service. It now holds vram admission
    only until the GPU answers again."""
    jt = _cuda_error_once(tmp_path)
    g = gpu.MockGpu()
    g.unresponsive = True
    sched = Scheduler(make_config(tmp_path, max_retries=1), registry(jt), LocalLauncher(), g)
    qr = str(tmp_path / "Q")
    fn = submit(qr, [JobManifest.from_dict({"job_id": "e", "job_type": jt.id})])
    assert drive(sched, qr, fn, ticks=2000)
    (res,) = wait_result(qr, fn, timeout_s=5)
    assert res.status == "done", res.message  # retried: a cpu job does not wait on the GPU
    assert sched._gpu_degraded and not sched._gpu_fault
    g.unresponsive = False
    sched.tick()
    assert not sched._gpu_degraded


def test_one_monitor_error_after_a_failure_is_asked_again(tmp_path):
    jt = _cuda_error_once(tmp_path)
    g = gpu.MockGpu()
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError("nvidia-smi took too long")
        return []

    g.hardware_faults = flaky
    sched = Scheduler(make_config(tmp_path, max_retries=1), registry(jt), LocalLauncher(), g)
    sched._next_health_poll = float("inf")  # only the check after the failure asks
    qr = str(tmp_path / "Q")
    fn = submit(qr, [JobManifest.from_dict({"job_id": "e", "job_type": jt.id})])
    assert drive(sched, qr, fn, ticks=2000)
    (res,) = wait_result(qr, fn, timeout_s=5)
    assert res.status == "done", res.message
    assert len(calls) >= 2
    assert not sched._gpu_degraded and not sched._gpu_fault


def test_a_monitor_that_reports_the_gpu_lost_faults_it(tmp_path, monkeypatch):
    import subprocess

    def lost(argv, **kw):
        raise subprocess.CalledProcessError(
            15, argv, output="Unable to determine the device handle for GPU0: GPU is lost."
        )

    monkeypatch.setattr(gpu.subprocess, "check_output", lost)
    m = gpu.NvidiaSmiMonitor()
    assert m.hardware_faults() == ["nvidia-smi reports the GPU is lost"]
    jt = _cuda_error_once(tmp_path)
    sched = Scheduler(make_config(tmp_path, max_retries=1), registry(jt), LocalLauncher(), m)
    qr = str(tmp_path / "Q")
    fn = submit(qr, [JobManifest.from_dict({"job_id": "e", "job_type": jt.id})])
    assert drive(sched, qr, fn, ticks=2000)
    (res,) = wait_result(qr, fn, timeout_s=5)
    assert res.status == "failed" and "GPU fault: nvidia-smi reports the GPU is lost" in res.message
    assert sched._gpu_fault and sched._gpu_degraded


def test_nvml_monitor_reports_hardware_faults(monkeypatch):
    """NvmlMonitor reads ECC, row-remap and Xid counters through pynvml; a fake module
    stands in for the driver."""
    import sys
    import types

    class Err(Exception):
        pass

    class NvmlError(Exception):
        def __init__(self, value):
            super().__init__(value)
            self.value = value

    state = {"ecc": 2, "rows": (0, 0, 0, 0), "xids": [13, 79]}
    fake = types.SimpleNamespace(
        NVML_MEMORY_ERROR_TYPE_UNCORRECTED=1,
        NVML_VOLATILE_ECC=0,
        nvmlEventTypeXidCriticalError=8,
        nvmlInit=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda i: "h",
        nvmlDeviceGetMemoryInfo=lambda h: types.SimpleNamespace(total=1, used=0, free=1),
        nvmlDeviceGetTotalEccErrors=lambda h, kind, counter: state["ecc"],
        nvmlDeviceGetRemappedRows=lambda h: state["rows"],
        nvmlEventSetCreate=lambda: "set",
        nvmlDeviceRegisterEvents=lambda h, kinds, events: None,
    )

    def wait(events, timeout_ms):
        if not state["xids"]:
            raise Err("timeout")
        return types.SimpleNamespace(eventData=state["xids"].pop(0))

    fake.nvmlEventSetWait = wait
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    m = gpu.NvmlMonitor()
    assert m.hardware_faults() == ["Xid 79"]  # Xid 13 is an application fault
    state["ecc"] = 5
    state["rows"] = (0, 1, 0, 1)
    assert m.hardware_faults() == ["3 new uncorrected ECC errors", "row remapping failed"]

    def gone(h):
        raise NvmlError(15)  # NVML_ERROR_GPU_IS_LOST

    fake.nvmlDeviceGetMemoryInfo = gone
    assert m.hardware_faults() == ["NVML reports the GPU is lost"]

    def busy(h):
        raise NvmlError(10)  # any other error: the GPU did not answer, not a fault

    fake.nvmlDeviceGetMemoryInfo = busy
    with pytest.raises(NvmlError):
        m.hardware_faults()


def test_a_serve_cuda_error_does_not_leave_the_container_warm(tmp_path, monkeypatch):
    # After a CUDA error the process's context is unusable; keeping it warm would fail every
    # later job as if each were at fault.
    launcher = FakeServe(error_once={"poisoned": serve.ERROR_CUDA})
    monkeypatch.setattr(serve, "connect", fake_connect(launcher))
    jt = serve_jt()
    sched = _serve_sched(tmp_path, jt, launcher, max_retries=0)
    qr = str(tmp_path / "Q")
    fn = submit(qr, [_job("poisoned", jt), _job("later", jt)])
    assert drive(sched, qr, fn, ticks=2000)
    res = {m.job_id: m for m in wait_result(qr, fn, timeout_s=5)}
    assert res["poisoned"].status == "failed" and "cuda_error" in res["poisoned"].message
    assert res["later"].status == "done", res["later"].message  # on a new container
    assert launcher.runs == 2


def test_serve_footprint_is_reconciled_after_every_job(tmp_path, monkeypatch):
    """A growing model's peak can come on any job. Reconciling only after the first one
    missed every later, larger peak, and the ledger kept counting the first size."""
    launcher = FakeServe()
    monkeypatch.setattr(serve, "connect", fake_connect(launcher))
    jt = serve_jt(vram=5000)
    g = gpu.MockGpu(total_mb=16384, used_mb=800)
    sched = _serve_sched(tmp_path, jt, launcher, monitor=g)
    qr = str(tmp_path / "Q")
    f1 = submit(qr, [_job("small", jt)])
    assert drive(sched, qr, f1)
    assert sched.footprints.observed(jt.id) is None  # nothing above the baseline yet
    g.used = 800 + 6000  # a longer input: the model grows past its declaration
    f2 = submit(qr, [_job("large", jt)])
    assert drive(sched, qr, f2)
    assert wait_result(qr, f2, timeout_s=5)[0].status == "done"
    assert sched.footprints.observed(jt.id) == 6000
    assert sched.ledger.residents[jt.id].jt.vram_mb == 6000  # the slot counts it too
    assert sched._serve[jt.id].respawn_free_target_mb == 6000 + sched.cfg.headroom_mb


# ---- output size cap ----------------------------------------------------------------


def test_an_output_over_the_cap_fails_the_job(tmp_path):
    code = (
        "import os; d = os.environ['GPUWORKER_OUT']; os.makedirs(os.path.join(d, 'out'));"
        "open(os.path.join(d, 'out', 'big.bin'), 'wb').write(b'x' * (2 * 1024 * 1024))"
    )
    jt = cpu_jt(code=code)
    sched = make_scheduler(tmp_path, registry(jt), max_output_mb=1)
    qr = str(tmp_path / "Q")
    m = JobManifest.from_dict({"job_id": "a", "job_type": jt.id, "outputs": ["out/big.bin"]})
    fn = submit(qr, [m])
    assert drive(sched, qr, fn)
    (res,) = wait_result(qr, fn, timeout_s=5)
    assert res.status == "failed" and "refused output" in res.message
    assert "larger than 1048576 bytes" in res.message


def test_copy_regular_leaves_no_partial_copy_over_the_cap(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "f.bin").write_bytes(b"y" * 5000)
    dest = tmp_path / "dest" / "f.bin"
    try:
        bundle.copy_regular(str(src), "f.bin", str(dest), max_bytes=4096)
    except bundle.OutputTooLarge:
        pass
    else:
        raise AssertionError("expected OutputTooLarge")
    assert not dest.exists()
    bundle.copy_regular(str(src), "f.bin", str(dest), max_bytes=5000)
    assert dest.read_bytes() == b"y" * 5000


def test_a_serve_log_slice_is_bounded(tmp_path):
    sc = ServeContainer(jt=serve_jt(), epoch=1, name="n", root=str(tmp_path), jobq=None, stop=None)
    with open(sc.log_path, "wb") as fh:
        fh.write(b"before\n" + b"H" * 100 + b"m" * 10_000 + b"T" * 100)
    got = sc.log_slice(7, head=100, tail=100)
    assert got.startswith(b"H" * 100) and got.endswith(b"T" * 100)
    assert b"[gpuworker: 10000 bytes of output omitted]" in got
    assert sc.log_slice(7) == b"H" * 100 + b"m" * 10_000 + b"T" * 100  # under the default cap


# ---- GPU health checks beyond a reported CUDA error ------------------------------------


@pytest.mark.parametrize(
    "code,timeout_s",
    [
        ("import sys; sys.exit(139)", 0),  # a container's SIGSEGV
        ("import time; time.sleep(60)", 1),  # killed at its timeout
    ],
)
def test_a_vram_job_that_dies_abnormally_gets_the_gpu_checked(tmp_path, code, timeout_s):
    """The fault check ran only after a worker reported a CUDA error, so a card that
    crashed or hung its job was handed the next one."""
    import sys

    from gpuworker.module import JobType

    jt = JobType(
        id="ex.vram",
        resource_class="vram",
        vram_mb=4000,
        env="host",
        entrypoint=[sys.executable, "-c", code],
        sandbox="none",
    )
    g = gpu.MockGpu()
    sched = Scheduler(make_config(tmp_path), registry(jt), LocalLauncher(), g, clock=lambda: 1000.0)
    sched.tick()  # the first periodic poll, before the card fails
    g.faults = ["Xid 79"]
    qr = str(tmp_path / "Q")
    m = {"job_id": "j", "job_type": jt.id, "timeout_s": timeout_s}
    fn = submit(qr, [JobManifest.from_dict(m)])
    assert drive(sched, qr, fn, ticks=2000)
    (res,) = wait_result(qr, fn, timeout_s=5)
    assert res.status == "failed" and "GPU fault: Xid 79" in res.message
    assert sched._gpu_fault == "Xid 79" and sched._gpu_degraded


def test_the_loop_polls_gpu_health_while_idle(tmp_path):
    from gpuworker.scheduler import HEALTH_POLL_S

    now = [1000.0]
    g = gpu.MockGpu()
    sched = make_scheduler(tmp_path, registry(cpu_jt()), monitor=g)
    sched.clock = lambda: now[0]
    sched.tick()
    g.faults = ["a row remap is pending a GPU reset"]
    sched.tick()
    assert not sched._gpu_fault  # not due yet
    now[0] += HEALTH_POLL_S
    sched.tick()
    assert sched._gpu_fault and sched._gpu_degraded


def test_a_poll_the_gpu_does_not_answer_degrades_it_only_until_it_answers(tmp_path):
    from gpuworker.scheduler import HEALTH_POLL_S

    now = [1000.0]
    g = gpu.MockGpu()
    sched = make_scheduler(tmp_path, registry(cpu_jt()), monitor=g)
    sched.clock = lambda: now[0]
    sched.tick()
    g.unresponsive = True
    now[0] += HEALTH_POLL_S
    sched.tick()
    assert sched._gpu_degraded and not sched._gpu_fault
    g.unresponsive = False
    sched.tick()
    assert not sched._gpu_degraded


def test_nvidia_smi_reports_the_fault_counters_it_has(monkeypatch):
    """The nvidia-smi fallback only checked that the GPU answered."""
    import subprocess

    values = {
        "memory.used": "900",
        "ecc.errors.uncorrected.volatile.total": "2",
        "remapped_rows.failure": "No",
        "remapped_rows.pending": "Yes",
    }

    def check_output(argv, **kw):
        field = argv[1].split("=", 1)[1]
        if field not in values:
            raise subprocess.CalledProcessError(2, argv)
        if values[field] == "slow":
            raise subprocess.TimeoutExpired(argv, 10)
        return values[field] + "\n"

    monkeypatch.setattr(gpu.subprocess, "check_output", check_output)
    m = gpu.NvidiaSmiMonitor()
    assert m.hardware_faults() == ["a row remap is pending a GPU reset"]
    values["ecc.errors.uncorrected.volatile.total"] = "5"
    values["remapped_rows.pending"] = "No"
    assert m.hardware_faults() == ["3 new uncorrected ECC errors"]
    values["ecc.errors.uncorrected.volatile.total"] = "[N/A]"
    del values["remapped_rows.failure"], values["remapped_rows.pending"]
    assert m.hardware_faults() == []
    # A slow optional counter is not a GPU that stopped answering.
    values["ecc.errors.uncorrected.volatile.total"] = "slow"
    assert m.hardware_faults() == []
    values["memory.used"] = "slow"
    with pytest.raises(subprocess.TimeoutExpired):
        m.hardware_faults()


class _Killed:
    """A container runtime stand-in: runs each job as a host process that exits 137, and
    answers the OOMKilled question with *oom*."""

    def __init__(self, oom):
        self.oom = oom
        self.asked = []

    def run(self, spec):
        import subprocess
        import sys

        return subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.exit(137)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def terminate(self, proc, spec):
        proc.kill()

    def terminate_by_name(self, name):
        pass

    def container_exists(self, name):
        return False

    def reap_serve_orphans(self, prefix=""):
        pass

    def reap_jobs(self, prefix="", exited_only=True):
        pass

    def cleanup(self, spec):
        pass

    def oom_killed(self, name):
        self.asked.append(name)
        return self.oom


@pytest.mark.parametrize(
    "oom,says",
    [(True, "OOM-killed"), (False, "the runtime recorded no OOM-kill"), (None, "likely OOM-kill")],
)
def test_an_oom_kill_is_named_from_the_runtimes_flag(tmp_path, oom, says):
    from gpuworker.module import JobType

    jt = JobType(
        id="ex.box",
        resource_class="cpu",
        vram_mb=0,
        env="container",
        entrypoint=["worker"],
        sandbox="standard",
    )
    launcher = _Killed(oom)
    sched = make_scheduler(
        tmp_path, registry(jt), launcher=launcher, max_retries=0, container_runtime="docker"
    )
    qr = str(tmp_path / "Q")
    fn = submit(qr, [JobManifest.from_dict({"job_id": "k", "job_type": jt.id})])
    assert drive(sched, qr, fn, ticks=2000)
    (res,) = wait_result(qr, fn, timeout_s=5)
    assert res.status == "failed" and f"worker killed (exit 137, {says})" in res.message
    assert launcher.asked and launcher.asked[0].startswith("gpuworker-job-")
