"""How the broker handles malformed bundles, mistaken paths and links a job leaves in
its output (SPEC 6.3).

Symlink tests need a host that lets this user create symlinks (Linux CI does; a default
Windows account does not), and skip otherwise.
"""

import json
import os
import subprocess
import sys
import zipfile

import pytest

from gpuworker import bundle
from gpuworker.client import submit, wait_result
from gpuworker.launcher import LocalLauncher
from gpuworker.manifest import JobManifest, ManifestError
from gpuworker.module import JobType
from gpuworker.serve_manager import ServeContainer

from helpers import (
    FakeServe,
    cpu_jt,
    drive,
    fake_connect,
    make_scheduler,
    registry,
    serve_jt,
)


def _can_symlink(tmp_path) -> bool:
    target = tmp_path / "symlink-probe-target"
    target.write_text("x")
    try:
        os.symlink(target, tmp_path / "symlink-probe")
    except (OSError, NotImplementedError):
        return False
    return True


@pytest.fixture
def symlinks(tmp_path):
    if not _can_symlink(tmp_path):
        pytest.skip("this host does not let the test user create symlinks")


def _result_members(qr, fn):
    with zipfile.ZipFile(os.path.join(qr, "outbox", fn)) as zf:
        return {n: zf.read(n) for n in zf.namelist()}


def _raw_bundle(qr, name, manifests):
    """Drop a bundle built from raw manifest dicts, bypassing client-side validation."""
    os.makedirs(os.path.join(qr, "inbox"), exist_ok=True)
    path = os.path.join(qr, "inbox", name)
    bundle.pack(path, manifests, root=qr)
    return name


# ---- reading job-written paths without following symlinks ---------------------------


def test_open_regular_reads_a_plain_file(tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "a.txt").write_bytes(b"ok")
    fd = bundle.open_regular(str(tmp_path), "out/a.txt")
    with os.fdopen(fd, "rb") as fh:
        assert fh.read() == b"ok"
    with pytest.raises(FileNotFoundError):
        bundle.open_regular(str(tmp_path), "out/missing.txt")
    with pytest.raises(ValueError):
        bundle.open_regular(str(tmp_path), "../escape.txt")
    with pytest.raises(bundle.UnsafeMember):
        bundle.open_regular(str(tmp_path), "out")  # a directory is not an output file


def test_open_regular_refuses_symlinks_anywhere_on_the_path(tmp_path, symlinks):
    host_file = tmp_path / "host-only"
    host_file.write_text("host data")
    hostdir = tmp_path / "hostdir"
    hostdir.mkdir()
    (hostdir / "f.txt").write_text("host data")
    root = tmp_path / "out"
    root.mkdir()
    os.symlink(host_file, root / "leaf.txt")
    os.symlink(hostdir, root / "linkdir")
    with pytest.raises(bundle.UnsafeMember):
        bundle.open_regular(str(root), "leaf.txt")
    with pytest.raises(bundle.UnsafeMember):
        bundle.open_regular(str(root), "linkdir/f.txt")


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs POSIX FIFOs")
def test_open_regular_refuses_a_fifo_without_blocking(tmp_path):
    os.mkfifo(tmp_path / "pipe")
    with pytest.raises(bundle.UnsafeMember):
        bundle.open_regular(str(tmp_path), "pipe")


def test_symlinked_output_fails_the_job_and_leaks_nothing(tmp_path, symlinks):
    host_file = tmp_path / "host-only.txt"
    host_file.write_text("HOST-ONLY")
    code = (
        "import os; o = os.path.join(os.environ['GPUWORKER_OUT'], 'out'); "
        f"os.makedirs(o); os.symlink({str(host_file)!r}, os.path.join(o, 'x.txt'))"
    )
    jt = cpu_jt(code=code)
    sched = make_scheduler(tmp_path, registry(jt))
    qr = str(tmp_path / "Q")
    fn = submit(
        qr, [JobManifest.from_dict({"job_id": "j", "job_type": jt.id, "outputs": ["out/x.txt"]})]
    )
    assert drive(sched, qr, fn)
    (res,) = wait_result(qr, fn, timeout_s=5)
    assert res.status == "failed" and "refused output" in res.message
    assert not any(b"HOST-ONLY" in data for data in _result_members(qr, fn).values())


def test_symlinked_output_dir_fails_the_job(tmp_path, symlinks):
    hostdir = tmp_path / "hostdir"
    hostdir.mkdir()
    (hostdir / "x.txt").write_text("HOST-ONLY")
    code = (
        f"import os; os.symlink({str(hostdir)!r}, os.path.join(os.environ['GPUWORKER_OUT'], 'out'))"
    )
    jt = cpu_jt(code=code)
    sched = make_scheduler(tmp_path, registry(jt))
    qr = str(tmp_path / "Q")
    fn = submit(
        qr, [JobManifest.from_dict({"job_id": "j", "job_type": jt.id, "outputs": ["out/x.txt"]})]
    )
    assert drive(sched, qr, fn)
    (res,) = wait_result(qr, fn, timeout_s=5)
    assert res.status == "failed" and "symbolic link" in res.message
    assert not any(b"HOST-ONLY" in data for data in _result_members(qr, fn).values())


def test_job_cannot_redirect_the_broker_worker_log(tmp_path, symlinks):
    # The worker log is written into the broker-only out dir, not the dir the job can
    # write, so a job that leaves `_gpuworker` as a link cannot point the write anywhere.
    host_file = tmp_path / "host_file"
    host_file.mkdir()
    code = (
        "import os; "
        f"os.symlink({str(host_file)!r}, os.path.join(os.environ['GPUWORKER_OUT'], '_gpuworker'))"
    )
    jt = cpu_jt(code=code)
    sched = make_scheduler(tmp_path, registry(jt))
    qr = str(tmp_path / "Q")
    fn = submit(qr, [JobManifest.from_dict({"job_id": "j", "job_type": jt.id})])
    assert drive(sched, qr, fn)
    assert os.listdir(host_file) == []
    assert "_gpuworker/j.log" in _result_members(qr, fn)


def test_serve_collect_refuses_symlinks(tmp_path, symlinks):
    host_file = tmp_path / "host-only.txt"
    host_file.write_text("HOST-ONLY")
    sc = ServeContainer(jt=None, epoch=1, name="n", root=str(tmp_path / "sc"), jobq=None, stop=None)
    b = type("B", (), {"out_dir": str(tmp_path / "batch-out")})()
    pj = type(
        "P",
        (),
        {
            "manifest": JobManifest.from_dict(
                {"job_id": "j", "job_type": "a.b", "outputs": ["out/x.txt"]}
            ),
            "batch": b,
            "uid": "u1",
            "attempts": 0,
        },
    )()
    key = ServeContainer.workspace(pj)  # the attempt's directory, never the job_id
    jdir = tmp_path / "sc" / "out" / key / "out"
    jdir.mkdir(parents=True)
    os.symlink(host_file, jdir / "x.txt")
    with pytest.raises(bundle.UnsafeMember):
        sc.collect_out(pj)
    # ... and the same when the container replaces its whole per-job dir with a link
    os.rename(tmp_path / "sc" / "out" / key, tmp_path / "sc" / "moved")
    os.symlink(tmp_path / "sc" / "moved", tmp_path / "sc" / "out" / key, target_is_directory=True)
    with pytest.raises(bundle.UnsafeMember):
        sc.collect_out(pj)


# ---- intake validation ------------------------------------------------------------


def test_malformed_submitted_time_fails_its_bundle_and_nothing_else(tmp_path):
    # A non-string submitted_utc used to reach the FIFO sort key, so every tick raised
    # comparing it with the other jobs' strings and no client was served.
    jt = cpu_jt(code="pass")
    sched = make_scheduler(tmp_path, registry(jt))
    qr = str(tmp_path / "Q")
    bad = _raw_bundle(qr, "bad.zip", [{"job_id": "b", "job_type": jt.id, "submitted_utc": 12345}])
    good = submit(qr, [JobManifest.from_dict({"job_id": "g", "job_type": jt.id})])
    assert drive(sched, qr, good)
    assert drive(sched, qr, bad)
    assert wait_result(qr, good, timeout_s=5)[0].status == "done"
    (res,) = wait_result(qr, bad, timeout_s=5)
    assert res.status == "failed" and "submitted_utc" in res.message


def test_broker_stamps_submitted_time_and_orders_by_intake(tmp_path):
    jt = cpu_jt(code="pass")
    sched = make_scheduler(tmp_path, registry(jt))
    qr = str(tmp_path / "Q")
    fn = submit(
        qr,
        [
            JobManifest.from_dict(
                {"job_id": "a", "job_type": jt.id, "submitted_utc": "1970-01-01T00:00:00+00:00"}
            )
        ],
    )
    assert drive(sched, qr, fn)
    (res,) = wait_result(qr, fn, timeout_s=5)
    assert res.submitted_utc != "1970-01-01T00:00:00+00:00"


def test_priority_is_bounded_and_capped_per_queue(tmp_path):
    with pytest.raises(ManifestError):
        JobManifest.from_dict({"job_id": "a", "job_type": "x.y", "priority": 10**9})
    jt = cpu_jt(code="pass")
    qr = str(tmp_path / "Q")
    sched = make_scheduler(tmp_path, registry(jt), max_priority={qr: 5})
    fn = submit(qr, [JobManifest.from_dict({"job_id": "a", "job_type": jt.id, "priority": 900})])
    assert drive(sched, qr, fn)
    assert wait_result(qr, fn, timeout_s=5)[0].priority == 5


@pytest.mark.parametrize(
    "outputs_a, outputs_b, why",
    [
        (["out/x.txt"], ["out/x.txt"], "overlaps"),
        (["out"], ["out/x.txt"], "overlaps"),
        (["_gpuworker/b.log"], [], "reserved"),
        (["jobs/pending/b.json"], [], "reserved"),
        (["_GPUWorker/b.log"], [], "reserved"),
        (["Jobs/pending/b.json"], [], "reserved"),
    ],
)
def test_output_paths_may_not_overlap_or_use_reserved_dirs(tmp_path, outputs_a, outputs_b, why):
    jt = cpu_jt(code="pass")
    sched = make_scheduler(tmp_path, registry(jt))
    qr = str(tmp_path / "Q")
    fn = submit(
        qr,
        [
            JobManifest.from_dict({"job_id": "a", "job_type": jt.id, "outputs": outputs_a}),
            JobManifest.from_dict({"job_id": "b", "job_type": jt.id, "outputs": outputs_b}),
        ],
    )
    assert drive(sched, qr, fn)
    res = wait_result(qr, fn, timeout_s=5)
    assert all(r.status == "failed" and why in r.message for r in res)


def test_bundle_limits_are_enforced_on_what_is_written(tmp_path):
    z = tmp_path / "b.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("jobs/pending/a.json", json.dumps({"job_id": "a", "job_type": "x.y"}))
        zf.writestr("big.bin", b"\0" * 300_000)  # compresses to almost nothing
    with pytest.raises(bundle.BundleTooLarge):
        bundle.unpack(str(z), str(tmp_path / "u1"), max_members=1)
    with pytest.raises(bundle.BundleTooLarge):
        bundle.unpack(str(z), str(tmp_path / "u2"), max_bytes=100_000)
    assert len(bundle.unpack(str(z), str(tmp_path / "u3"))) == 2


def test_oversized_manifest_member_is_not_parsed(tmp_path):
    z = tmp_path / "b.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("jobs/pending/a.json", " " * (bundle.MAX_MANIFEST_BYTES + 1) + "{}")
        zf.writestr("jobs/pending/b.json", json.dumps({"job_id": "b", "job_type": "x.y"}))
    assert [m["job_id"] for m in bundle.read_pending_manifests(str(z))] == ["b"]


# ---- job identity -------------------------------------------------------------------


class _HoldingLauncher:
    """Records each LaunchSpec and runs a short host sleep in place of a container."""

    def __init__(self):
        self.specs = []

    def run(self, spec):
        self.specs.append(spec)
        return subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(0.5)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def terminate(self, proc, spec):
        proc.kill()

    def terminate_by_name(self, name):
        pass

    def reap_serve_orphans(self, prefix=""):
        pass

    def reap_jobs(self, prefix="", exited_only=True):
        pass

    def cleanup(self, spec):
        pass


def test_same_job_id_in_two_bundles_is_two_jobs(tmp_path):
    jt = JobType(
        id="ex.box",
        resource_class="cpu",
        vram_mb=0,
        env="container",
        entrypoint=["worker"],
        sandbox="standard",
    )
    launcher = _HoldingLauncher()
    sched = make_scheduler(tmp_path, registry(jt), launcher=launcher)
    qr = str(tmp_path / "Q")
    # A job_id shaped like a serve container name must not produce one.
    f1 = submit(qr, [JobManifest.from_dict({"job_id": "serve-ex.box-1", "job_type": jt.id})])
    f2 = submit(qr, [JobManifest.from_dict({"job_id": "serve-ex.box-1", "job_type": jt.id})])
    sched.tick()
    assert len(sched._inflight) == 2  # both tracked, neither overwrote the other
    assert drive(sched, qr, f1) and drive(sched, qr, f2)
    names = [s.name for s in launcher.specs]
    assert len(set(names)) == 2
    assert all(n.startswith("gpuworker-job-") for n in names)
    assert all(wait_result(qr, f, timeout_s=5)[0].status == "done" for f in (f1, f2))


def _serve_sched(tmp_path, monkeypatch):
    from gpuworker import serve

    launcher = FakeServe()
    monkeypatch.setattr(serve, "connect", fake_connect(launcher))
    jt = serve_jt()
    sched = make_scheduler(tmp_path, registry(jt), launcher=launcher, container_runtime="docker")
    sched.cfg.serve_connect_timeout_s = sched.cfg.serve_ready_timeout_s = 5.0
    return sched, launcher, jt


def test_serve_jobs_with_the_same_job_id_get_separate_workspaces(tmp_path, monkeypatch):
    """Serve staging used to be keyed by the client's job_id and reused an existing
    directory, so a bundle could read or be satisfied by what an earlier bundle with the
    same job_id left behind."""
    from gpuworker import serve

    sched, launcher, jt = _serve_sched(tmp_path, monkeypatch)
    keys = []
    real_run_job = serve.ServeSession.run_job

    def run_job(self, job_id, manifest, timeout=None):
        keys.append(job_id)
        return real_run_job(self, job_id, manifest, timeout)

    monkeypatch.setattr(serve.ServeSession, "run_job", run_job)
    qr = str(tmp_path / "Q")
    fns = [
        submit(
            qr, [JobManifest.from_dict({"job_id": "x", "job_type": jt.id, "outputs": ["o.txt"]})]
        )
        for _ in range(2)
    ]
    for fn in fns:
        assert drive(sched, qr, fn)
        assert wait_result(qr, fn, timeout_s=5)[0].status == "done"
    assert len(set(keys)) == 2 and "x" not in keys
    assert launcher.runs == 1  # the same warm container served both


def test_serve_stage_in_refuses_a_leftover_workspace(tmp_path):
    import queue
    import threading

    from gpuworker.jobs import Batch, PendingJob
    from gpuworker.share import QueueTree

    sc = ServeContainer(
        jt=serve_jt(),
        epoch=1,
        name="n",
        root=str(tmp_path / "sc"),
        jobq=queue.Queue(),
        stop=threading.Event(),
    )
    work = tmp_path / "work"
    work.mkdir()
    batch = Batch(
        queue=QueueTree(str(tmp_path / "Q")),
        filename="b.zip",
        work_dir=str(work),
        out_dir=str(tmp_path / "out"),
        run_dir=str(tmp_path / "run"),
    )
    pj = PendingJob(
        manifest=JobManifest.from_dict({"job_id": "x", "job_type": "ex.serve"}),
        jt=sc.jt,
        batch=batch,
        seq=1,
    )
    os.makedirs(os.path.join(sc.bundle_root, sc.workspace(pj)))
    with pytest.raises(FileExistsError):
        sc.stage_in(pj)


def test_a_failed_workspace_cleanup_retires_the_serve_container(tmp_path, monkeypatch):
    sched, launcher, jt = _serve_sched(tmp_path, monkeypatch)
    monkeypatch.setattr(ServeContainer, "cleanup_job", lambda self, key: False)
    qr = str(tmp_path / "Q")
    first = submit(
        qr, [JobManifest.from_dict({"job_id": "a", "job_type": jt.id, "outputs": ["o.txt"]})]
    )
    assert drive(sched, qr, first)
    for _ in range(100):
        sched.tick()
        if not sched._serve:
            break
    assert not sched._serve  # not kept warm beside a workspace it could not remove
    second = submit(
        qr, [JobManifest.from_dict({"job_id": "a", "job_type": jt.id, "outputs": ["o.txt"]})]
    )
    assert drive(sched, qr, second)
    assert launcher.runs == 2
    assert all(wait_result(qr, f, timeout_s=5)[0].status == "done" for f in (first, second))


def test_a_serve_job_missing_one_declared_output_fails(tmp_path, monkeypatch):
    sched, launcher, jt = _serve_sched(tmp_path, monkeypatch)
    real = ServeContainer.collect_out
    monkeypatch.setattr(
        ServeContainer, "collect_out", lambda self, pj, mb=None: real(self, pj, mb)[:1]
    )
    qr = str(tmp_path / "Q")
    m = {"job_id": "a", "job_type": jt.id, "outputs": ["one.txt", "two.txt"]}
    fn = submit(qr, [JobManifest.from_dict(m)])
    assert drive(sched, qr, fn)
    (res,) = wait_result(qr, fn, timeout_s=5)
    assert res.status == "failed" and "'two.txt'" in res.message


# ---- retries see only their own attempt's files -----------------------------------------


def test_a_retry_cannot_succeed_on_a_dead_attempts_output(tmp_path):
    marker = tmp_path / "first-attempt-ran"
    code = (
        "import os, sys, pathlib\n"
        f"marker = pathlib.Path({str(marker)!r})\n"
        "out = pathlib.Path(os.environ['GPUWORKER_OUT']) / 'out' / 'r.txt'\n"
        "if not marker.exists():\n"
        "    marker.write_text('1')\n"
        "    out.parent.mkdir(parents=True, exist_ok=True)\n"
        "    out.write_text('stale')\n"
        "    sys.exit(137)  # looks like an OOM-kill: an infrastructure failure, retried\n"
        "sys.exit(0)  # the retry produces nothing\n"
    )
    jt = cpu_jt(code=code)
    sched = make_scheduler(tmp_path, registry(jt))
    qr = str(tmp_path / "Q")
    fn = submit(
        qr, [JobManifest.from_dict({"job_id": "r", "job_type": jt.id, "outputs": ["out/r.txt"]})]
    )
    assert drive(sched, qr, fn)
    (res,) = wait_result(qr, fn, timeout_s=5)
    assert marker.exists()
    assert res.status == "failed" and "did not produce" in res.message
    assert "out/r.txt" not in _result_members(qr, fn)


# ---- container types under the local launcher -----------------------------------------


def _container_jt(**kw):
    base = dict(
        id="ex.model",
        resource_class="vram",
        vram_mb=100,
        env="container",
        entrypoint=[sys.executable, "-c", "pass"],
        sandbox="standard",
    )
    base.update(kw)
    return JobType(**base)


@pytest.mark.parametrize(
    "jt",
    [
        _container_jt(),
        _container_jt(resource_class="cpu", vram_mb=0, sandbox="strict"),
        _container_jt(resource_class="cpu", vram_mb=0, sandbox="standard"),
        _container_jt(resource_class="cpu", vram_mb=0, sandbox="none"),
    ],
)
def test_local_runtime_refuses_every_container_type(tmp_path, jt):
    # Under "local" a container type would run as a host process with none of the
    # controls its module declared, whatever its lane or sandbox level.
    sched = make_scheduler(tmp_path, registry(jt), launcher=LocalLauncher())
    qr = str(tmp_path / "Q")
    fn = submit(qr, [JobManifest.from_dict({"job_id": "a", "job_type": jt.id})])
    assert drive(sched, qr, fn)
    (res,) = wait_result(qr, fn, timeout_s=5)
    assert res.status == "failed" and "no isolation" in res.message


def test_local_runtime_runs_them_only_with_the_explicit_opt_in(tmp_path):
    jt = _container_jt()
    sched = make_scheduler(
        tmp_path, registry(jt), launcher=LocalLauncher(), allow_unsandboxed_containers=True
    )
    qr = str(tmp_path / "Q")
    fn = submit(qr, [JobManifest.from_dict({"job_id": "a", "job_type": jt.id})])
    assert drive(sched, qr, fn)
    assert wait_result(qr, fn, timeout_s=5)[0].status == "done"


def test_local_runtime_still_runs_host_types(tmp_path):
    jt = _container_jt(resource_class="cpu", vram_mb=0, sandbox="none", env="host")
    sched = make_scheduler(tmp_path, registry(jt), launcher=LocalLauncher())
    qr = str(tmp_path / "Q")
    fn = submit(qr, [JobManifest.from_dict({"job_id": "a", "job_type": jt.id})])
    assert drive(sched, qr, fn)
    assert wait_result(qr, fn, timeout_s=5)[0].status == "done"
