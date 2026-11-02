"""Output delivery: the result bundle must carry the output bytes, and the
client collect() helper must extract them (and honor deliver_to)."""

import os
import time
import zipfile

from gpuworker import bundle, gpu
from gpuworker.client import collect, submit, wait_result
from gpuworker.launcher import make_launcher
from gpuworker.manifest import JobManifest
from gpuworker.module import scan_registry
from gpuworker.scheduler import Scheduler

from helpers import make_config


def _sched(tmp_path, queue_root):
    cfg = make_config(
        tmp_path,
        queue_roots=[queue_root],
        registry_dir="modules",
        container_runtime="local",
        weights_dir=str(tmp_path / "w"),
    )
    return Scheduler(cfg, scan_registry("modules"), make_launcher("local"), gpu.MockGpu())


def _run(sched, queue_root, fn, timeout=30):
    out = os.path.join(queue_root, "outbox", fn)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        sched.tick()
        if os.path.exists(out):
            sched.tick()
            return
        time.sleep(0.02)
    raise AssertionError("batch did not finish")


def test_result_bundle_carries_output_bytes(tmp_path):
    queue_root = str(tmp_path / "Echo")
    m = JobManifest.from_dict(
        {
            "job_id": "e1",
            "job_type": "echo.echo",
            "outputs": ["out/e1.txt"],
            "options": {"text": "PRODUCED"},
        }
    )
    fn = submit(queue_root, [m])
    _run(_sched(tmp_path, queue_root), queue_root, fn)
    out_zip = os.path.join(queue_root, "outbox", fn)
    with zipfile.ZipFile(out_zip) as zf:
        names = zf.namelist()
        assert "out/e1.txt" in names  # the file, not just manifest
        assert "jobs/pending/e1.json" in names
        assert zf.read("out/e1.txt").decode().strip() == "PRODUCED"


def test_collect_extracts_outputs(tmp_path):
    queue_root = str(tmp_path / "Echo2")
    m = JobManifest.from_dict(
        {
            "job_id": "c1",
            "job_type": "echo.echo",
            "outputs": ["out/c1.txt"],
            "options": {"text": "hi"},
        }
    )
    fn = submit(queue_root, [m])
    _run(_sched(tmp_path, queue_root), queue_root, fn)
    dest = tmp_path / "collected"
    mans = collect(queue_root, fn, str(dest), timeout_s=5)
    assert mans[0].status == "done"
    assert (dest / "out" / "c1.txt").read_text().strip() == "hi"


def test_deliver_to_copies_primary_output_idempotently(tmp_path):
    queue_root = str(tmp_path / "Echo3")
    target = tmp_path / "external" / "delivered.txt"
    m = JobManifest.from_dict(
        {
            "job_id": "d1",
            "job_type": "echo.echo",
            "outputs": ["out/d1.txt"],
            "options": {"text": "ship"},
            "deliver_to": str(target),
        }
    )
    fn = submit(queue_root, [m])
    _run(_sched(tmp_path, queue_root), queue_root, fn)
    dest = tmp_path / "collected3"
    mans = collect(queue_root, fn, str(dest), timeout_s=5)
    assert target.read_text().strip() == "ship"
    assert mans[0].result["delivered"] == str(target)
    # idempotent: a second collect lands the same bytes, no error
    mans2 = collect(queue_root, fn, str(tmp_path / "collected3b"), timeout_s=5)
    assert target.read_text().strip() == "ship"
    assert mans2[0].result["delivered"] == str(target)


def test_multiple_outputs_all_packed(tmp_path):
    # Echo writes the same text to every declared output; all must land in the bundle.
    queue_root = str(tmp_path / "Echo4")
    m = JobManifest.from_dict(
        {
            "job_id": "m1",
            "job_type": "echo.echo",
            "outputs": ["out/a.txt", "out/b.json", "out/c.json"],
        }
    )
    fn = submit(queue_root, [m])
    _run(_sched(tmp_path, queue_root), queue_root, fn)
    with zipfile.ZipFile(os.path.join(queue_root, "outbox", fn)) as zf:
        names = zf.namelist()
    assert {"out/a.txt", "out/b.json", "out/c.json"} <= set(names)


def test_pack_failure_degrades_to_manifest_only(tmp_path):
    # If packing the output bytes raises, the client must still get terminal status
    # and the batch must be archived (not stranded for stale-reclaim re-run). The job is
    # reported failed: its outputs never arrived, so `done` would be false.
    queue_root = str(tmp_path / "Echo5")
    m = JobManifest.from_dict({"job_id": "p1", "job_type": "echo.echo", "outputs": ["out/p1.txt"]})
    fn = submit(queue_root, [m])
    sched = _sched(tmp_path, queue_root)
    q = sched.queues[0]
    orig = q.write_result

    def flaky(filename, results, files_root=None, files=None):
        if files:  # fail only the bytes-carrying call
            raise OSError("simulated share failure")
        return orig(filename, results)  # manifest-only fallback succeeds

    q.write_result = flaky
    _run(sched, queue_root, fn)
    # client still unblocks with terminal status (degraded: no bytes)
    mans = wait_result(queue_root, fn, timeout_s=5)
    assert mans[0].status == "failed"
    assert "could not be delivered" in mans[0].message
    # processing file archived, not left for stale-reclaim
    assert not os.path.exists(os.path.join(queue_root, "processing", fn))
    assert os.path.exists(os.path.join(queue_root, "archive", fn))


def test_unwritable_result_keeps_the_batch_and_retries(tmp_path):
    # If no result at all can be written (share gone), nothing is archived or deleted:
    # the batch keeps its processing/ file and outputs, and delivery is retried.
    queue_root = str(tmp_path / "Echo6")
    m = JobManifest.from_dict(
        {
            "job_id": "k1",
            "job_type": "echo.echo",
            "outputs": ["out/k1.txt"],
            "options": {"text": "kept"},
        }
    )
    fn = submit(queue_root, [m])
    sched = _sched(tmp_path, queue_root)
    clk = [1000.0]
    sched.clock = lambda: clk[0]
    q = sched.queues[0]
    orig = q.write_result
    share_up = [False]

    def flaky(filename, results, files_root=None, files=None):
        if not share_up[0]:
            raise OSError("simulated share outage")
        return orig(filename, results, files_root=files_root, files=files)

    q.write_result = flaky
    for _ in range(300):
        sched.tick()
        if sched._undelivered:
            break
        time.sleep(0.01)
    assert sched._undelivered
    assert os.path.exists(os.path.join(queue_root, "processing", fn))  # kept, not archived
    assert not os.path.exists(os.path.join(queue_root, "outbox", fn))
    assert (q.root, fn) in sched._active_bundles  # stale reclaim keeps skipping it
    share_up[0] = True
    sched.tick()  # not due yet
    assert sched._undelivered
    clk[0] += 60
    sched.tick()
    assert not sched._undelivered
    with zipfile.ZipFile(os.path.join(queue_root, "outbox", fn)) as zf:
        assert zf.read("out/k1.txt").decode().strip() == "kept"
    assert os.path.exists(os.path.join(queue_root, "archive", fn))


def test_extract_outputs_skips_manifests_and_guards_traversal(tmp_path):
    # extract_outputs only pulls non-manifest members and refuses traversal.
    zip_path = tmp_path / "r.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("jobs/pending/x.json", "{}")
        zf.writestr("out/ok.bin", b"data")
    extracted = bundle.extract_outputs(str(zip_path), str(tmp_path / "d"))
    assert extracted == ["out/ok.bin"]
    assert (tmp_path / "d" / "out" / "ok.bin").read_bytes() == b"data"
    assert not (tmp_path / "d" / "jobs").exists()

    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("../escape.bin", b"x")
    import pytest

    with pytest.raises(ValueError):
        bundle.extract_outputs(str(bad), str(tmp_path / "d2"))


def test_a_result_that_never_writes_is_given_up(tmp_path):
    """A result that cannot be written (a full disk, say) is retried a bounded number of
    times. Meanwhile the unpacked input is already gone; then the input moves to failed/
    with the reason and the batch's state is dropped."""
    import io
    import json

    from gpuworker import scheduler as scheduler_mod
    from gpuworker.obs import JobLogger

    queue_root = str(tmp_path / "Echo7")
    m = JobManifest.from_dict(
        {"job_id": "g1", "job_type": "echo.echo", "outputs": ["out/g1.txt"], "options": {}}
    )
    fn = submit(queue_root, [m])
    sched = _sched(tmp_path, queue_root)
    stream = io.StringIO()
    sched.log = JobLogger(stream)
    clk = [1000.0]
    sched.clock = lambda: clk[0]
    q = sched.queues[0]

    def full_disk(filename, results, files_root=None, files=None):
        raise OSError(28, "No space left on device")

    q.write_result = full_disk
    for _ in range(300):
        sched.tick()
        if sched._undelivered:
            break
        time.sleep(0.01)
    ((b, _),) = sched._undelivered
    assert not os.path.exists(b.work_dir) and not os.path.exists(b.run_dir)
    assert os.path.isdir(b.out_dir)  # the outputs wait for the next attempt
    for _ in range(scheduler_mod.RESULT_ATTEMPTS):
        clk[0] += scheduler_mod.RESULT_RETRY_S + 1
        sched.tick()
    assert not sched._undelivered and not sched.batches
    assert (q.root, fn) not in sched._active_bundles
    assert not os.path.exists(os.path.join(queue_root, "processing", fn))
    assert os.path.exists(os.path.join(queue_root, "failed", fn))
    with open(os.path.join(queue_root, "failed", fn + ".error.txt")) as fh:
        assert "no result could be written" in fh.read()
    assert not os.path.exists(b.root)
    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    (gave_up,) = [e for e in events if e["event"] == "result_abandoned"]
    assert gave_up["job_id"] == fn and str(scheduler_mod.RESULT_ATTEMPTS) in gave_up["message"]
