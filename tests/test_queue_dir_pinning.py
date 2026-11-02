"""The directories of a queue tree can be renamed or relinked while the broker runs, so
it pins each one when it starts and never follows a directory replaced by a link
afterwards (SPEC 6.3). Directory links are symlinks on POSIX and junctions on Windows,
which any user can create. On Linux and macOS the broker keeps working through the
directory it pinned; on Windows it refuses to touch a replaced directory.
"""

import os
import threading
import time
import zipfile

import pytest

from gpuworker import bundle
from gpuworker.client import submit, wait_result
from gpuworker.manifest import JobManifest, ManifestError
from gpuworker.share import QueueTree

from helpers import cpu_jt, drive, make_scheduler, registry


def _dir_link(target, link):
    if os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
        return
    os.symlink(target, link, target_is_directory=True)


def _swap_for_link(tree, sub, target):
    """Move *sub* aside and put a link to *target* in its place. Returns the moved dir."""
    moved = os.path.join(tree, sub + ".moved")
    os.rename(os.path.join(tree, sub), moved)
    _dir_link(target, os.path.join(tree, sub))
    return moved


def _job(job_id="a", **kw):
    return JobManifest.from_dict({"job_id": job_id, "job_type": "ex.cpu", **kw})


def _two_clients(tmp_path):
    a, b = str(tmp_path / "A"), str(tmp_path / "B")
    sched = make_scheduler(tmp_path, registry(cpu_jt()), queue_roots=[a, b])
    for q in sched.queues:
        q.ensure()  # what run() does at startup
    return sched, a, b


def _run(sched, ticks=200, until=None):
    """Tick, with a short sleep so real worker processes can finish, until *until()*."""
    for _ in range(ticks):
        sched.tick()
        if until is not None and until():
            sched.tick()
            return
        time.sleep(0.005)


def _files_under(root):
    return [f for _, _, files in os.walk(root) for f in files]


def test_archive_swapped_for_another_clients_inbox_is_not_followed(tmp_path):
    sched, a, b = _two_clients(tmp_path)
    fn = submit(a, [_job()])
    sched.tick()  # claimed into A's processing/
    moved = _swap_for_link(a, "archive", os.path.join(b, "inbox"))
    assert drive(sched, a, fn)
    _run(sched, 20)
    assert _files_under(b) == []  # never reached B, so never ran as B's work
    if os.name != "nt":
        assert os.listdir(moved) == [fn]  # archived into the directory that was pinned
    assert wait_result(a, fn, timeout_s=5)[0].status == "done"


def test_processing_swapped_after_startup_is_not_followed(tmp_path):
    sched, a, b = _two_clients(tmp_path)
    fn = submit(a, [_job()])
    moved = _swap_for_link(a, "processing", os.path.join(b, "inbox"))
    _run(sched, 400, until=lambda: os.path.exists(os.path.join(a, "outbox", fn)))
    assert _files_under(b) == []
    if os.name != "nt":
        assert os.path.exists(os.path.join(a, "outbox", fn))
        assert not os.listdir(moved)  # claimed, run and archived through the pinned dir
    else:
        assert os.listdir(os.path.join(a, "inbox")) == [fn]  # refused, left where it was


def test_outbox_swapped_for_another_clients_inbox_is_not_followed(tmp_path):
    sched, a, b = _two_clients(tmp_path)
    fn = submit(a, [_job()])
    sched.tick()
    moved = _swap_for_link(a, "outbox", os.path.join(b, "inbox"))
    _run(sched, 400, until=lambda: os.path.exists(os.path.join(moved, fn)))
    assert _files_under(b) == []
    if os.name != "nt":
        assert os.listdir(moved) == [fn]


# ---- a backslash in a bundle path ---------------------------------------------------


@pytest.mark.parametrize(
    "path", [r"jobs\pending\x.json", r"out\x.wav", "../x", "a//b", "./a", "/abs", "C:x"]
)
def test_outputs_and_inputs_must_be_plain_relative_paths(path):
    for key in ("outputs", "inputs"):
        with pytest.raises(ManifestError, match="not a plain relative path"):
            JobManifest.from_dict({"job_id": "a", "job_type": "t", key: [path]})


def test_a_backslash_output_cannot_forge_a_result_manifest(tmp_path):
    """On Linux jobs\\pending\\x.json is one file name, so it passed the reserved-prefix
    check, and packing it as jobs/pending/x.json put a forged manifest in the result."""
    sched = make_scheduler(tmp_path, registry(cpu_jt()))
    qr = str(tmp_path / "Q")
    os.makedirs(os.path.join(qr, "inbox"))
    raw = {"job_id": "a", "job_type": "ex.cpu", "outputs": [r"jobs\pending\other.json"]}
    with zipfile.ZipFile(os.path.join(qr, "inbox", "x.zip"), "w") as zf:
        zf.writestr("jobs/pending/a.json", __import__("json").dumps(raw))
    assert drive(sched, qr, "x.zip")
    with zipfile.ZipFile(os.path.join(qr, "outbox", "x.zip")) as zf:
        assert zf.namelist() == ["jobs/pending/a.json"]
    (res,) = wait_result(qr, "x.zip", timeout_s=5)
    assert res.status == "failed" and "not a plain relative path" in res.message


def test_pack_and_unpack_refuse_backslash_member_names(tmp_path):
    with pytest.raises(bundle.UnsafeMember, match="not a plain relative path"):
        bundle.pack(str(tmp_path / "r.zip"), [], root=str(tmp_path), extra_members=[r"a\b"])
    z = tmp_path / "in.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zi = zipfile.ZipInfo("x")
        zi.filename = "jobs\\pending\\x.json"  # ZipInfo() turns os.sep into / on Windows
        zf.writestr(zi, "{}")
    if os.name == "nt":
        # zipfile itself reads a backslash as a separator here, so the name arrives clean.
        assert zipfile.ZipFile(z).namelist() == ["jobs/pending/x.json"]
        return
    with pytest.raises(bundle.UnsafeMember, match="not a plain relative path"):
        bundle.unpack(str(z), str(tmp_path / "u"))


# ---- the claimed bundle is opened once, never by path again --------------------------


def _claim_then_swap(q, monkeypatch, swap):
    real = q.claim

    def claim(fn):
        path = real(fn)
        if path is not None:
            swap(path)
        return path

    monkeypatch.setattr(q, "claim", claim)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs POSIX FIFOs")
def test_a_fifo_swapped_in_after_the_claim_does_not_block_the_broker(tmp_path, monkeypatch):
    sched = make_scheduler(tmp_path, registry(cpu_jt()))
    qr = str(tmp_path / "Q")
    fn = submit(qr, [_job()])

    def to_fifo(path):
        os.unlink(path)
        os.mkfifo(path)

    _claim_then_swap(sched.queues[0], monkeypatch, to_fifo)
    t = threading.Thread(target=sched.tick, daemon=True)
    t.start()
    t.join(10)
    assert not t.is_alive(), "the tick blocked opening a FIFO"
    (res,) = wait_result(qr, fn, timeout_s=5)
    assert res.status == "failed" and "regular file" in res.message


def test_a_symlink_swapped_in_after_the_claim_is_not_unpacked(tmp_path, monkeypatch):
    other = tmp_path / "other.zip"
    bundle.pack(str(other), [_job("other").to_dict()], root=str(tmp_path))
    probe = tmp_path / "probe"
    try:
        os.symlink(other, probe)
    except (OSError, NotImplementedError):
        pytest.skip("this host does not let the test user create symlinks")
    os.unlink(probe)
    sched = make_scheduler(tmp_path, registry(cpu_jt()))
    qr = str(tmp_path / "Q")
    fn = submit(qr, [_job("mine")])

    def to_link(path):
        os.unlink(path)
        os.symlink(other, path)

    _claim_then_swap(sched.queues[0], monkeypatch, to_link)
    assert drive(sched, qr, fn)
    results = wait_result(qr, fn, timeout_s=5)
    assert [r.job_id for r in results] != ["other"]
    assert all(r.status == "failed" for r in results)


def test_status_snapshot_creates_nothing_in_a_queue_tree(tmp_path):
    sched = make_scheduler(tmp_path, registry(cpu_jt()))
    sched.write_status()
    assert not os.path.exists(tmp_path / "Q")
    assert QueueTree(str(tmp_path / "Q")).list_inbox() == []
