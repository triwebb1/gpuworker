"""The queue tree may be writable beyond inbox/, so nothing the broker writes there may
follow a stray link, the broker's working files live outside it, and no bundle can wedge
intake or replace another one (SPEC 6.3, 7, 10).

Symlink tests skip where the test user cannot create symlinks (a default Windows
account); Linux CI runs them.
"""

import json
import os
import time
import zipfile

import pytest

from gpuworker import bundle
from gpuworker.client import submit, wait_result
from gpuworker.manifest import JobManifest
from gpuworker.share import ClaimRejected, QueueTree

from helpers import cpu_jt, drive, make_config, make_scheduler, registry


@pytest.fixture
def symlinks(tmp_path):
    probe = tmp_path / "symlink-probe"
    try:
        os.symlink(tmp_path, probe, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this host does not let the test user create symlinks")
    os.unlink(probe)


def _host_file(tmp_path, name="host_file.txt"):
    d = tmp_path / "home" / ".ssh"
    d.mkdir(parents=True)
    f = d / name
    f.write_text("original\n")
    return f


def _job(job_id="a", **kw):
    return JobManifest.from_dict({"job_id": job_id, "job_type": "ex.cpu", **kw})


def _raw_zip(path, members):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)


# ---- the broker's own files are outside the queue tree ------------------------------


def test_working_files_never_sit_in_the_queue_tree(tmp_path):
    seen = []
    code = "import os; open(os.path.join(os.environ['GPUWORKER_OUT'], 'o.txt'), 'w').write('x')"
    sched = make_scheduler(tmp_path, registry(cpu_jt(code=code)))
    qr = str(tmp_path / "Q")
    fn = submit(qr, [_job(outputs=["o.txt"])])
    for _ in range(2000):
        sched.tick()
        for dp, dirs, files in os.walk(qr):
            seen += [os.path.join(dp, n) for n in dirs + files]
        if os.path.exists(os.path.join(qr, "outbox", fn)):
            break
        time.sleep(0.005)  # the job is a subprocess; give it time to finish
    assert wait_result(qr, fn, timeout_s=5)[0].status == "done"
    assert not [p for p in seen if p.endswith((".work", ".out", ".run"))]


@pytest.mark.parametrize("inside", ["state_in_queue", "queue_in_state"])
def test_state_dir_and_a_queue_root_must_not_nest(tmp_path, inside):
    q = tmp_path / "Q"
    state = q / "state" if inside == "state_in_queue" else tmp_path / "S"
    roots = [str(q)] if inside == "state_in_queue" else [str(state / "Q")]
    with pytest.raises(ValueError, match="must not contain"):
        make_scheduler(tmp_path, registry(cpu_jt()), queue_roots=roots, state_dir=str(state))


def test_serve_clears_what_a_previous_run_left_in_the_state_dir(tmp_path):
    sched = make_scheduler(tmp_path, registry(cpu_jt()))
    stale = os.path.join(sched.state_dir, "batches", "old-abc", "work")
    os.makedirs(stale)
    sched._clear_state()
    assert not os.path.exists(stale)


# ---- writes into queue directories do not follow links ------------------------------


def test_result_write_replaces_a_stray_link_instead_of_writing_through_it(tmp_path, symlinks):
    host_file = _host_file(tmp_path)
    sched = make_scheduler(tmp_path, registry(cpu_jt()))
    qr = str(tmp_path / "Q")
    fn = submit(qr, [_job()])
    sched.tick()  # claimed: a link made now is not seen by the claim-time check
    assert os.listdir(os.path.join(qr, "processing")) == [fn]
    os.symlink(host_file, os.path.join(qr, "outbox", fn))
    for _ in range(2000):
        sched.tick()
        if not os.path.islink(os.path.join(qr, "outbox", fn)):
            break
        time.sleep(0.005)
    assert host_file.read_text() == "original\n"
    assert not os.path.islink(os.path.join(qr, "outbox", fn))
    assert wait_result(qr, fn, timeout_s=5)[0].status == "done"


def test_failed_batch_files_do_not_follow_stray_links(tmp_path, symlinks):
    host_file = _host_file(tmp_path)
    sched = make_scheduler(tmp_path, registry(cpu_jt()))
    qr = str(tmp_path / "Q")
    name = "bad.zip"
    sched.queues[0].ensure()  # the broker creates the tree at start
    with open(os.path.join(qr, "inbox", name), "wb") as fh:
        fh.write(b"not a zip")
    for stray in (name, name + ".error.txt"):
        os.symlink(host_file, os.path.join(qr, "failed", stray))
    assert drive(sched, qr, name)
    assert host_file.read_text() == "original\n"
    (res,) = wait_result(qr, name, timeout_s=5)
    assert res.status == "failed"


def test_broker_limits_do_not_follow_a_stray_link(tmp_path, symlinks):
    host_file = _host_file(tmp_path)
    sched = make_scheduler(tmp_path, registry(cpu_jt()))
    sched.queues[0].ensure()
    path = os.path.join(str(tmp_path / "Q"), "broker_limits.json")
    os.symlink(host_file, path)
    sched._publish_limits()
    assert host_file.read_text() == "original\n"
    with open(path, encoding="utf-8") as fh:
        assert json.load(fh)["schema"] == "gpuworker-broker-limits/v1"


def test_a_linked_queue_directory_is_refused(tmp_path, symlinks):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    q = tmp_path / "Q"
    q.mkdir()
    os.symlink(elsewhere, q / "outbox", target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        QueueTree(str(q)).ensure()


def test_write_new_is_atomic_and_leaves_no_temp_file(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    bundle.write_new_bytes(str(d), "x.json", b"one")
    bundle.write_new_bytes(str(d), "x.json", b"two")  # replaces a regular file
    assert (d / "x.json").read_bytes() == b"two"

    def boom(fh):
        fh.write(b"partial")
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError):
        bundle.write_new(str(d), "y.json", boom)
    assert sorted(os.listdir(d)) == ["x.json"]


# ---- claims never replace a bundle or an earlier result -----------------------------


def test_a_symlink_in_inbox_is_rejected_not_claimed(tmp_path, symlinks):
    host_file = _host_file(tmp_path)
    q = QueueTree(str(tmp_path / "Q")).ensure()
    os.symlink(host_file, os.path.join(q.dirs["inbox"], "x.zip"))
    with pytest.raises(ClaimRejected, match="not a regular file"):
        q.claim("x.zip")
    assert not os.listdir(q.dirs["processing"]) and not os.listdir(q.dirs["inbox"])
    assert host_file.read_text() == "original\n"


def test_a_name_already_in_processing_is_rejected(tmp_path):
    q = QueueTree(str(tmp_path / "Q")).ensure()
    for d in ("processing", "inbox"):
        with open(os.path.join(q.dirs[d], "x.zip"), "w") as fh:
            fh.write(d)
    with pytest.raises(ClaimRejected, match="already in processing"):
        q.claim("x.zip")
    with open(os.path.join(q.dirs["processing"], "x.zip")) as fh:
        assert fh.read() == "processing"  # the active bundle is untouched
    assert any(n.startswith("x.zip.rejected-") for n in os.listdir(q.dirs["failed"]))


def test_a_reused_name_cannot_overwrite_an_earlier_result(tmp_path):
    sched = make_scheduler(tmp_path, registry(cpu_jt()))
    qr = str(tmp_path / "Q")
    fn = submit(qr, [_job("first")])
    assert drive(sched, qr, fn)
    with open(os.path.join(qr, "outbox", fn), "rb") as fh:
        first_result = fh.read()
    # The same file name again, as a buggy client might send it.
    bundle.pack(os.path.join(qr, "inbox", fn), [_job("second").to_dict()], root=qr)
    for _ in range(20):
        sched.tick()
    with open(os.path.join(qr, "outbox", fn), "rb") as fh:
        assert fh.read() == first_result
    assert not os.listdir(os.path.join(qr, "inbox"))
    assert any(n.startswith(fn + ".rejected-") for n in os.listdir(os.path.join(qr, "failed")))


def test_reclaim_never_replaces_a_new_inbox_bundle(tmp_path):
    q = QueueTree(str(tmp_path / "Q")).ensure()
    for d in ("processing", "inbox"):
        with open(os.path.join(q.dirs[d], "x.zip"), "w") as fh:
            fh.write(d)
    old = os.path.getmtime(os.path.join(q.dirs["processing"], "x.zip")) - 10_000
    os.utime(os.path.join(q.dirs["processing"], "x.zip"), (old, old))
    assert q.reclaim_stale(3600) == []
    with open(os.path.join(q.dirs["inbox"], "x.zip")) as fh:
        assert fh.read() == "inbox"


# ---- one bundle cannot wedge intake -------------------------------------------------


def test_deeply_nested_manifest_gets_a_terminal_result(tmp_path):
    """json.loads raises RecursionError, not ValueError, on deep nesting. That used to
    escape both intake and the batch-failure path, so the tick aborted, no result was
    written, and the bundle re-ran every reclaim_after_s."""
    sched = make_scheduler(tmp_path, registry(cpu_jt()))
    qr = str(tmp_path / "Q")
    os.makedirs(os.path.join(qr, "inbox"), exist_ok=True)
    deep = "[" * 200_000 + "]" * 200_000  # 400 KB, under the per-manifest limit
    good = _job("b").to_dict()
    _raw_zip(
        os.path.join(qr, "inbox", "deep.zip"),
        {"jobs/pending/a.json": deep, "jobs/pending/b.json": json.dumps(good)},
    )
    problems: list[str] = []
    found = bundle.read_pending_manifests(os.path.join(qr, "inbox", "deep.zip"), problems=problems)
    assert found == [good] and problems == ["jobs/pending/a.json: unreadable (RecursionError)"]
    assert drive(sched, qr, "deep.zip")
    results = wait_result(qr, "deep.zip", timeout_s=5)
    assert results and all(r.status == "failed" for r in results)
    assert os.path.exists(os.path.join(qr, "failed", "deep.zip"))
    # The next bundle is still served.
    fn = submit(qr, [_job("after")])
    assert drive(sched, qr, fn)
    assert wait_result(qr, fn, timeout_s=5)[0].status == "done"


def test_manifest_reading_is_bounded_on_the_failure_path(tmp_path):
    z = str(tmp_path / "many.zip")
    _raw_zip(z, {f"jobs/pending/j{i}.json": json.dumps({"job_id": f"j{i}"}) for i in range(50)})
    assert bundle.read_pending_manifests(z, max_members=49) == []  # over the member cap
    some = bundle.read_pending_manifests(z, max_total_bytes=100)
    assert 0 < len(some) < 50  # stops once the byte budget is spent
    q = QueueTree(str(tmp_path / "Q")).ensure()
    os.replace(z, os.path.join(q.dirs["processing"], "many.zip"))
    reported = q.fail_batch(os.path.join(q.dirs["processing"], "many.zip"), "x", max_members=49)
    assert reported == ["unknown-0"]  # one synthesized entry, read nothing
    [m] = bundle.read_pending_manifests(os.path.join(q.dirs["outbox"], "many.zip"))
    assert "50 members (limit 49)" in m["message"]


def test_a_non_object_manifest_fails_the_batch_cleanly(tmp_path):
    sched = make_scheduler(tmp_path, registry(cpu_jt()))
    qr = str(tmp_path / "Q")
    os.makedirs(os.path.join(qr, "inbox"), exist_ok=True)
    _raw_zip(os.path.join(qr, "inbox", "list.zip"), {"jobs/pending/a.json": "[1, 2]"})
    assert drive(sched, qr, "list.zip")
    (res,) = wait_result(qr, "list.zip", timeout_s=5)
    assert res.status == "failed" and "not a JSON object" in res.message


def test_a_result_write_failure_on_the_failure_path_does_not_stop_the_loop(tmp_path, monkeypatch):
    sched = make_scheduler(tmp_path, registry(cpu_jt()))
    qr = str(tmp_path / "Q")
    os.makedirs(os.path.join(qr, "inbox"), exist_ok=True)
    with open(os.path.join(qr, "inbox", "bad.zip"), "wb") as fh:
        fh.write(b"not a zip")
    q = sched.queues[0]

    def no_share(*a, **kw):
        raise OSError("share gone")

    monkeypatch.setattr(q, "fail_batch", no_share)
    sched.tick()  # must not raise
    assert os.path.exists(os.path.join(qr, "processing", "bad.zip"))  # kept for reclaim


def test_make_config_keeps_state_under_tmp(tmp_path):
    assert make_config(tmp_path).state_dir == str(tmp_path / "state")
