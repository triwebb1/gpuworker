import os
import zipfile

import pytest

from gpuworker import bundle
from gpuworker.share import QueueTree


def test_atomic_claim_only_one_winner(tmp_path):
    q = QueueTree(str(tmp_path / "client")).ensure()
    # drop a bundle in inbox
    bundle.pack(
        os.path.join(q.dirs["inbox"], "j1.zip"),
        [{"job_id": "j1", "job_type": "echo.echo"}],
        root=str(tmp_path),
    )
    first = q.claim("j1.zip")
    second = q.claim("j1.zip")
    assert first is not None
    assert second is None  # already claimed


def test_fail_batch_reports_every_job(tmp_path):
    q = QueueTree(str(tmp_path / "client")).ensure()
    # A "bundle" that read_pending_manifests can read but is otherwise unusable.
    proc = os.path.join(q.dirs["processing"], "batch.zip")
    with zipfile.ZipFile(proc, "w") as zf:
        zf.writestr("jobs/pending/j1.json", '{"job_id":"j1","job_type":"echo.echo"}')
        zf.writestr("jobs/pending/j2.json", '{"job_id":"j2","job_type":"echo.echo"}')
    reported = q.fail_batch(proc, "unpack exploded")
    assert set(reported) == {"j1", "j2"}
    # result bundle landed in outbox with both jobs failed
    out = os.path.join(q.dirs["outbox"], "batch.zip")
    assert os.path.exists(out)
    results = bundle.read_pending_manifests(out)
    assert all(r["status"] == "failed" for r in results)
    # the offending batch moved to failed/ with an error note
    assert os.path.exists(os.path.join(q.dirs["failed"], "batch.zip"))
    assert os.path.exists(os.path.join(q.dirs["failed"], "batch.zip.error.txt"))


def test_reclaim_stale(tmp_path):
    q = QueueTree(str(tmp_path / "client")).ensure()
    proc = os.path.join(q.dirs["processing"], "old.zip")
    with open(proc, "w") as fh:
        fh.write("x")
    old = os.path.getmtime(proc) - 10000
    os.utime(proc, (old, old))
    reclaimed = q.reclaim_stale(max_age_s=3600)
    assert reclaimed == ["old.zip"]
    assert os.path.exists(os.path.join(q.dirs["inbox"], "old.zip"))


def test_fail_batch_never_echoes_an_invalid_job_id_as_a_member_name(tmp_path):
    # A traversal job_id is rejected at parse time; the synthesized failure must not put
    # it back into a zip member path, but should still say what was submitted.
    q = QueueTree(str(tmp_path / "client")).ensure()
    proc = os.path.join(q.dirs["processing"], "bad.zip")
    with zipfile.ZipFile(proc, "w") as zf:
        zf.writestr("jobs/pending/x.json", '{"job_id": "../../escape", "job_type": "echo.echo"}')
    reported = q.fail_batch(proc, "bad bundle")
    assert reported == ["unknown-0"]
    with zipfile.ZipFile(os.path.join(q.dirs["outbox"], "bad.zip")) as zf:
        assert zf.namelist() == ["jobs/pending/unknown-0.json"]
    [m] = bundle.read_pending_manifests(os.path.join(q.dirs["outbox"], "bad.zip"))
    assert "'../../escape'" in m["message"] and m["status"] == "failed"


def test_fail_batch_ships_result_for_unreadable_zip(tmp_path):
    """An unreadable/empty batch yields no manifests; fail_batch must still synthesize
    a terminal entry + ship a result bundle to outbox, or a --wait submitter hangs forever."""
    from gpuworker.share import QueueTree

    q = QueueTree(str(tmp_path / "Q")).ensure()
    bad = os.path.join(q.dirs["processing"], "bad.zip")
    with open(bad, "wb") as fh:
        fh.write(b"not a zip at all")
    reported = q.fail_batch(bad, "corrupt bundle")
    assert reported  # >=1 entry even with 0 parsed manifests
    assert os.path.exists(os.path.join(q.dirs["outbox"], "bad.zip"))  # result was shipped
    assert os.path.exists(os.path.join(q.dirs["failed"], "bad.zip"))  # batch moved aside


def test_fail_batch_survives_non_object_manifest(tmp_path):
    """A zip member that parses to a non-object (a bare JSON list/str) must
    not AttributeError out of fail_batch (raw.get(...)); it still ships a terminal failure."""
    import zipfile

    from gpuworker.bundle import PENDING_PREFIX
    from gpuworker.share import QueueTree

    q = QueueTree(str(tmp_path / "Q")).ensure()
    weird = os.path.join(q.dirs["processing"], "weird.zip")
    with zipfile.ZipFile(weird, "w") as zf:
        zf.writestr(f"{PENDING_PREFIX}x.json", "[1, 2, 3]")  # valid JSON, not an object
    reported = q.fail_batch(weird, "non-object member")  # must not raise
    assert reported
    assert os.path.exists(os.path.join(q.dirs["outbox"], "weird.zip"))


def test_the_claim_stamp_updates_the_claimed_file(tmp_path):
    q = QueueTree(str(tmp_path / "client")).ensure()
    p = os.path.join(q.dirs["processing"], "a.zip")
    with open(p, "w") as fh:
        fh.write("x")
    os.utime(p, (1000, 1000))
    q._dir("processing").touch("a.zip")
    assert os.path.getmtime(p) > 1000


def test_the_claim_stamp_never_follows_a_link(tmp_path):
    """A client that swaps the claimed bundle for a link must not get the broker to set
    the times of the link's target."""
    import pytest

    q = QueueTree(str(tmp_path / "client")).ensure()
    target = tmp_path / "host_file.txt"
    target.write_text("x")
    os.utime(target, (1000, 1000))
    try:
        os.symlink(target, os.path.join(q.dirs["processing"], "a.zip"))
    except OSError:
        pytest.skip("creating a symlink needs a privilege on this host")
    with pytest.raises(bundle.UnsafeMember):
        q._dir("processing").touch("a.zip")
    assert os.path.getmtime(target) == 1000


def test_a_delivered_bundle_left_in_processing_is_archived(tmp_path):
    """If archiving failed after delivery, reclaim skipped the bundle (it has a result) and
    nothing ever moved it."""
    q = QueueTree(str(tmp_path / "client")).ensure()
    for d in ("processing", "outbox"):
        with open(os.path.join(q.dirs[d], "done.zip"), "w") as fh:
            fh.write(d)
    with open(os.path.join(q.dirs["processing"], "live.zip"), "w") as fh:
        fh.write("live")
    with open(os.path.join(q.dirs["outbox"], "live.zip"), "w") as fh:
        fh.write("result")
    assert q.archive_leftovers(skip={"live.zip"}) == ["done.zip"]
    assert os.listdir(q.dirs["processing"]) == ["live.zip"]
    with open(os.path.join(q.dirs["archive"], "done.zip")) as fh:
        assert fh.read() == "processing"


def test_fail_batch_lists_unreadable_manifests_up_to_its_cap(tmp_path):
    """SPEC 7: one synthetic entry for each of the first MAX_UNREADABLE_ENTRIES manifests
    that could not be read, beside the ones that could."""
    from gpuworker.share import MAX_UNREADABLE_ENTRIES

    q = QueueTree(str(tmp_path / "client")).ensure()
    proc = os.path.join(q.dirs["processing"], "batch.zip")
    with zipfile.ZipFile(proc, "w") as zf:
        zf.writestr("jobs/pending/ok.json", '{"job_id":"ok","job_type":"echo.echo"}')
        for i in range(MAX_UNREADABLE_ENTRIES + 5):
            zf.writestr(f"jobs/pending/bad{i}.json", "{not json")
    reported = q.fail_batch(proc, "bad bundle")
    assert "ok" in reported
    assert len(reported) == 1 + MAX_UNREADABLE_ENTRIES


def test_reclaim_never_reruns_a_bundle_when_the_share_cannot_say_if_it_has_a_result(
    tmp_path, monkeypatch
):
    """has_result turned a storage error into "no result", so a share hiccup made reclaim
    put a delivered bundle back in inbox/ to run again."""
    from gpuworker import share

    q = QueueTree(str(tmp_path / "client")).ensure()
    proc = os.path.join(q.dirs["processing"], "done.zip")
    with open(proc, "w") as fh:
        fh.write("x")
    old = os.path.getmtime(proc) - 10000
    os.utime(proc, (old, old))
    with open(os.path.join(q.dirs["outbox"], "done.zip"), "w") as fh:
        fh.write("result")
    real_exists = share.QueueDir.exists

    def flaky_exists(self, name):
        if self.path == q.dirs["outbox"]:
            raise OSError(5, "Input/output error")
        return real_exists(self, name)

    monkeypatch.setattr(share.QueueDir, "exists", flaky_exists)
    assert q.reclaim_stale(max_age_s=3600) == []
    assert q.archive_leftovers() == []
    assert os.path.exists(proc)
    monkeypatch.setattr(share.QueueDir, "exists", real_exists)
    assert q.archive_leftovers() == ["done.zip"]


def test_a_bundle_that_could_not_be_removed_after_its_copy_has_not_moved(tmp_path, monkeypatch):
    """Across filesystems a move is a copy and an unlink. When the unlink failed it still
    reported the move done, so a bundle sent to failed/ stayed in processing/ untracked,
    and stale reclaim ran it again."""
    from gpuworker import share

    q = QueueTree(str(tmp_path / "client")).ensure()
    with open(os.path.join(q.dirs["processing"], "b.zip"), "w") as fh:
        fh.write("x")

    def cross_device(*a):
        raise OSError(18, "Invalid cross-device link")

    real_unlink = share.QueueDir.unlink

    def stuck_unlink(self, name):
        if self.path == q.dirs["processing"]:
            raise PermissionError(13, "Permission denied")
        real_unlink(self, name)

    monkeypatch.setattr(share, "_move_noreplace", cross_device)
    monkeypatch.setattr(share.QueueDir, "unlink", stuck_unlink)
    with pytest.raises(OSError):
        q.move_to_failed("b.zip", "no result could be written")
    assert os.listdir(q.dirs["failed"]) == []
    assert os.listdir(q.dirs["processing"]) == ["b.zip"]
    monkeypatch.setattr(share.QueueDir, "unlink", real_unlink)
    assert q.move_to_failed("b.zip", "no result could be written") == "b.zip"
    assert os.listdir(q.dirs["processing"]) == []
