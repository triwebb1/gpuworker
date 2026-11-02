import zipfile

import pytest

from gpuworker import bundle


def test_pack_unpack_roundtrip(tmp_path):
    root = tmp_path / "root"
    (root / "inputs").mkdir(parents=True)
    (root / "inputs" / "a.txt").write_text("hello")
    zip_path = tmp_path / "b.zip"
    manifests = [{"job_id": "j1", "job_type": "echo.echo"}]
    bundle.pack(str(zip_path), manifests, root=str(root), extra_members=["inputs/a.txt"])
    dest = tmp_path / "dest"
    members = bundle.unpack(str(zip_path), str(dest))
    assert "jobs/pending/j1.json" in members
    assert (dest / "inputs" / "a.txt").read_text() == "hello"


def test_read_pending_manifests_from_bad_zip(tmp_path):
    # A zip that has a manifest but also a corrupt entry still yields the manifest.
    zip_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("jobs/pending/j1.json", '{"job_id":"j1","job_type":"echo.echo"}')
        zf.writestr("jobs/pending/broken.json", "{not json")
    out = bundle.read_pending_manifests(str(zip_path))
    assert [m["job_id"] for m in out] == ["j1"]


def test_read_pending_manifests_on_nonzip(tmp_path):
    p = tmp_path / "notazip.zip"
    p.write_text("this is not a zip")
    assert bundle.read_pending_manifests(str(p)) == []


def test_traversal_guard_blocks_escape(tmp_path):
    zip_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../escape.txt", "x")
    with pytest.raises(ValueError, match="not a plain relative path"):
        bundle.unpack(str(zip_path), str(tmp_path / "dest"))


def test_traversal_guard_blocks_absolute(tmp_path):
    zip_path = tmp_path / "bad2.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        # zip with a member that normalizes outside root
        zf.writestr("sub/../../x.txt", "x")
    with pytest.raises(ValueError):
        bundle.unpack(str(zip_path), str(tmp_path / "dest2"))


def test_write_new_takes_a_name_near_the_length_limit(tmp_path):
    """The temporary file's name does not grow with *name*, so any name the filesystem
    accepts can be written."""
    import os

    d = str(tmp_path)
    if os.name == "nt":
        d = "\\\\?\\" + d  # past MAX_PATH; the per-name limit of 255 still applies
    name = "r" * 251 + ".zip"
    bundle.write_new_bytes(d, name, b"x")
    assert os.listdir(d) == [name]
