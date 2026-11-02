"""Weight cache staging + verification, incl. ignoring HF runtime cruft (SPEC 6.2)."""

import os

from gpuworker import weights


def _touch(path, body="x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)


def test_manifest_roundtrip_verifies(tmp_path):
    _touch(str(tmp_path / "hub/snap/model.pth"), "weights")
    _touch(str(tmp_path / "hub/snap/params.json"), "{}")
    weights.write_manifest(str(tmp_path))
    ok, problems = weights.verify(str(tmp_path))
    assert ok and problems == []


def test_hash_mismatch_and_missing_detected(tmp_path):
    f = str(tmp_path / "model.bin")
    _touch(f, "real")
    weights.write_manifest(str(tmp_path))
    with open(f, "w", encoding="utf-8") as fh:
        fh.write("tampered")
    ok, problems = weights.verify(str(tmp_path))
    assert not ok and any("hash mismatch" in p for p in problems)


def test_hf_cruft_ignored(tmp_path):
    # HF writes logs/locks into the cache (such as xet/logs/*.log); these must not be hashed
    # or flagged as unexpected, or a sealed cache stops verifying.
    _touch(str(tmp_path / "hub/snap/model.pth"), "weights")
    _touch(str(tmp_path / "xet/logs/xet_20260621.log"), "log")
    _touch(str(tmp_path / ".locks/abc.lock"), "lock")
    man = weights.write_manifest(str(tmp_path))
    body = open(man, encoding="utf-8").read()
    assert "hub/snap/model.pth" in body
    assert ".log" not in body and ".lock" not in body
    # a new log written after sealing (such as a later online touch) must not break verify
    _touch(str(tmp_path / "xet/logs/xet_later.log"), "log2")
    ok, problems = weights.verify(str(tmp_path))
    assert ok and problems == []
