"""Weight cache staging + verification (SPEC 6.2).

Offline containers (``--network none``) can only load models from a pre-staged,
hash-verified, read-only cache. The host builds the cache once (online), writes a
``weights.sha256`` manifest, and the broker refuses to launch a job type whose cache
fails verification.
"""

from __future__ import annotations

import hashlib
import os

MANIFEST_NAME = "weights.sha256"


def _is_cruft(rel: str) -> bool:
    """HF cache runtime cruft (logs/locks the downloader writes into the cache) that
    isn't model data and changes between runs. Hashing it makes the cache unverifiable
    (for example 'unexpected file: xet/logs/xet_*.log'). Skip it in both manifest and verify."""
    if rel.endswith(".log") or rel.endswith(".lock"):
        return True
    parts = rel.split("/")
    return ".locks" in parts or ".no_exist" in parts


def _sha256(path: str, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _iter_files(root: str):
    for dirpath, _dirs, files in os.walk(root):
        for fn in sorted(files):
            if fn == MANIFEST_NAME:
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace("\\", "/")
            if _is_cruft(rel):
                continue
            yield rel, full


def write_manifest(cache_dir: str) -> str:
    """Hash every file under *cache_dir* and write the manifest. Returns its path."""
    lines = []
    for rel, full in _iter_files(cache_dir):
        lines.append(f"{_sha256(full)}  {rel}")
    manifest_path = os.path.join(cache_dir, MANIFEST_NAME)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))
    return manifest_path


def verify(cache_dir: str) -> tuple[bool, list[str]]:
    """Verify a cache against its manifest. Returns (ok, problems)."""
    manifest_path = os.path.join(cache_dir, MANIFEST_NAME)
    if not os.path.isfile(manifest_path):
        return False, [f"missing {MANIFEST_NAME}"]
    expected: dict[str, str] = {}
    with open(manifest_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            digest, _, rel = line.partition("  ")
            expected[rel] = digest
    problems: list[str] = []
    seen = set()
    for rel, full in _iter_files(cache_dir):
        seen.add(rel)
        if rel not in expected:
            problems.append(f"unexpected file: {rel}")
        elif _sha256(full) != expected[rel]:
            problems.append(f"hash mismatch: {rel}")
    for rel in expected:
        if rel not in seen:
            problems.append(f"missing file: {rel}")
    return (not problems), problems
