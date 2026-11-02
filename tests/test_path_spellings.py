"""One path, one spelling (SPEC 6.3).

A path written differently from the spelling the broker checked (a backslash, a ``.``
part, a link) could be validated in one form and used in another. So the broker accepts
a bundle member or a manifest path only in its canonical form, so no two spellings can
name one file, and it validates the job manifests from the unpacked
files a worker reads. These tests generate spelling variants and check that every one is
refused.
"""

import json
import os
import posixpath
import random
import zipfile

import pytest

from gpuworker import bundle, gpu
from gpuworker.client import submit, wait_result
from gpuworker.launcher import LocalLauncher
from gpuworker.manifest import (
    MAX_PATHS_PER_JOB,
    JobManifest,
    ManifestError,
    is_bundle_path,
)
from gpuworker.module import scan_registry
from gpuworker.scheduler import Scheduler

from helpers import cpu_jt, drive, make_config, make_scheduler, registry


def _variants(path: str) -> set[str]:
    """Spellings that name *path*, or that some reader could take for it."""
    parts = path.split("/")
    out = {
        "/" + path,
        "./" + path,
        path + "/.",
        path.replace("/", "//"),
        path.replace("/", "\\"),
        path + ".",
        path + " ",
        path + ":stream",
        path.upper(),
    }
    for i in range(1, len(parts)):
        head, tail = "/".join(parts[:i]), "/".join(parts[i:])
        out |= {
            f"{head}/./{tail}",
            f"{head}//{tail}",
            f"{head}/x/../{tail}",
            f"{head}/../{parts[i - 1]}/{tail}",
            f"{head}\\{tail}",
            f"{head}. /{tail}",
        }
    out.discard(path)
    return out


MANIFEST = "jobs/pending/x.json"
OUTPUT = "out/sub/x.wav"


@pytest.mark.parametrize("spelling", sorted(_variants(MANIFEST) - {MANIFEST.upper()}))
def test_no_other_spelling_of_a_manifest_is_unpacked(tmp_path, spelling):
    """A second member that unpacks onto the validated manifest. Every spelling is
    refused before anything is written."""
    z = tmp_path / "b.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr(MANIFEST, json.dumps({"job_id": "x", "job_type": "ex.cpu"}))
        zi = zipfile.ZipInfo("placeholder")
        zi.filename = spelling  # ZipInfo() would normalize a backslash on Windows
        zf.writestr(zi, json.dumps({"job_id": "x", "job_type": "ex.cpu", "outputs": ["/x"]}))
    names = zipfile.ZipFile(z).namelist()
    if names[1] == MANIFEST:
        # Windows' zipfile reads a backslash as a separator: the second member arrives as
        # a duplicate of the first, which is refused as such.
        assert os.name == "nt"
    with pytest.raises(bundle.UnsafeMember):
        bundle.unpack(str(z), str(tmp_path / "u"))
    assert not (tmp_path / "u" / "jobs").exists()


def test_members_that_differ_only_in_case_are_refused(tmp_path):
    """On a case-insensitive filesystem these unpack onto one file."""
    z = tmp_path / "b.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr(MANIFEST, "{}")
        zf.writestr(MANIFEST.upper(), "{}")
    with pytest.raises(bundle.UnsafeMember, match="same path"):
        bundle.unpack(str(z), str(tmp_path / "u"))


@pytest.mark.parametrize("spelling", sorted(_variants(OUTPUT) - {OUTPUT.upper()}))
def test_no_other_spelling_of_an_output_is_accepted(spelling):
    with pytest.raises(ManifestError, match="not a plain relative path"):
        JobManifest.from_dict({"job_id": "a", "job_type": "t", "outputs": [spelling]})
    with pytest.raises(ManifestError, match="not a plain relative path"):
        JobManifest.from_dict({"job_id": "a", "job_type": "t", "inputs": [spelling]})


@pytest.mark.parametrize(
    "path", ["C:/data/file.txt", "/etc/passwd", "out/\x00x", "out/x\ty", "a/" * 600 + "b"]
)
def test_host_paths_and_control_characters_are_refused(path):
    assert not is_bundle_path(path)


def test_every_accepted_path_is_already_normalized():
    """A property over random strings from an alphabet of the characters that matter: a
    path is accepted only if normalizing it would change nothing."""
    rng = random.Random(4)
    alphabet = "ab./\\: "
    accepted = 0
    for _ in range(20000):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 10)))
        if is_bundle_path(s):
            accepted += 1
            assert posixpath.normpath(s) == s and not s.startswith("/")
            assert all(part not in (".", "..") and part.strip(". ") for part in s.split("/"))
    assert accepted  # the property was not vacuous


def _echo_scheduler(tmp_path):
    return Scheduler(
        make_config(tmp_path), scan_registry("modules"), LocalLauncher(), gpu.MockGpu()
    )


def test_a_second_manifest_spelling_cannot_make_echo_write_outside_the_queue(tmp_path):
    """End to end, through the scheduler and the shipped echo.echo
    worker: the batch fails and the target is never written."""
    target = tmp_path / "outside.txt"
    sched = _echo_scheduler(tmp_path)
    qr = str(tmp_path / "Q")
    for q in sched.queues:
        q.ensure()
    benign = {"job_id": "x", "job_type": "echo.echo", "outputs": ["out/hello.txt"]}
    other = dict(benign, outputs=["out/hello.txt", str(target).replace("\\", "/")])
    zp = os.path.join(qr, "inbox", "other.zip")
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr("jobs/pending/x.json", json.dumps(benign))
        zf.writestr("jobs/./pending/x.json", json.dumps(other))
    assert drive(sched, qr, "other.zip", ticks=400)
    (res,) = wait_result(qr, "other.zip", timeout_s=5)
    assert res.status == "failed" and "not a plain relative path" in res.message
    assert not target.exists()


def test_the_worker_reads_the_manifest_the_broker_validated(tmp_path):
    """The broker rewrites each manifest from what it parsed, so a key it does not know
    never reaches the worker, and the worker sees the broker's own copy."""
    seen = tmp_path / "seen.json"
    jt = cpu_jt(
        code="import os, shutil\n"
        "src = os.path.join(os.environ['GPUWORKER_BUNDLE'], 'jobs', 'pending', 'a.json')\n"
        f"shutil.copy(src, {str(seen)!r})\n"
        "os.makedirs(os.path.join(os.environ['GPUWORKER_OUT'], 'out'), exist_ok=True)\n"
        "open(os.path.join(os.environ['GPUWORKER_OUT'], 'out', 'a.txt'), 'w').close()\n"
    )
    sched = make_scheduler(tmp_path, registry(jt))
    qr = str(tmp_path / "Q")
    z = os.path.join(qr, "inbox", "b.zip")
    os.makedirs(os.path.dirname(z))
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr(
            "jobs/pending/a.json",
            json.dumps(
                {"job_id": "a", "job_type": jt.id, "outputs": ["out/a.txt"], "unknown": "x"}
            ),
        )
    assert drive(sched, qr, "b.zip", ticks=2000)
    assert wait_result(qr, "b.zip", timeout_s=5)[0].status == "done"
    on_disk = json.loads(seen.read_text(encoding="utf-8"))
    assert "unknown" not in on_disk and on_disk["status"] == "queued"


@pytest.mark.parametrize(
    "members, match",
    [
        ({"jobs/pending/a.json": {"job_id": "b", "job_type": "ex.cpu"}}, "named <job_id>.json"),
        (
            {"jobs/pending/a.json": {"job_id": "a", "job_type": "ex.cpu"}, "jobs/pending/n": {}},
            "not a .json file",
        ),
        (
            {
                "jobs/pending/a.json": {"job_id": "a", "job_type": "ex.cpu"},
                "jobs/pending/sub/b.json": {"job_id": "b", "job_type": "ex.cpu"},
            },
            "not a .json file",
        ),
    ],
)
def test_the_pending_directory_holds_only_job_id_named_manifests(tmp_path, members, match):
    """A manifest under another name, or anything else under jobs/pending/, could be read
    by a worker looking for its own file, so the batch fails."""
    sched = make_scheduler(tmp_path, registry(cpu_jt()))
    qr = str(tmp_path / "Q")
    z = os.path.join(qr, "inbox", "b.zip")
    os.makedirs(os.path.dirname(z))
    with zipfile.ZipFile(z, "w") as zf:
        for name, body in members.items():
            zf.writestr(name, json.dumps(body))
    assert drive(sched, qr, "b.zip", ticks=400)
    res = wait_result(qr, "b.zip", timeout_s=5)
    assert all(r.status == "failed" for r in res)
    assert any(match in r.message for r in res), [r.message for r in res]


# ---- output declarations stay cheap to check -------------------------------------------


@pytest.mark.parametrize(
    "first, second, overlaps",
    [
        ("out/a", "out/a", True),
        ("out/a", "out/a/b", True),
        ("out/a/b", "out/a", True),
        ("out", "out/a/b/c", True),
        ("out/a", "out/ab", False),
        ("out/a/b", "out/a/c", False),
        # One file on a case-insensitive filesystem.
        ("out/A", "out/a", True),
        ("OUT/a/b", "out/A", True),
        ("Out", "out/a", True),
    ],
)
def test_output_overlap_is_found_in_either_order(tmp_path, first, second, overlaps):
    sched = make_scheduler(tmp_path, registry(cpu_jt()))
    qr = str(tmp_path / "Q")
    fn = submit(qr, [_jm("a", [first]), _jm("b", [second])])
    sched.intake()
    if overlaps:
        res = wait_result(qr, fn, timeout_s=5)
        assert all(r.status == "failed" for r in res)
        assert "overlaps" in res[0].message
    else:
        assert len(sched.pending) == 2


def _jm(job_id, outputs):
    return JobManifest.from_dict({"job_id": job_id, "job_type": "ex.cpu", "outputs": outputs})


def test_outputs_are_capped_per_job():
    outputs = [f"o/{i}" for i in range(MAX_PATHS_PER_JOB + 1)]
    with pytest.raises(ManifestError, match="more than"):
        JobManifest.from_dict({"job_id": "a", "job_type": "t", "outputs": outputs})


def test_outputs_are_capped_per_bundle(tmp_path):
    """Each output becomes a result member, so a bundle may not declare more outputs than
    the member limit, however they are spread across its jobs."""
    sched = make_scheduler(tmp_path, registry(cpu_jt()), max_bundle_members=10)
    qr = str(tmp_path / "Q")
    fn = submit(
        qr, [_jm("a", [f"a/{i}" for i in range(6)]), _jm("b", [f"b/{i}" for i in range(6)])]
    )
    sched.intake()
    res = wait_result(qr, fn, timeout_s=5)
    assert all(r.status == "failed" for r in res)
    assert "more than 10 outputs" in res[0].message
