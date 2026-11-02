import pytest

from gpuworker.manifest import MANIFEST_SCHEMA, JobManifest, ManifestError


def test_minimal_valid():
    m = JobManifest.from_dict({"job_id": "j1", "job_type": "echo.echo"})
    assert m.job_id == "j1"
    assert m.status == "queued"
    assert not m.is_terminal


def test_rejects_missing_ids():
    with pytest.raises(ManifestError):
        JobManifest.from_dict({"job_type": "echo.echo"})
    with pytest.raises(ManifestError):
        JobManifest.from_dict({"job_id": "  ", "job_type": "echo.echo"})


def test_rejects_future_schema():
    with pytest.raises(ManifestError):
        JobManifest.from_dict({"job_id": "j", "job_type": "t", "schema": MANIFEST_SCHEMA + 1})


def test_rejects_bad_types():
    with pytest.raises(ManifestError):
        JobManifest.from_dict({"job_id": "j", "job_type": "t", "inputs": "notalist"})
    with pytest.raises(ManifestError):
        JobManifest.from_dict({"job_id": "j", "job_type": "t", "timeout_s": -1})
    with pytest.raises(ManifestError):  # absurd value would overflow settimeout's C timeval
        JobManifest.from_dict({"job_id": "j", "job_type": "t", "timeout_s": 10**9})


def test_transitions_are_terminal():
    m = JobManifest.from_dict({"job_id": "j", "job_type": "t"})
    m.mark_running()
    assert not m.is_terminal
    m.mark_done(result={"outputs": []})
    assert m.is_terminal and m.status == "done" and m.finished_utc

    m2 = JobManifest.from_dict({"job_id": "j2", "job_type": "t"})
    m2.mark_failed("boom")
    assert m2.is_terminal and m2.status == "failed" and m2.message == "boom"


def test_roundtrip_dict():
    m = JobManifest.from_dict({"job_id": "j", "job_type": "t", "priority": 3})
    d = m.to_dict()
    m2 = JobManifest.from_dict(d)
    assert m2.priority == 3 and m2.job_id == "j"


@pytest.mark.parametrize(
    "job_id",
    [
        "../escape",
        "../../../../etc",
        "/abs/path",
        "a/b",
        r"a\b",
        ".hidden",
        "-flag",
        "has space",
        "x" * 129,
        r"C:\x",
    ],
)
def test_rejects_job_ids_that_are_not_plain_names(job_id):
    # job_id becomes a host path component (work dirs, worker logs, serve staging that is
    # later rmtree'd), and a mistyped one must fail cleanly.
    with pytest.raises(ManifestError, match="job_id"):
        JobManifest.from_dict({"job_id": job_id, "job_type": "echo.echo"})


@pytest.mark.parametrize("job_id", ["j1", "echo-1a2b3c4d", "A.b_c-9", "x" * 128])
def test_accepts_plain_job_ids(job_id):
    assert JobManifest.from_dict({"job_id": job_id, "job_type": "echo.echo"}).job_id == job_id
