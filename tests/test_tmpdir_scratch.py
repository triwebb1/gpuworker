"""Each container gets its own scratch directory, and TMPDIR lands there, not on the RAM
tmpfs.

A long job can fill the /tmp tmpfs, and a library's per-init shared-object copy then
hits ENOSPC, often re-raised as an unrelated load error. These tests pin the redirect
and the ways it could be silently undone (TMPDIR pointing back at /tmp, or the guest dir
not existing inside a --read-only sandbox), and the isolation between containers: one
shared scratch root mounted into every container let a job swap a directory for a link
and redirect the next job's TMPDIR.
"""

import os

import pytest

from gpuworker.launcher import (
    SCRATCH_GUEST,
    TMPDIR_GUEST,
    DockerLauncher,
    LaunchSpec,
    _scratch_slug,
    scratch_dir,
)
from gpuworker.module import JobType


def _jt(**kw):
    base = dict(
        id="ex.model",
        resource_class="vram",
        vram_mb=5000,
        env="container",
        entrypoint=["python", "-m", "w"],
        image="img@sha256:" + "a" * 64,
        sandbox="standard",
    )
    base.update(kw)
    return JobType(**base)


def _env_of(argv):
    return dict(
        kv.split("=", 1) for i, kv in enumerate(argv) if i and argv[i - 1] == "--env" and "=" in kv
    )


def _scratch_mount(argv):
    vols = [argv[i + 1] for i, a in enumerate(argv) if a == "--volume"]
    (vol,) = [v for v in vols if v.endswith(f":{SCRATCH_GUEST}:rw")]
    return vol.rsplit(":", 2)[0]


def _spec(name):
    return LaunchSpec(_jt(), bundle_dir="/b", out_dir="/o", name=name)


def test_tmpdir_is_on_scratch_not_tmpfs(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUWORKER_SCRATCH_DIR", str(tmp_path / "scratch"))
    env = _env_of(DockerLauncher().build_argv(_spec("job-abc")))
    assert env["TMPDIR"] == TMPDIR_GUEST
    assert env["TMPDIR"].startswith(SCRATCH_GUEST + "/")
    assert not env["TMPDIR"].startswith("/tmp")  # never the RAM-backed tmpfs


def test_each_container_mounts_its_own_dir_not_the_root(tmp_path, monkeypatch):
    root = tmp_path / "scratch"
    monkeypatch.setenv("GPUWORKER_SCRATCH_DIR", str(root))
    a = _scratch_mount(DockerLauncher().build_argv(_spec("job-a")))
    b = _scratch_mount(DockerLauncher().build_argv(_spec("job-b")))
    assert a != b
    assert os.path.abspath(str(root)) not in (a, b)
    assert a == os.path.abspath(str(root / "job-a"))


def test_tmpdir_host_dir_is_created(tmp_path, monkeypatch):
    """A --read-only sandbox cannot mkdir its own TMPDIR, so the host must pre-create it."""
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("GPUWORKER_SCRATCH_DIR", str(scratch))
    DockerLauncher().build_argv(_spec("job-abc"))
    assert (scratch / "job-abc" / "tmp").is_dir()


def test_scratch_starts_empty_and_cleanup_removes_it(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUWORKER_SCRATCH_DIR", str(tmp_path / "scratch"))
    spec = _spec("gpuworker-serve-ex.m-1")  # serve names repeat across broker restarts
    launcher = DockerLauncher()
    launcher.build_argv(spec)
    left = os.path.join(scratch_dir(spec), "tmp", "left.so")
    with open(left, "w") as fh:
        fh.write("from an earlier container")
    launcher.build_argv(spec)  # the next launch under the same name
    assert not os.path.exists(left)
    launcher.cleanup(spec)
    assert not os.path.exists(scratch_dir(spec))


def test_a_stray_link_is_replaced_not_followed(tmp_path, monkeypatch):
    root = tmp_path / "scratch"
    monkeypatch.setenv("GPUWORKER_SCRATCH_DIR", str(root))
    host_file = tmp_path / "host_file"
    host_file.mkdir()
    (host_file / "keep.txt").write_text("host data")
    root.mkdir()
    try:
        os.symlink(host_file, root / "job-x", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("cannot create symlinks here")
    DockerLauncher().build_argv(_spec("job-x"))
    assert (host_file / "keep.txt").read_text() == "host data"  # nothing removed through it
    assert not os.path.islink(root / "job-x") and (root / "job-x" / "tmp").is_dir()


def test_no_tmpdir_when_scratch_disabled(monkeypatch):
    """Scratch off -> no TMPDIR override; /tmp remains the default."""
    monkeypatch.setenv("GPUWORKER_SCRATCH_DIR", "")
    argv = DockerLauncher().build_argv(_spec("job-abc"))
    assert "TMPDIR" not in _env_of(argv)
    assert not any(a.endswith(f":{SCRATCH_GUEST}:rw") for a in argv)


def test_explicit_tmpdir_wins(tmp_path, monkeypatch):
    """setdefault, not force: a worker that pins its own TMPDIR keeps it."""
    monkeypatch.setenv("GPUWORKER_SCRATCH_DIR", str(tmp_path / "scratch"))
    spec = LaunchSpec(
        _jt(), bundle_dir="/b", out_dir="/o", name="job-abc", extra_env={"TMPDIR": "/job/out/mytmp"}
    )
    assert _env_of(DockerLauncher().build_argv(spec))["TMPDIR"] == "/job/out/mytmp"


def test_slug_never_empty_and_is_path_safe():
    assert _scratch_slug("") == "job"
    assert _scratch_slug("../../etc") == "etc"
    assert "/" not in _scratch_slug("a/b/c")
    assert _scratch_slug("ok-name_1.2") == "ok-name_1.2"
