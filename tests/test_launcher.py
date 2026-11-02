import pytest

from gpuworker.launcher import (
    GUEST_BUNDLE,
    GUEST_OUT,
    GUEST_WEIGHTS,
    DockerLauncher,
    LaunchSpec,
    PodmanLauncher,
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


def test_podman_uses_cdi_not_gpus(tmp_path):
    spec = LaunchSpec(_jt(), bundle_dir=str(tmp_path / "b"), out_dir=str(tmp_path / "o"))
    argv = PodmanLauncher().build_argv(spec)
    assert argv[0] == "podman"
    assert "--device" in argv and "nvidia.com/gpu=0" in argv
    assert "--gpus" not in argv


def test_docker_uses_gpus():
    spec = LaunchSpec(_jt(), bundle_dir="/b", out_dir="/o")
    argv = DockerLauncher().build_argv(spec)
    assert argv[0] == "docker"
    assert "--gpus" in argv
    # compute and utility only: no graphics or video driver libraries
    gpus_val = argv[argv.index("--gpus") + 1]
    assert gpus_val.endswith('"capabilities=compute,utility"')


def test_gpu_index_selects_the_device_the_broker_monitors():
    spec = LaunchSpec(_jt(), bundle_dir="/b", out_dir="/o")
    podman = PodmanLauncher(gpu_index=1).build_argv(spec)
    assert "nvidia.com/gpu=1" in podman
    assert "NVIDIA_VISIBLE_DEVICES=1" in _envs(podman)
    docker = DockerLauncher(gpu_index=1).build_argv(spec)
    assert docker[docker.index("--gpus") + 1].startswith('"device=1",')


def test_cpu_lane_containers_get_no_gpu():
    spec = LaunchSpec(_jt(resource_class="cpu", vram_mb=0), bundle_dir="/b", out_dir="/o")
    for launcher in (PodmanLauncher(), DockerLauncher()):
        argv = launcher.build_argv(spec)
        assert "--device" not in argv and "--gpus" not in argv
        assert "NVIDIA_VISIBLE_DEVICES=void" in _envs(argv)
        assert not any(e.startswith("NVIDIA_DRIVER_CAPABILITIES=") for e in _envs(argv))


def test_security_hardening_present():
    spec = LaunchSpec(_jt(), bundle_dir="/b", out_dir="/o")
    argv = PodmanLauncher().build_argv(spec)
    for flag in [
        "--rm",
        "--cap-drop",
        "ALL",
        "no-new-privileges",
        "--read-only",
        "--network",
        "none",
    ]:
        assert flag in argv
    assert argv[argv.index("--user") + 1] == "1000:1000"  # non-root


def test_the_container_user_comes_from_the_spec():
    """It was fixed at 1000:1000, so a host whose weights and state dirs belong to another
    uid could not run containers without changing ownership."""
    spec = LaunchSpec(_jt(), bundle_dir="/b", out_dir="/o", user="2000:2000")
    for launcher in (PodmanLauncher(), DockerLauncher()):
        argv = launcher.build_argv(spec)
        assert argv[argv.index("--user") + 1] == "2000:2000"


def test_mounts_bundle_ro_and_out_rw():
    import os

    spec = LaunchSpec(_jt(), bundle_dir="/host/b", out_dir="/host/o")
    argv = PodmanLauncher().build_argv(spec)
    vols = [argv[i + 1] for i, a in enumerate(argv) if a == "--volume"]
    # sources are absolutized (abspath is a no-op on POSIX abs paths at runtime)
    assert f"{os.path.abspath('/host/b')}:{GUEST_BUNDLE}:ro" in vols
    assert f"{os.path.abspath('/host/o')}:{GUEST_OUT}:rw" in vols


def test_serve_socket_env_set():
    jt = _jt(serve_socket="/run/job.sock", modes=("serve",))
    spec = LaunchSpec(jt, bundle_dir="/b", out_dir="/o")
    argv = PodmanLauncher().build_argv(spec)
    envs = [argv[i + 1] for i, a in enumerate(argv) if a == "--env"]
    assert "GPUWORKER_SERVE_SOCKET=/run/job.sock" in envs


def test_weights_offline_env_and_mount():
    import os

    spec = LaunchSpec(_jt(), bundle_dir="/b", out_dir="/o", weights_dir="/host/w")
    argv = PodmanLauncher().build_argv(spec)
    vols = [argv[i + 1] for i, a in enumerate(argv) if a == "--volume"]
    envs = [argv[i + 1] for i, a in enumerate(argv) if a == "--env"]
    assert f"{os.path.abspath('/host/w')}:{GUEST_WEIGHTS}:ro" in vols
    assert f"HF_HOME={GUEST_WEIGHTS}" in envs
    assert "HF_HUB_OFFLINE=1" in envs
    assert "TRANSFORMERS_OFFLINE=1" in envs


def test_relative_mount_sources_are_made_absolute():
    # docker/podman treat a relative -v source as a named volume, not a host dir, so a
    # relative weights_dir like "weights/<jt>" must be absolutized (else the runtime
    # fails with "invalid characters for a local volume name").
    import os

    spec = LaunchSpec(
        _jt(), bundle_dir="/abs/b", out_dir="/abs/o", weights_dir="weights/example.generate"
    )
    argv = PodmanLauncher().build_argv(spec)
    vols = [argv[i + 1] for i, a in enumerate(argv) if a == "--volume"]
    wvol = [v for v in vols if v.endswith(f":{GUEST_WEIGHTS}:ro")][0]
    host = wvol.rsplit(":", 2)[0]
    assert os.path.isabs(host)  # not a relative named-volume source
    assert host == os.path.abspath("weights/example.generate")
    # already-absolute mounts pass through unchanged at runtime (abspath is a no-op on
    # POSIX abs paths on Linux; on a Windows test host abspath normalizes them)
    assert f"{os.path.abspath('/abs/b')}:{GUEST_BUNDLE}:ro" in vols


def test_gpu_containers_get_compute_capabilities_only():
    # The base CUDA image would accept any capability; the vram lane needs CUDA and
    # nvidia-smi, so nothing else is requested.
    argv = PodmanLauncher().build_argv(LaunchSpec(_jt(), bundle_dir="/b", out_dir="/o"))
    assert "NVIDIA_DRIVER_CAPABILITIES=compute,utility" in _envs(argv)


def test_image_is_last_before_entrypoint():
    spec = LaunchSpec(_jt(), bundle_dir="/b", out_dir="/o")
    argv = PodmanLauncher().build_argv(spec)
    img_idx = argv.index("img@sha256:" + "a" * 64)
    assert argv[img_idx + 1 :] == ["python", "-m", "w"]


def test_spec_image_overrides_jt_image():
    # When the broker resolves the shared sandbox image, it passes it via spec.image.
    spec = LaunchSpec(
        _jt(), bundle_dir="/b", out_dir="/o", image="gpuworker/sandbox:cuda12.8@sha256:" + "c" * 64
    )
    argv = PodmanLauncher().build_argv(spec)
    assert "gpuworker/sandbox:cuda12.8@sha256:" + "c" * 64 in argv
    assert ("img@sha256:" + "a" * 64) not in argv  # jt.image not used


def test_container_name_is_addressable():
    spec = LaunchSpec(_jt(), bundle_dir="/b", out_dir="/o", name="gpuworker-j1")
    argv = PodmanLauncher().build_argv(spec)
    assert "--name" in argv
    assert argv[argv.index("--name") + 1] == "gpuworker-j1"


def _envs(argv):
    return [argv[i + 1] for i, a in enumerate(argv) if a == "--env"]


def test_env_passthrough_forwards_only_declared_names(monkeypatch):
    # A job type names the broker-env variables it wants; nothing else leaks in, an
    # unset name is skipped, and a broker-set value is never overridden.
    monkeypatch.setenv("MODEL_THREADS", "4")
    monkeypatch.setenv("UNDECLARED_KNOB", "x")
    monkeypatch.delenv("MODEL_DTYPE", raising=False)
    monkeypatch.setenv("PYTHONUNBUFFERED", "0")
    jt = _jt(env_passthrough=("MODEL_THREADS", "MODEL_DTYPE", "PYTHONUNBUFFERED"))
    envs = _envs(PodmanLauncher().build_argv(LaunchSpec(jt, bundle_dir="/b", out_dir="/o")))
    assert "MODEL_THREADS=4" in envs
    assert not any(e.startswith(("UNDECLARED_KNOB=", "MODEL_DTYPE=")) for e in envs)
    assert "PYTHONUNBUFFERED=1" in envs and "PYTHONUNBUFFERED=0" not in envs


def test_no_env_passthrough_by_default(monkeypatch):
    monkeypatch.setenv("MODEL_THREADS", "4")
    envs = _envs(PodmanLauncher().build_argv(LaunchSpec(_jt(), bundle_dir="/b", out_dir="/o")))
    assert not any(e.startswith("MODEL_THREADS=") for e in envs)


def test_serve_name_prefix_scopes_names_and_reaping(monkeypatch):
    # run-once uses its own prefix so its startup reap can never remove a live broker's
    # warm containers (docker/podman name filters are substring matches).
    import subprocess

    from gpuworker import launcher as L

    assert L.serve_container_name("ex.m", 3) == "gpuworker-serve-ex.m-3"
    assert L.serve_container_name("ex.m", 3, "gpuworker-runonce-serve-") == (
        "gpuworker-runonce-serve-ex.m-3"
    )
    assert "gpuworker-serve-" not in "gpuworker-runonce-serve-ex.m-3"
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="")

    monkeypatch.setattr(L.subprocess, "run", fake_run)
    PodmanLauncher().reap_serve_orphans("gpuworker-runonce-serve-")
    assert calls[0][-1] == "name=gpuworker-runonce-serve-"


def test_the_cuda_cache_is_scoped_per_job_type(monkeypatch, tmp_path):
    """One shared cache let one job type load compiled kernels another type had left."""
    import os

    from gpuworker.launcher import CUDA_CACHE_GUEST

    monkeypatch.setenv("GPUWORKER_CUDA_CACHE_DIR", str(tmp_path / "cache"))
    vols = {}
    for jt_id in ("ex.one", "ex.two"):
        argv = PodmanLauncher().build_argv(LaunchSpec(_jt(id=jt_id), bundle_dir="/b", out_dir="/o"))
        (vol,) = [
            argv[i + 1]
            for i, a in enumerate(argv)
            if a == "--volume" and CUDA_CACHE_GUEST in argv[i + 1]
        ]
        vols[jt_id] = vol
    assert vols["ex.one"] != vols["ex.two"]
    assert os.path.isdir(tmp_path / "cache" / "ex.one")


def test_a_cpu_lane_host_job_is_not_given_the_gpu():
    """The ledger does not count a cpu-lane job, so it gets no CUDA device; a vram-lane job
    run by the local launcher (development only) keeps it."""
    import sys

    from gpuworker.launcher import LocalLauncher

    code = "import os; print(repr(os.environ.get('CUDA_VISIBLE_DEVICES')))"
    for lane, expected in (("cpu", "''"), ("vram", "None")):
        jt = _jt(resource_class=lane, env="host", entrypoint=[sys.executable, "-c", code])
        proc = LocalLauncher().run(LaunchSpec(jt, bundle_dir=".", out_dir="."))
        out, _ = proc.communicate(timeout=30)
        assert out.decode().strip() == expected, lane
        assert proc.returncode == 0


def test_a_local_job_can_look_up_its_user_name(monkeypatch):
    """torch's compile cache calls getpass.getuser() at import, which reads USERNAME on
    Windows; the local launcher dropped it, so a torch host job died before it started."""
    import sys

    from gpuworker.launcher import LocalLauncher

    monkeypatch.setenv("USERNAME", "someone")
    monkeypatch.setenv("USER", "someone")
    monkeypatch.setenv("LOGNAME", "someone")  # getpass reads LOGNAME first
    code = "import getpass; print(getpass.getuser())"
    jt = _jt(resource_class="cpu", env="host", entrypoint=[sys.executable, "-c", code])
    proc = LocalLauncher().run(LaunchSpec(jt, bundle_dir=".", out_dir="."))
    out, err = proc.communicate(timeout=30)
    assert proc.returncode == 0, err
    assert out.decode().strip() == "someone"


def test_a_oneshot_container_keeps_its_exit_state_until_cleanup(monkeypatch):
    """With --rm the runtime threw away the OOMKilled flag, so exit 137 could only be
    reported as a likely OOM-kill."""
    import subprocess

    from gpuworker import launcher as L

    spec = LaunchSpec(_jt(), bundle_dir="/b", out_dir="/o", name="gpuworker-job-1-0")
    assert "--rm" in PodmanLauncher().build_argv(spec)
    spec.auto_remove = False
    assert "--rm" not in PodmanLauncher().build_argv(spec)

    calls, answer = [], {"stdout": "true\n", "rc": 0}

    def fake_run(argv, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, answer["rc"], stdout=answer["stdout"])

    monkeypatch.setattr(L.subprocess, "run", fake_run)
    docker = DockerLauncher()
    assert docker.oom_killed(spec.name) is True
    assert calls[-1][-3:] == ["--format", "{{.State.OOMKilled}}", spec.name]
    answer["stdout"] = "false\n"
    assert docker.oom_killed(spec.name) is False
    answer.update(stdout="", rc=1)  # no such container
    assert docker.oom_killed(spec.name) is None
    calls.clear()
    docker.cleanup(spec)
    assert calls == [["docker", "rm", "-f", spec.name]]
    calls.clear()
    spec.auto_remove = True
    docker.cleanup(spec)
    assert calls == []  # --rm removed it already
    answer.update(stdout="abc\n", rc=0)
    docker.reap_jobs()
    assert calls[0][-4:] == ["--filter", "name=gpuworker-job-", "--filter", "status=exited"]
    assert calls[1] == ["docker", "rm", "-f", "abc"]


def test_a_broker_removes_its_own_oneshot_containers_even_running(monkeypatch):
    """Only exited ones were reaped, so a container left running by a crashed broker kept
    its VRAM and the first reclaim gate after the restart marked the card degraded."""
    import subprocess

    from gpuworker import launcher as L

    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="abc\n")

    monkeypatch.setattr(L.subprocess, "run", fake_run)
    PodmanLauncher().reap_jobs(L.job_name_prefix("b0b0"), exited_only=False)
    assert calls[0] == ["podman", "ps", "-aq", "--filter", "name=gpuworker-job-b0b0-"]
    assert calls[1] == ["podman", "rm", "-f", "abc"]


@pytest.mark.parametrize(
    "rc, stderr, exists",
    [
        (0, "", True),
        (1, "Error: No such container: gpuworker-job-x\n", False),  # Docker
        (125, 'Error: no such container "gpuworker-job-x"\n', False),  # Podman
        (1, "Error response from daemon: No such object: gpuworker-job-x\n", False),
        (1, "Cannot connect to the Docker daemon. Is the docker daemon running?\n", True),
        (125, "Error: database is locked\n", True),
    ],
)
def test_a_container_counts_as_gone_only_when_the_runtime_says_so(monkeypatch, rc, stderr, exists):
    """Any failed inspect read as "gone", so a restarting daemon or a busy store let the
    broker free the slot of a worker that was still running."""
    import subprocess

    from gpuworker import launcher as L

    def fake_run(argv, **kw):
        return subprocess.CompletedProcess(argv, rc, stderr=stderr)

    monkeypatch.setattr(L.subprocess, "run", fake_run)
    assert DockerLauncher().container_exists("gpuworker-job-x") is exists
