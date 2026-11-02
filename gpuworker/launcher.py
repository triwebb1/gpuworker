"""Container launcher abstraction (SPEC 6.1).

Builds and runs the command line for a per-job sandbox. Podman (CDI GPU injection) is the
default; Docker is a drop-in. Arg-building is pure; ``run()`` is the only part that shells
out. :class:`LocalLauncher` runs the entrypoint as a plain subprocess, with no container.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Protocol

from .module import LANE_VRAM, JobType

# Fixed paths the worker expects inside the container.
GUEST_BUNDLE = "/job/bundle"
GUEST_OUT = "/job/out"
GUEST_WEIGHTS = "/job/weights"

# Persistent CUDA JIT cache, off by default (GPUWORKER_CUDA_CACHE_DIR=<path>): a host dir,
# with one subdirectory per job type, so compiled kernels outlive the /tmp tmpfs. A job
# loads what its cache holds, so no type sees kernels another type compiled. On WSL2 the
# driver keeps its own cache, but it can be evicted after an idle or a rebuild, and a cold
# JIT may exceed a job's timeout.
CUDA_CACHE_GUEST = "/cuda-cache"

# Local-SSD scratch for high-churn intermediate files: /tmp is a small RAM tmpfs and
# /job/out is copied out file by file. Each container gets its own fresh directory, removed
# after it exits (Launcher.cleanup), so no job can see another's scratch.
SCRATCH_GUEST = "/job/scratch"
TMPDIR_GUEST = SCRATCH_GUEST + "/tmp"


def _cuda_cache_host() -> str | None:
    """Host dir for the CUDA JIT cache, or None. Set it in the broker's launch environment,
    not an interactive shell, or a restart silently reverts it."""
    d = os.environ.get("GPUWORKER_CUDA_CACHE_DIR")
    return d or None


def _type_dir(jt_id: str) -> str:
    """A directory name for a job type id."""
    return "".join(c if c.isalnum() or c in "._-" else "-" for c in jt_id)


def _cpu_budget() -> int:
    """Cores a job may use: host cpus minus GPUWORKER_CPU_SPARE. Used for both the --cpus
    cap and GPUWORKER_CPU_BUDGET, so the cap and the worker's thread count agree."""
    spare = int(os.environ.get("GPUWORKER_CPU_SPARE", "2"))
    return max(1, (os.cpu_count() or 4) - spare)


def _mem_budget_mb() -> int:
    """System RAM (MB) a job should stay under: total minus GPUWORKER_MEM_SPARE_MB. Lets a
    worker size a subprocess pool by RAM as well as cores. Not a --memory cap: a large model's
    load can legitimately peak near total RAM."""
    reserve = int(os.environ.get("GPUWORKER_MEM_SPARE_MB", "3072"))
    try:
        total_mb = (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) // (1024 * 1024)
    except (ValueError, OSError, AttributeError):
        total_mb = 8192
    return max(512, total_mb - reserve)


def _scratch_slug(name: str) -> str:
    """Filesystem-safe directory name, never empty, so the mount is never the root itself."""
    safe = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in (name or ""))
    return safe.strip("-.") or "job"


def _scratch_host() -> str | None:
    """Host root for job scratch, or None. GPUWORKER_SCRATCH_DIR overrides it ("" disables).
    The root itself is never mounted; each container gets a fresh subdirectory."""
    d = os.environ.get("GPUWORKER_SCRATCH_DIR", os.path.expanduser("~/.cache/gpuworker-scratch"))
    return d or None


def scratch_dir(spec: LaunchSpec) -> str | None:
    """The host directory mounted at SCRATCH_GUEST for *spec*'s container, or None."""
    root = _scratch_host()
    return os.path.join(root, _scratch_slug(spec.name)) if root else None


def _fresh_scratch(path: str) -> None:
    """Create *path* empty with a tmp/ subdirectory: a --read-only sandbox cannot create
    its own TMPDIR, and a missing one makes every tempfile call fail. The root is never
    mounted, so no container can replace *path* with a link."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.islink(path):
        os.unlink(path)
    elif os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
    os.mkdir(os.path.join(path, "tmp"))


# Serve containers are named so they can be killed and reaped by name.
SERVE_NAME_PREFIX = "gpuworker-serve-"
# And oneshot job containers, which the broker removes itself (LaunchSpec.auto_remove).
# A broker adds its own id (job_name_prefix), so at start it can remove the ones it left
# running without touching another broker's or a `run-once`.
JOB_NAME_PREFIX = "gpuworker-job-"


# What Docker and Podman print when inspect finds no container by that name.
NO_SUCH_CONTAINER = ("no such container", "no such object")


def job_name_prefix(broker_id: str) -> str:
    return f"{JOB_NAME_PREFIX}{broker_id}-"


def serve_container_name(jt_id: str, epoch: int, prefix: str = SERVE_NAME_PREFIX) -> str:
    """Serve container name per (job type, epoch). The epoch stops an evicted container's
    async `rm -f <name>` from removing a same-type successor; reaping matches the prefix."""
    return f"{prefix}{jt_id.replace('/', '-')}-{epoch}"


@dataclass
class Mount:
    host: str
    guest: str
    ro: bool = True

    def as_arg(self) -> str:
        # Docker and podman treat a relative bind source as a named volume, not a host dir.
        return f"{os.path.abspath(self.host)}:{self.guest}:{'ro' if self.ro else 'rw'}"


@dataclass
class LaunchSpec:
    """Everything needed to launch one job's sandbox."""

    job_type: JobType
    bundle_dir: str  # host path, mounted RO at GUEST_BUNDLE
    out_dir: str  # host path, mounted RW at GUEST_OUT
    weights_dir: str | None = None  # host path, mounted RO at GUEST_WEIGHTS
    serve_socket_host: str | None = None  # host path of the bind-mounted unix socket
    extra_env: dict[str, str] = field(default_factory=dict)
    name: str = ""  # container name, for kill and remove
    image: str = ""  # resolved effective image (jt.image or shared sandbox)
    tmpfs_mb: int = 2048  # size cap of the /tmp tmpfs
    user: str = "1000:1000"  # uid:gid the container runs as (Config.container_user)
    log_path: str | None = None  # if set, stdout+stderr go to this file (see run)
    # Run with --rm. A oneshot job clears it, so its exit state can still be read
    # (oom_killed); cleanup then removes the container.
    auto_remove: bool = True


class Launcher(Protocol):
    def build_argv(self, spec: LaunchSpec) -> list[str]: ...
    def run(self, spec: LaunchSpec) -> subprocess.Popen: ...
    def terminate(self, proc: subprocess.Popen, spec: LaunchSpec) -> None: ...
    def terminate_by_name(self, name: str) -> None: ...
    def container_exists(self, name: str) -> bool: ...
    def oom_killed(self, name: str) -> bool | None: ...
    def reap_serve_orphans(self, prefix: str = ...) -> None: ...
    def reap_jobs(self, prefix: str = ..., exited_only: bool = ...) -> None: ...
    def cleanup(self, spec: LaunchSpec) -> None: ...


def uses_gpu(jt: JobType) -> bool:
    """Only the vram lane gets a GPU device; a cpu-lane container gets none."""
    return jt.resource_class == LANE_VRAM


class _ContainerLauncher:
    """Shared arg-building for Podman/Docker. Subclasses set the binary + GPU flag."""

    binary = "podman"

    def __init__(self, binary: str | None = None, gpu_index: int = 0):
        if binary:
            self.binary = binary
        # The NVML index (nvidia-smi order), which is also how CDI names devices and how
        # --gpus device= counts them, so the container gets the card the broker monitors.
        self.gpu_index = gpu_index

    def gpu_args(self) -> list[str]:
        return ["--device", f"nvidia.com/gpu={self.gpu_index}"]  # CDI (not --gpus)

    def _security_args(self, spec: LaunchSpec) -> list[str]:
        """Per-job hardening (SPEC 6.1), applied whatever the manifest asks (SPEC 6.3).
        /tmp is mounted `exec` because some models JIT-compile kernels there and dlopen
        them; the model already runs arbitrary code, so `noexec` would add little."""
        args = ["--rm"] if spec.auto_remove else []
        args += [
            "--user",
            spec.user,  # non-root; Config refuses uid 0
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--read-only",  # read-only rootfs
            "--tmpfs",
            f"/tmp:rw,exec,size={spec.tmpfs_mb}m",
            # Fork-bomb guard. Threads count against it, so it is sized for a serve
            # container that holds torch thread pools and starts a threaded subprocess.
            "--pids-limit",
            "4096",
        ]
        # Leave spare cores so one job cannot starve the broker or a concurrent lane.
        args += ["--cpus", str(_cpu_budget())]
        # Every job runs offline; --network none also neutralizes trust_remote_code fetches.
        args += ["--network", "none"]
        return args

    def _mounts(self, spec: LaunchSpec) -> list[Mount]:
        mounts = [
            Mount(spec.bundle_dir, GUEST_BUNDLE, ro=True),
            Mount(spec.out_dir, GUEST_OUT, ro=False),
        ]
        if spec.weights_dir:
            mounts.append(Mount(spec.weights_dir, GUEST_WEIGHTS, ro=True))
        if spec.serve_socket_host and spec.job_type.serve_socket:
            # The socket's directory, RW so the container can bind the socket.
            sock_dir = os.path.dirname(spec.serve_socket_host)
            guest_dir = os.path.dirname(spec.job_type.serve_socket)
            mounts.append(Mount(sock_dir, guest_dir, ro=False))
        cache_host = _cuda_cache_host()
        if cache_host:
            per_type = os.path.join(cache_host, _type_dir(spec.job_type.id))
            os.makedirs(per_type, exist_ok=True)
            mounts.append(Mount(per_type, CUDA_CACHE_GUEST, ro=False))
        scratch = scratch_dir(spec)
        if scratch:
            _fresh_scratch(scratch)
            mounts.append(Mount(scratch, SCRATCH_GUEST, ro=False))
        return mounts

    def _env_args(self, spec: LaunchSpec) -> list[str]:
        env = dict(spec.extra_env)
        if uses_gpu(spec.job_type):
            # Only the driver libraries CUDA and nvidia-smi need, for a runtime hook that
            # reads this. Docker's --gpus value says the same (DockerLauncher.gpu_args);
            # under Podman with CDI the generated spec decides which libraries are mounted.
            env.setdefault("NVIDIA_DRIVER_CAPABILITIES", "compute,utility")
            env.setdefault("NVIDIA_VISIBLE_DEVICES", str(self.gpu_index))
        else:
            # CUDA base images set NVIDIA_VISIBLE_DEVICES=all, which a runtime with the
            # NVIDIA hook as its default would honour even without a device flag.
            env["NVIDIA_VISIBLE_DEVICES"] = "void"
        env.setdefault("GPUWORKER_BUNDLE", GUEST_BUNDLE)
        env.setdefault("GPUWORKER_OUT", GUEST_OUT)
        env.setdefault("PYTHONUNBUFFERED", "1")  # stream progress lines live
        # Workers size thread pools to this, not nproc, which still reports every host core.
        env.setdefault("GPUWORKER_CPU_BUDGET", str(_cpu_budget()))
        env.setdefault("GPUWORKER_MEM_BUDGET_MB", str(_mem_budget_mb()))
        if _scratch_host():
            env.setdefault("GPUWORKER_SCRATCH", SCRATCH_GUEST)
            # Some libraries copy a shared object into a fresh mkdtemp on every init, which on
            # a long job fills the RAM tmpfs (ENOSPC, often re-raised as an unrelated load
            # error). Growing the tmpfs would spend host RAM. The scratch mount must not be
            # noexec, or dlopen fails.
            env.setdefault("TMPDIR", TMPDIR_GUEST)
        if spec.job_type.serve_socket:
            env.setdefault("GPUWORKER_SERVE_SOCKET", spec.job_type.serve_socket)  # SPEC 8
        if spec.weights_dir:
            # Offline weight discovery (SPEC 6.2).
            env.setdefault("HF_HOME", GUEST_WEIGHTS)
            env.setdefault("HF_HUB_OFFLINE", "1")
            env.setdefault("TRANSFORMERS_OFFLINE", "1")
        # Forward only the variables module.toml env_passthrough names; broker values win.
        for _k in spec.job_type.env_passthrough:
            _v = os.environ.get(_k)
            if _v:
                env.setdefault(_k, _v)
        if _cuda_cache_host():
            env["CUDA_CACHE_PATH"] = CUDA_CACHE_GUEST
            env.setdefault("CUDA_CACHE_MAXSIZE", "4294967296")
            env.setdefault("TORCHINDUCTOR_CACHE_DIR", CUDA_CACHE_GUEST + "/inductor")
        out: list[str] = []
        for k, v in env.items():
            out += ["--env", f"{k}={v}"]
        return out

    def build_argv(self, spec: LaunchSpec) -> list[str]:
        jt = spec.job_type
        if not jt.is_container:
            raise ValueError(f"{jt.id}: not a container env ({jt.env})")
        argv = [self.binary, "run"]
        if spec.name:
            argv += ["--name", spec.name]
        argv += self._security_args(spec)
        if uses_gpu(jt):
            argv += self.gpu_args()
        for m in self._mounts(spec):
            argv += ["--volume", m.as_arg()]
        argv += self._env_args(spec)
        image = spec.image or jt.image
        if not image:
            raise ValueError(f"{jt.id}: no image resolved for launch")
        argv += [image]
        argv += jt.entrypoint
        return argv

    def run(self, spec: LaunchSpec) -> subprocess.Popen:
        argv = self.build_argv(spec)
        if spec.log_path:
            # A long-lived container would fill an unread PIPE and block on write, so it
            # never sends `ready`. The child dups the fd, so the parent closes its copy.
            # Append mode, so a write after ServeContainer.cap_log empties the file lands
            # at its new end instead of past a hole.
            logf = open(spec.log_path, "ab")
            try:
                return subprocess.Popen(argv, stdout=logf, stderr=subprocess.STDOUT)
            finally:
                logf.close()
        return subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def terminate(self, proc: subprocess.Popen, spec: LaunchSpec) -> None:
        """SIGTERM the client, then force-remove the named container (SPEC 10)."""
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            pass
        if spec.name:
            # In case conmon outlived the client.
            self.terminate_by_name(spec.name)

    def terminate_by_name(self, name: str) -> None:
        """Force-remove a container by name."""
        try:
            subprocess.run(
                [self.binary, "rm", "-f", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
            )
        except Exception:
            pass

    def container_exists(self, name: str) -> bool:
        """True if a container named *name* is still there, or if the runtime does not
        answer: a stop is confirmed only when the runtime says there is no such container.
        A daemon that is restarting or a busy store makes inspect fail too."""
        try:
            r = subprocess.run(
                [self.binary, "container", "inspect", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=15,
                text=True,
            )
        except Exception:
            return True
        if r.returncode == 0:
            return True
        return not any(m in (r.stderr or "").lower() for m in NO_SUCH_CONTAINER)

    def oom_killed(self, name: str) -> bool | None:
        """Whether the runtime recorded the exited container *name* as OOM-killed, or None
        if it cannot say. Needs a container run without --rm (LaunchSpec.auto_remove)."""
        try:
            r = subprocess.run(
                [self.binary, "container", "inspect", "--format", "{{.State.OOMKilled}}", name],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=15,
                text=True,
            )
        except Exception:
            return None
        value = (r.stdout or "").strip().lower() if r.returncode == 0 else ""
        return {"true": True, "false": False}.get(value)

    def cleanup(self, spec: LaunchSpec) -> None:
        """Remove the container's scratch directory once the container is gone, and the
        container itself if it ran without --rm. Never raises: the next launch of the same
        name replaces a leftover."""
        if spec.name and not spec.auto_remove:
            self.terminate_by_name(spec.name)
        scratch = scratch_dir(spec)
        if scratch:
            shutil.rmtree(scratch, ignore_errors=True)

    def reap_serve_orphans(self, prefix: str = SERVE_NAME_PREFIX) -> None:
        """Force-remove leftover serve containers named *prefix**. Each broker reaps only its
        own prefix, so `run-once` never kills a live broker's warm containers."""
        try:
            out = subprocess.run(
                [self.binary, "ps", "-aq", "--filter", f"name={prefix}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=15,
                text=True,
            )
            ids = [x for x in (out.stdout or "").split() if x]
            if ids:
                subprocess.run(
                    [self.binary, "rm", "-f", *ids],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                )
        except Exception:
            pass

    def reap_jobs(self, prefix: str = JOB_NAME_PREFIX, exited_only: bool = True) -> None:
        """Remove oneshot containers named *prefix** that a crashed broker left (they run
        without --rm). With *exited_only*, only exited ones, so a running `run-once` job is
        never touched; a broker clears that for its own prefix, whose running containers
        still hold VRAM."""
        argv = [self.binary, "ps", "-aq", "--filter", f"name={prefix}"]
        if exited_only:
            argv += ["--filter", "status=exited"]
        try:
            out = subprocess.run(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=15,
                text=True,
            )
            ids = [x for x in (out.stdout or "").split() if x]
            if ids:
                subprocess.run(
                    [self.binary, "rm", "-f", *ids],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                )
        except Exception:
            pass


class PodmanLauncher(_ContainerLauncher):
    binary = "podman"


class DockerLauncher(_ContainerLauncher):
    binary = "docker"

    def gpu_args(self) -> list[str]:
        # Capabilities go in the --gpus value, which overrides NVIDIA_DRIVER_CAPABILITIES;
        # each field is quoted so its commas survive the CSV parse.
        return ["--gpus", f'"device={self.gpu_index}","capabilities=compute,utility"']


class LocalLauncher:
    """Run the entrypoint as a plain host subprocess (no container), for host-venv job
    types and the test suite. Provides no isolation."""

    def build_argv(self, spec: LaunchSpec) -> list[str]:
        return list(spec.job_type.entrypoint)

    # A local job has no isolation, so it gets only what an interpreter needs to start, not
    # the broker's environment (which may hold credentials).
    _ENV_ALLOW = (
        "PATH",
        "SYSTEMROOT",
        "SYSTEMDRIVE",
        "WINDIR",
        "TEMP",
        "TMP",
        "HOME",
        "USERPROFILE",
        # The user's name, not a secret: getpass.getuser() reads these, and torch's
        # compile cache calls it at import, so a host job without them fails on Windows.
        "USER",
        "USERNAME",
        "LOGNAME",
        "LANG",
        "LC_ALL",
        "PATHEXT",
        "COMSPEC",
    )

    def run(self, spec: LaunchSpec) -> subprocess.Popen:
        env = {k: v for k, v in os.environ.items() if k.upper() in self._ENV_ALLOW}
        env.update(spec.extra_env)
        if not uses_gpu(spec.job_type):
            # A cpu-lane host process is not counted in the VRAM ledger, so it is not given
            # the GPU. The code is trusted and could still ignore this; the scheduler also
            # skips attribution while one runs.
            env["CUDA_VISIBLE_DEVICES"] = ""
        env["GPUWORKER_BUNDLE"] = spec.bundle_dir
        env["GPUWORKER_OUT"] = spec.out_dir
        return subprocess.Popen(
            self.build_argv(spec),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def terminate(self, proc: subprocess.Popen, spec: LaunchSpec) -> None:
        try:
            proc.kill()
        except Exception:
            pass

    def terminate_by_name(self, name: str) -> None:
        """No-op: a local serve worker is a host process, killed via its Popen."""

    def container_exists(self, name: str) -> bool:
        return False  # a local worker is only its process

    def oom_killed(self, name: str) -> bool | None:
        return None

    def reap_serve_orphans(self, prefix: str = SERVE_NAME_PREFIX) -> None:
        """No-op: no containers to reap for the local launcher."""

    def reap_jobs(self, prefix: str = JOB_NAME_PREFIX, exited_only: bool = True) -> None:
        """No-op, as above."""

    def cleanup(self, spec: LaunchSpec) -> None:
        """No-op: a host process gets no scratch mount."""


def make_launcher(runtime: str, gpu_index: int = 0) -> Launcher:
    runtime = (runtime or "podman").lower()
    if runtime == "podman":
        return PodmanLauncher(gpu_index=gpu_index)
    if runtime == "docker":
        return DockerLauncher(gpu_index=gpu_index)
    if runtime == "local":
        return LocalLauncher()
    raise ValueError(f"unknown container runtime: {runtime!r}")
