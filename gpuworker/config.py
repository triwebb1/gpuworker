"""Broker configuration (SPEC 11). Loaded from ``gpuworker.toml``."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field

# Numeric fields and their type; a float field also accepts an int.
_NUMBERS: dict[str, type] = {
    "gpu_budget_mb": int,
    "headroom_mb": int,
    "nvml_poll_ms": int,
    "reclaim_timeout_s": float,
    "serve_connect_timeout_s": float,
    "serve_ready_timeout_s": float,
    "serve_stage_slack_s": float,
    "tmpfs_mb": int,
    "footprint_variance_tol": float,
    "cpu_slots": int,
    "max_claims_per_tick": int,
    "max_bundle_members": int,
    "max_bundle_mb": int,
    "max_output_mb": int,
    "gpu_index": int,
    "default_timeout_s": int,
    "max_retries": int,
    "reclaim_after_s": float,
}
# Of those, the ones where 0 would mean nothing can ever run or every wait fails at once.
_POSITIVE = frozenset(_NUMBERS) - {
    "headroom_mb",
    "serve_stage_slack_s",
    "footprint_variance_tol",
    "gpu_index",
    "max_retries",
}
_STRINGS = (
    "registry_dir",
    "weights_dir",
    "state_dir",
    "footprints_path",
    "status_path",
    "sandbox_image",
)


@dataclass
class Config:
    # Total VRAM the broker is allowed to manage (MB), usually the card's physical
    # total. The margin under physical comes from headroom_mb below, applied once in
    # admit_budget_mb(); do not also pre-shrink this value or
    # a single heavy model (such as ~14 GB on a 16 GB card) becomes unschedulable.
    gpu_budget_mb: int = 16384
    headroom_mb: int = 1024
    nvml_poll_ms: int = 250
    reclaim_timeout_s: float = 30.0
    # Serve mode timeouts. connect = how long to wait for the worker to bind its socket
    # (it should bind early, before the heavy load; keep this short so a broken worker
    # fails fast). ready = how long to wait for its `ready` frame (covers the model load).
    serve_connect_timeout_s: float = 60.0
    serve_ready_timeout_s: float = 600.0  # heavy models: load + JIT + an end-to-end probe
    # Extra watchdog allowance for per-job serve stage-in (a copy of the job's unpacked
    # inputs into the container's own directory, both under state_dir) on top of the job's
    # own timeout, so a slow copy of a large bundle can't fail a healthy job (which would
    # drop running + let keep-warm evict a live container mid-job). Typical batches are
    # small, so this is a backstop.
    serve_stage_slack_s: float = 300.0
    # /tmp tmpfs size cap (MB) inside each container: holds JIT-compiled kernels, local
    # sockets and small scratch. On-demand (not pre-allocated); heavier serve stacks need
    # more than 512.
    tmpfs_mb: int = 2048

    # Where modules are discovered and queues are drained.
    registry_dir: str = "modules"  # scans modules/<name>/module.toml
    queue_roots: list[str] = field(default_factory=list)
    weights_dir: str = "weights"
    # The broker's own working files: unpacked bundles, job output dirs, serve container
    # roots. Must be outside every queue tree, which clients can write (SPEC 6.3). Owned by
    # one broker: `serve` clears what a previous run left here.
    state_dir: str = "state"
    # Persisted observed VRAM footprints (reality-vs-spec reconciliation, SPEC 9).
    footprints_path: str = "footprints.json"
    # Live status snapshot (occupied/free VRAM + per-module memory), refreshed each loop.
    status_path: str = "status.json"
    # Fractional overshoot over the declared footprint before logging a variance
    # warning (0.10 = 10%).
    footprint_variance_tol: float = 0.10

    # Lane concurrency.
    cpu_slots: int = 2

    # Bound on bundles claimed+unpacked per scheduler tick (keeps the single-writer
    # loop responsive under a flooded inbox, SPEC 3).
    max_claims_per_tick: int = 8
    # Intake limits per bundle: member count and bytes actually extracted (SPEC 6.3).
    max_bundle_members: int = 10_000
    max_bundle_mb: int = 8192
    # Largest single output file a job may return (SPEC 6.3); a larger one fails the job.
    max_output_mb: int = 8192
    # Optional per-queue priority ceiling, keyed by the queue root as written in
    # queue_roots. A job above it is lowered to it at intake (and logged).
    max_priority: dict[str, int] = field(default_factory=dict)

    # Runtime.
    container_runtime: str = "podman"  # podman | docker | local
    # uid:gid every container runs as. Match the owner of the weights and state dirs on
    # the host; root (uid 0) is refused.
    container_user: str = "1000:1000"
    gpu_index: int = 0
    # The single shared sandbox image all container job types inherit when they don't
    # declare their own `image` (one image, per-module venvs inside). During bring-up
    # this is a tag; pin it to a digest once the image is built.
    sandbox_image: str = "gpuworker/sandbox:cuda12.8"
    # Permit unpinned (tag) images for dev bring-up. Not for normal use, and never
    # for `strict` job types (RCE-class code on a mutable tag). Default off.
    allow_unpinned_images: bool = False
    # With container_runtime = "local", run vram-lane and strict container job types as
    # plain host processes anyway. They load model code, so this is development only.
    allow_unsandboxed_containers: bool = False
    # With container_runtime podman or docker, also run job types declared env = "host"
    # (such as echo.echo) as host processes. They run with no sandbox, so outside dev
    # bring-up a container broker runs only containers. Default off.
    allow_host_jobs: bool = False

    # Reliability.
    default_timeout_s: int = 1800
    # Automatic retries for infrastructure failures only (SPEC 10); 0 disables.
    max_retries: int = 1
    reclaim_after_s: float = 7200.0  # dead-worker batch reclaim

    def __post_init__(self):
        # Every field is checked here, so a mistyped toml fails at load with the field's
        # name instead of as a TypeError deep in scheduling, and a zero capacity cannot
        # leave jobs that can never run.
        for name, kind in _NUMBERS.items():
            v = getattr(self, name)
            ok = isinstance(v, int) if kind is int else isinstance(v, (int, float))
            if isinstance(v, bool) or not ok or not math.isfinite(v):
                raise ValueError(f"{name} must be a finite {kind.__name__}, got {v!r}")
            if name in _POSITIVE and v <= 0:
                raise ValueError(f"{name} must be greater than 0, got {v!r}")
            if v < 0:
                raise ValueError(f"{name} must not be negative, got {v!r}")
        for name in _STRINGS:
            v = getattr(self, name)
            if not isinstance(v, str) or (not v and name != "status_path"):
                raise ValueError(f"{name} must be a non-empty string, got {v!r}")
        for name in ("allow_unpinned_images", "allow_unsandboxed_containers", "allow_host_jobs"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be true or false")
        uid, _, gid = str(self.container_user).partition(":")
        if (
            not isinstance(self.container_user, str)
            or not (uid.isdigit() and gid.isdigit())
            or int(uid) == 0
        ):
            raise ValueError(
                f"container_user must be uid:gid with a non-zero uid, got {self.container_user!r}"
            )
        if self.container_runtime not in ("podman", "docker", "local"):
            raise ValueError(
                f"container_runtime must be podman, docker or local, got {self.container_runtime!r}"
            )
        if not isinstance(self.queue_roots, list) or not all(
            isinstance(r, str) and r for r in self.queue_roots
        ):
            raise ValueError("queue_roots must be a list of non-empty strings")
        if not isinstance(self.max_priority, dict) or not all(
            isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool)
            for k, v in self.max_priority.items()
        ):
            raise ValueError("max_priority must map queue roots to ints")
        # A job on the default timeout must fit the reclaim window. The check against each
        # batch's summed timeouts runs at intake (batch_may_outlive_reclaim_window).
        if self.reclaim_after_s <= self.default_timeout_s + 60:
            raise ValueError(
                f"reclaim_after_s ({self.reclaim_after_s}) must exceed "
                f"default_timeout_s + 60 ({self.default_timeout_s + 60})"
            )
        if self.admit_budget_mb() <= 0:
            raise ValueError(
                f"admit_budget_mb() = {self.admit_budget_mb()} <= 0 (gpu_budget_mb "
                f"{self.gpu_budget_mb} - headroom {self.headroom_mb}); no vram job could "
                "ever be admitted"
            )

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def load(cls, path: str) -> Config:
        with open(path, "rb") as fh:
            return cls.from_dict(tomllib.load(fh))

    def admit_budget_mb(self) -> int:
        """Effective VRAM ceiling for vram-lane admission: the budget minus headroom."""
        return self.gpu_budget_mb - self.headroom_mb
