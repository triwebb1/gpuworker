"""Records passed between the scheduler loop and its worker threads (SPEC 3).

The loop owns all scheduling state. Worker threads (a oneshot supervisor per job, a
manager per warm serve container) get a :class:`WorkerContext` of read-only
collaborators and report back only by putting a :class:`Completion` on the queue.
"""

from __future__ import annotations

import os
import queue
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import gpu
from .config import Config
from .launcher import JOB_NAME_PREFIX, Launcher, LaunchSpec
from .manifest import JobManifest
from .module import JobType
from .obs import JobLogger
from .share import QueueTree


def name_paths(paths: list[str], n: int = 5) -> str:
    """Up to *n* paths for a message, quoted and clipped."""
    shown = ", ".join(repr(p[:200]) for p in paths[:n])
    return shown + (f" and {len(paths) - n} more" if len(paths) > n else "")


def _new_uid() -> str:
    return uuid.uuid4().hex[:16]


@dataclass(eq=False)
class PendingJob:
    manifest: JobManifest
    jt: JobType
    batch: Batch
    attempts: int = 0  # times re-queued after an infrastructure failure (SPEC 10)
    requeues: int = 0  # of those, re-queues of a job that never started (no retry spent)
    # Broker-generated identity. The submitter's job_id is unique only within its own
    # bundle, so loop state, container names and attempt dirs are keyed by this instead.
    uid: str = field(default_factory=_new_uid)
    seq: int = 0  # intake order across all queues: the FIFO tie-break (SPEC 5)


@dataclass
class Batch:
    """One claimed bundle: its jobs, where they run, and the results collected so far.

    ``out_dir`` is broker-only: it holds promoted outputs and worker logs and is what the
    result bundle is packed from. A oneshot attempt writes into its own directory under
    ``run_dir`` instead (:meth:`attempt_dir`), so a retry can never pick up a dead
    attempt's files and a job never has write access to what the broker packs. All three
    live under ``root``, a directory of the broker's state dir, never in a queue tree."""

    queue: QueueTree
    filename: str
    work_dir: str
    out_dir: str
    jobs: list[PendingJob] = field(default_factory=list)
    results: dict[str, dict] = field(default_factory=dict)
    run_dir: str = ""
    root: str = ""  # removed whole when the batch retires; "" for test-built batches
    delivery_attempts: int = 0  # result writes that failed outright
    delivery_error: str = ""

    def __post_init__(self) -> None:
        if not self.run_dir:
            self.run_dir = self.out_dir + ".run"

    @property
    def done(self) -> bool:
        return bool(self.jobs) and len(self.results) == len(self.jobs)

    def attempt_dir(self, pj: PendingJob) -> str:
        return os.path.join(self.run_dir, f"{pj.uid}-{pj.attempts}")


@dataclass
class Completion:
    """A worker thread's report to the loop. ``kind="job"`` finishes one attempt of a
    job; the ``serve_*`` kinds are serve-container lifecycle messages."""

    pending: PendingJob | None
    status: str = ""
    message: str = ""
    outputs: list[str] = field(default_factory=list)
    peak_used_mb: int | None = None  # observed NVML peak during the run (vram lane)
    # "job" | "serve_started" | "serve_gone" | "serve_reloading" | "serve_respawn"
    kind: str = "job"
    serve_epoch: int | None = None  # [serve] which container this came from (stale guard)
    jt_id: str | None = None  # [serve_gone] container key (no pending)
    # The broker's infrastructure failed, not the job: launch error, OOM-kill, serve
    # load/death, reclaim-gate timeout. Only these are retried (SPEC 10).
    infra: bool = False
    # [serve] the container is being torn down (crash, protocol error, or timeout), so the
    # loop must stop dispatching to it even when the failure itself is not retried.
    container_gone: bool = False
    # [serve] a recycle's reclaim gate timed out: VRAM did not return, so the loop marks
    # the GPU degraded exactly as for the admission gate (SPEC 9).
    reclaim_timeout: bool = False
    # [serve] the job was still queued when its container went away, so it never ran. It
    # is re-queued without spending a retry: the failure belongs to another job.
    not_started: bool = False
    # The worker reported that CUDA ran out of memory (oneshot exit code, serve error
    # kind). Retried only if the job did not have the GPU to itself (SPEC 10).
    cuda_oom: bool = False
    # The worker reported another CUDA error. The loop checks the GPU's health before it
    # retries the job.
    cuda_error: bool = False
    # The worker timed out, was killed or died on a signal. For a vram job the loop then
    # checks the GPU's health, as after a CUDA error.
    abnormal_exit: bool = False
    # [oneshot] the attempt was stopped from outside its supervisor (watchdog, shutdown),
    # but its process did not exit, so its memory may still be in use.
    still_running: bool = False
    # Which attempt of the job this reports. Stamped at creation (in the reporting
    # thread), so a report that arrives after the job was re-queued is dropped as stale.
    attempt: int | None = None

    def __post_init__(self) -> None:
        if self.attempt is None and self.pending is not None:
            self.attempt = self.pending.attempts


class Attempt:
    """One oneshot attempt's process, shared by its supervisor thread and the loop so the
    watchdog or a shutdown can stop it (SPEC 10). Exactly one side reports the attempt:
    the supervisor, unless :meth:`stop` took it over first."""

    def __init__(self, number: int):
        self.number = number  # PendingJob.attempts when it was spawned
        self._lock = threading.Lock()
        self.proc: Any = None
        self.spec: LaunchSpec | None = None
        # The supervisor's captured streams, so a stop can ship them in the worker log.
        self.streams: dict[str, bytes] = {}
        self.stopped = False
        self.reported = False

    def launch(self, start: Callable[[], Any], spec: LaunchSpec) -> Any:
        """Start the process with *start*, unless the attempt was already stopped (then
        None). Under the lock, so a stop sees either no process or the one started."""
        with self._lock:
            if self.stopped:
                return None
            self.proc, self.spec = start(), spec
            return self.proc

    def claim_report(self) -> bool:
        """True once, for the supervisor, if no stop took the attempt over."""
        with self._lock:
            if self.stopped or self.reported:
                return False
            self.reported = True
            return True

    def stop(self) -> tuple[Any, LaunchSpec | None] | None:
        """Take the attempt over: the supervisor will neither start a process nor report.
        Returns (proc, spec) for the caller to end and report, or None if the supervisor
        has already reported or another stop got there first."""
        with self._lock:
            if self.reported or self.stopped:
                return None
            self.stopped = True
            return self.proc, self.spec


# Sentinel queued to a serve manager to tell it to unload + tear down.
UNLOAD = object()


@dataclass(frozen=True)
class WorkerContext:
    """What a worker thread may use. Everything here is safe to call off the loop; the
    loop's own state (ledger, pending list, serve registry) is deliberately absent."""

    cfg: Config
    launcher: Launcher
    monitor: gpu.GpuMonitor
    completions: queue.Queue[Completion]
    log: JobLogger
    weights_for: Callable[[JobType], str | None]
    write_worker_log: Callable[[PendingJob, dict[str, bytes], str], str | None]
    job_name_prefix: str = JOB_NAME_PREFIX  # oneshot container names (Scheduler.job_name_prefix)
