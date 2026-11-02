"""Oneshot execution: one container (or host process) per job (SPEC 6.1, 10).

:func:`run_oneshot` runs on a supervisor thread, off the scheduler loop. Exactly one
:class:`~gpuworker.jobs.Completion` is posted per attempt whatever happens: by the
supervisor, or by :func:`stop_attempt` when the watchdog or a shutdown stops it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from typing import Any

from . import bundle, module
from .config import Config
from .jobs import Attempt, Completion, PendingJob, WorkerContext, name_paths
from .launcher import Launcher, LaunchSpec
from .manifest import JobManifest
from .module import JobType
from .worker_log import clip_both_ends

# Worker exit codes that mean SIGKILL from outside the job: 137 from a container runtime
# (128 + 9), -9 from a host subprocess. The broker's own timeout kill never reaches the
# exit-code check (it raises TimeoutError first), so these are usually OOM-kills. For a
# container the runtime's OOMKilled flag is read too; it can miss a kill by the host's OOM
# killer on cgroup v1, so its "false" is reported, not trusted.
KILLED_RETURNCODES = frozenset({137, -9})
# Docker and Podman exit 125 when the runtime itself failed (for example it could not create
# the container); the job never started. Only meaningful for container job types.
RUNTIME_ERROR_RETURNCODE = 125
# A worker exits with this code when CUDA ran out of memory (MODULE_GUIDE 3.6). The job may
# be fine on a card with more free memory, so the scheduler retries it if something else
# was using the GPU at the time.
CUDA_OOM_RETURNCODE = 75
# And this one after any other CUDA error. The next attempt runs in a fresh process, so
# it is retried too, once the scheduler has checked the GPU still answers.
CUDA_ERROR_RETURNCODE = 76


def died_on_signal(returncode: int) -> bool:
    """True for a worker ended by a signal: negative from a host subprocess, 128 + N from
    a container runtime (139 for SIGSEGV, 134 for SIGABRT)."""
    return returncode < 0 or 128 < returncode <= 128 + 64


# How much of each worker stream the broker keeps: the first STREAM_HEAD_BYTES and the last
# STREAM_TAIL_BYTES. The head holds the fingerprint line and the first error of a chained
# traceback, the tail holds the exception that ended the run; the worker log clips further.
STREAM_HEAD_BYTES = 256 * 1024
STREAM_TAIL_BYTES = 1024 * 1024

# How long a stopped worker's process gets to exit, at its timeout or when stop_attempt
# stops it, before the job is reported with the worker still running.
STOP_WAIT_S = 30.0


class WorkerTimeout(TimeoutError):
    """await_proc stopped a worker at its timeout. *still_running*: it had not exited
    STOP_WAIT_S later, or its container was still there."""

    def __init__(self, message: str, still_running: bool = False):
        super().__init__(message)
        self.still_running = still_running


def worker_gone(proc: Any, spec: LaunchSpec | None, launcher: Launcher, wait_s: float) -> bool:
    """After a terminate: True once the worker's process has exited within *wait_s* and
    its named container, if it has one, no longer exists. Both terminate implementations
    swallow their errors, so this is what tells the scheduler the memory is free."""
    try:
        proc.wait(timeout=wait_s)
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        pass
    if spec is not None and spec.name:
        return not launcher.container_exists(spec.name)
    return True


def oneshot_timeout(jm: JobManifest, jt: JobType, cfg: Config) -> float:
    """Per-job oneshot timeout: the type's oneshot_job_timeout_s, else the broker default,
    or the manifest's timeout_s if that is shorter. The scheduler arms its watchdog from
    the same value, so the two cannot drift apart."""
    bound = jt.oneshot_job_timeout_s or cfg.default_timeout_s
    return min(jm.timeout_s, bound) if jm.timeout_s else bound


def run_oneshot(
    pj: PendingJob, ctx: WorkerContext, sample_vram: bool, attempt: Attempt | None = None
) -> None:
    """Run one job to completion and post its Completion. *sample_vram* polls NVML for
    the run's peak (the scheduler sets it when the job is the sole resident, so the peak
    is attributable for footprint reconciliation). *attempt* is the handle the scheduler
    keeps to stop the job; once stopped, this thread starts nothing and reports nothing."""
    jm, jt = pj.manifest, pj.jt
    attempt = attempt or Attempt(pj.attempts)
    posted = False
    peak_used: dict[str, int | None] = {"mb": None}
    # Filled by await_proc as soon as the worker's streams exist (including after a
    # timeout kill) so post() can ship them to the client whatever happened. Shared with
    # the attempt, so a stop from outside can ship them too.
    streams = attempt.streams

    def post(
        status: str,
        message: str,
        outputs: list[str],
        infra: bool = False,
        oom: bool = False,
        cuda_error: bool = False,
        abnormal: bool = False,
        still_running: bool = False,
    ) -> None:
        nonlocal posted
        if not posted:
            posted = True
            if not attempt.claim_report():
                return  # stopped from outside: stop_attempt reports it
            # The worker log goes before the completion: the loop packs the result bundle
            # the moment it drains the completion, so a log written afterwards would miss it.
            ctx.write_worker_log(pj, streams, "" if status == "done" else message)
            ctx.completions.put(
                Completion(
                    pj,
                    status,
                    message,
                    outputs,
                    peak_used["mb"],
                    infra=infra,
                    cuda_oom=oom,
                    cuda_error=cuda_error,
                    abnormal_exit=abnormal,
                    still_running=still_running,
                )
            )

    # This attempt's own output dir. A retry gets a fresh one, so it can only succeed on
    # files it wrote itself; declared outputs are promoted into the broker-only batch out
    # dir after a clean exit.
    attempt_dir = pj.batch.attempt_dir(pj)
    spec: LaunchSpec | None = None
    try:
        timeout = oneshot_timeout(jm, jt, ctx.cfg)
        weights = ctx.weights_for(jt)
        # Shared sandbox unless the module pins its own; raises on an unpinned or strict
        # misconfiguration, which fails the job cleanly.
        image = module.resolve_image(jt, ctx.cfg.sandbox_image, ctx.cfg.allow_unpinned_images)
        shutil.rmtree(attempt_dir, ignore_errors=True)
        os.makedirs(attempt_dir)
        spec = LaunchSpec(
            job_type=jt,
            bundle_dir=pj.batch.work_dir,
            out_dir=attempt_dir,
            weights_dir=weights,
            extra_env={**jt.fixed_env, "GPUWORKER_JOB_ID": jm.job_id},
            # Named from the broker's ids, never the submitter's job_id, which could
            # collide across bundles or with a serve container's name. The attempt's own
            # number: pj.attempts moves on if the job is retried meanwhile.
            name=f"{ctx.job_name_prefix}{pj.uid}-{attempt.number}" if jt.is_container else "",
            image=image,
            tmpfs_mb=ctx.cfg.tmpfs_mb,
            user=ctx.cfg.container_user,
            auto_remove=False,  # removed in cleanup, after its OOMKilled flag is read
        )
        try:
            proc = attempt.launch(lambda: ctx.launcher.run(spec), spec)
        except Exception as exc:  # the runtime failed, not the job
            post("failed", f"launch error: {exc}", [], infra=True)
            return
        if proc is None:
            return  # stopped before it started
        stdout, stderr = await_proc(
            proc, spec, timeout, ctx, ctx.launcher, peak_used if sample_vram else None, streams
        )
        # A oneshot's own diagnostics would otherwise vanish on success; log a tail of each
        # non-empty stream to gpuworker.log without affecting the outcome.
        for stream_name, data in (("stdout", stdout), ("stderr", stderr)):
            text = (data or b"").decode("utf-8", "replace").strip()
            if text:
                ctx.log.event(
                    jm.job_id, "worker_out", stream=stream_name, tail=clip_both_ends(text, 2000)
                )
        if proc.returncode != 0:
            # The tail, not the head: the last lines of a traceback carry the exception.
            msg = (stderr or b"").decode("utf-8", "replace").strip()[-500:]
            if proc.returncode in KILLED_RETURNCODES:
                oom = ctx.launcher.oom_killed(spec.name) if spec.name else None
                why = {
                    True: "OOM-killed",
                    False: "the runtime recorded no OOM-kill",
                    None: "likely OOM-kill",
                }[oom]
                post(
                    "failed",
                    f"worker killed (exit {proc.returncode}, {why}): {msg}",
                    [],
                    infra=True,
                    abnormal=True,
                )
                return
            if jt.is_container and proc.returncode == RUNTIME_ERROR_RETURNCODE:
                post("failed", f"container runtime error (exit 125): {msg}", [], infra=True)
                return
            if proc.returncode == CUDA_OOM_RETURNCODE:
                post(
                    "failed",
                    f"worker ran out of CUDA memory (exit {CUDA_OOM_RETURNCODE}): {msg}",
                    [],
                    infra=True,
                    oom=True,
                )
                return
            if proc.returncode == CUDA_ERROR_RETURNCODE:
                post(
                    "failed",
                    f"worker hit a CUDA error (exit {CUDA_ERROR_RETURNCODE}): {msg}",
                    [],
                    infra=True,
                    cuda_error=True,
                )
                return
            if died_on_signal(proc.returncode):
                post("failed", f"worker exited {proc.returncode}: {msg}", [], abnormal=True)
                return
            raise RuntimeError(f"worker exited {proc.returncode}: {msg}")
        try:
            outputs = bundle.collect_outputs(
                attempt_dir, jm.outputs, pj.batch.out_dir, ctx.cfg.max_output_mb * 1024 * 1024
            )
        except (bundle.UnsafeMember, bundle.OutputTooLarge) as exc:
            raise RuntimeError(f"refused output: {exc}") from None
        found = set(outputs)
        missing = [rel for rel in jm.outputs if rel not in found]
        if missing:
            # Exit 0 without every declared output is not a success, whole or partial.
            raise RuntimeError(f"worker exited 0 but did not produce {name_paths(missing)}")
        post("done", "done", outputs)
    except TimeoutError as exc:  # await_proc stopped the worker at its timeout
        still = isinstance(exc, WorkerTimeout) and exc.still_running
        post("failed", str(exc), [], abnormal=True, still_running=still)
    except Exception as exc:
        post("failed", str(exc), [])
    finally:
        # Last-resort guarantee: never leave a job unreported (SPEC 10).
        post("failed", "worker thread ended without reporting", [])
        shutil.rmtree(attempt_dir, ignore_errors=True)
        if spec is not None:
            ctx.launcher.cleanup(spec)


def stop_attempt(
    pj: PendingJob,
    attempt: Attempt,
    ctx: WorkerContext,
    message: str,
    wait_s: float | None = None,
) -> None:
    """Stop a oneshot attempt from outside its supervisor (the watchdog, a shutdown) and
    report it failed with *message* once its process has exited, so the scheduler frees
    the job's ledger slot only when the memory is no longer in use. A process still there
    after *wait_s* (default STOP_WAIT_S) is reported with ``still_running``. The worker
    log holds whatever output the supervisor had captured. Does nothing if the supervisor
    has already reported."""
    wait_s = STOP_WAIT_S if wait_s is None else wait_s
    taken = attempt.stop()
    if taken is None:
        return
    proc, spec = taken
    gone = True
    if proc is not None and spec is not None:
        try:
            ctx.launcher.terminate(proc, spec)
        except Exception:
            pass
        gone = worker_gone(proc, spec, ctx.launcher, wait_s)
    ctx.write_worker_log(pj, dict(attempt.streams), message)
    ctx.completions.put(
        Completion(
            pj,
            "failed",
            message,
            [],
            abnormal_exit=True,
            still_running=not gone,
            attempt=attempt.number,
        )
    )


class _StreamReader:
    """Drain one pipe on its own thread, keeping only a bounded head and tail, so a worker
    that writes without limit can neither block on a full pipe nor fill the broker's
    memory."""

    def __init__(self, pipe: Any):
        self.head = bytearray()
        self.tail = bytearray()
        self.dropped = 0
        self._pipe = pipe
        self._thread: threading.Thread | None = None
        if pipe is not None:
            self._thread = threading.Thread(target=self._drain, daemon=True)
            self._thread.start()

    def _drain(self) -> None:
        read = getattr(self._pipe, "read1", self._pipe.read)
        try:
            while chunk := read(64 * 1024):
                room = STREAM_HEAD_BYTES - len(self.head)
                if room > 0:
                    self.head += chunk[:room]
                    chunk = chunk[room:]
                self.tail += chunk
                excess = len(self.tail) - STREAM_TAIL_BYTES
                if excess > 0:
                    del self.tail[:excess]
                    self.dropped += excess
        except (OSError, ValueError):  # the pipe was closed under us (teardown)
            pass

    def join(self, timeout: float) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def value(self) -> bytes:
        if not self.dropped:
            return bytes(self.head + self.tail)
        note = f"\n[gpuworker: {self.dropped} bytes of output omitted]\n".encode()
        return bytes(self.head) + note + bytes(self.tail)


def await_proc(
    proc: Any,
    spec: LaunchSpec,
    timeout: float,
    ctx: WorkerContext,
    launcher: Launcher,
    peak_used: dict[str, int | None] | None = None,
    streams: dict[str, bytes] | None = None,
) -> tuple[bytes, bytes]:
    """Wait for the worker. With *peak_used*, poll and record the peak NVML usage.
    Returns (stdout, stderr), each bounded to a head and a tail (STREAM_HEAD_BYTES,
    STREAM_TAIL_BYTES); raises WorkerTimeout on overrun, once the stopped worker has
    exited or STOP_WAIT_S has passed.

    Both pipes are drained on reader threads the whole time, so a worker that writes more
    than a pipe buffer cannot block and run into its timeout. *streams* receives the
    captured bytes as soon as the worker is gone, including after a timeout kill: a
    timed-out job is the one whose output you most want.
    """
    readers = (_StreamReader(proc.stdout), _StreamReader(proc.stderr))

    def collect(grace: float) -> tuple[bytes, bytes]:
        # The pipes close when the worker exits; the grace only bounds a grandchild that
        # kept one open.
        for r in readers:
            r.join(grace)
        out, err = readers[0].value(), readers[1].value()
        if streams is not None:
            streams["stdout"], streams["stderr"] = out, err
        return out, err

    poll_s = max(0.02, ctx.cfg.nvml_poll_ms / 1000.0) if peak_used is not None else timeout
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                proc.wait(timeout=max(0.0, min(poll_s, deadline - time.monotonic())))
                break
            except subprocess.TimeoutExpired:
                pass
            if time.monotonic() >= deadline:
                launcher.terminate(proc, spec)
                gone = worker_gone(proc, spec, launcher, STOP_WAIT_S)
                collect(5.0 if gone else 0.0)  # a worker still running keeps its pipes open
                raise WorkerTimeout(f"job exceeded timeout {timeout}s", still_running=not gone)
            if peak_used is not None:
                try:
                    used = ctx.monitor.used_mb()
                    if peak_used["mb"] is None or used > peak_used["mb"]:
                        peak_used["mb"] = used
                except Exception:  # a missed sample only weakens reconciliation
                    pass
    except TimeoutError:
        raise
    except Exception:
        # Anything else is not a timeout and must not be reported as one; still make sure
        # the worker does not outlive the failure.
        launcher.terminate(proc, spec)
        raise
    return collect(10.0)
