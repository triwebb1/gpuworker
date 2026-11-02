"""Warm serve containers (SPEC 8).

The loop creates a :class:`ServeContainer` and starts :func:`run_serve_manager` on its
own thread. The manager owns that container's process and socket: it loads the model
once, streams queued jobs over the socket one at a time, recycles the container when
asked, and tears it down on stop or death. It never touches loop-owned state; it talks
to the loop only through ``sc.jobq`` / ``sc.stop`` (inbound) and epoch-stamped
completions (outbound), and it always ends by posting ``serve_gone``.
"""

from __future__ import annotations

import os
import queue
import shutil
import threading
from dataclasses import dataclass
from typing import Any

from . import bundle, gpu, module, serve
from .jobs import UNLOAD, Completion, PendingJob, WorkerContext, name_paths
from .launcher import Launcher, LaunchSpec
from .manifest import JobManifest
from .module import JobType

# The container log is emptied between jobs once it passes this size. Each job's slice
# has already gone into that job's worker log (MODULE_GUIDE 6.1).
SERVE_LOG_MAX_BYTES = 64 * 1024 * 1024


def serve_timeout(jm: JobManifest, jt: JobType) -> float:
    """Per-job serve timeout: the type's serve_job_timeout, or the manifest's timeout_s if
    that is shorter. Used for both the socket timeout and the scheduler's watchdog arm, so
    they cannot drift."""
    return min(jm.timeout_s, jt.serve_job_timeout) if jm.timeout_s else jt.serve_job_timeout


@dataclass
class ServeContainer:
    """One warm serve container. Created by the loop; its socket and process I/O are owned
    by a single manager thread. The loop and the manager share only the thread-safe
    ``jobq`` (loop puts jobs) and ``stop`` (loop signals teardown)."""

    jt: JobType
    epoch: int  # stamps every manager->loop message (stale-drop guard)
    name: str  # stable container name (kill/reap addressably)
    root: str  # its directory in the broker's state dir: bundle/ + out/ + sock/ subdirs
    jobq: queue.Queue
    stop: threading.Event
    # Loop-only: the manager reported an infrastructure failure, so it is tearing down.
    # The loop stops dispatching to it and waits for serve_gone before starting a new one.
    dying: bool = False
    thread: threading.Thread | None = None  # the manager thread (loop-set; joined at shutdown)
    # Free VRAM a recycle waits for before relaunching (the reclaim gate, SPEC 9). Written
    # only by the loop (at spawn and after footprint reconciliation) and only read by the
    # manager, so the manager never calls into loop-owned footprint state.
    respawn_free_target_mb: int = 0

    @property
    def bundle_root(self) -> str:
        return os.path.join(self.root, "bundle")  # mounted RO at GUEST_BUNDLE

    @property
    def out_root(self) -> str:
        return os.path.join(self.root, "out")  # mounted RW at GUEST_OUT

    @property
    def sockdir(self) -> str:
        return os.path.join(self.root, "sock")

    @property
    def host_socket(self) -> str:
        return os.path.join(self.sockdir, "job.sock")

    @property
    def log_path(self) -> str:
        return os.path.join(self.root, "serve.log")

    # ---- the container log, shared by every job this container serves ---------------

    def log_size(self) -> int:
        """Current length of the container log, or 0 if it isn't there yet."""
        try:
            return os.path.getsize(self.log_path)
        except OSError:
            return 0

    def cap_log(self, max_bytes: int | None = None) -> None:
        """Empty the container log if it is over *max_bytes* (SERVE_LOG_MAX_BYTES). Called
        between jobs only, so no job's slice is cut. The container writes in append mode,
        so its next line goes to the start of the emptied file."""
        limit = SERVE_LOG_MAX_BYTES if max_bytes is None else max_bytes
        if self.log_size() > limit:
            try:
                os.truncate(self.log_path, 0)
            except OSError:
                pass

    def log_slice(self, start: int, head: int = 256 * 1024, tail: int = 1024 * 1024) -> bytes:
        """The bytes appended to serve.log since *start*: one job's own output. Jobs run
        strictly one at a time per container, which makes the offset delta unambiguous.
        A slice longer than *head* + *tail* keeps its first *head* and last *tail* bytes,
        as for a oneshot worker's streams. Non-raising: no log, a shorter file, or a read
        error all yield b''."""
        try:
            with open(self.log_path, "rb") as fh:
                size = os.fstat(fh.fileno()).st_size
                if size < start:
                    return b""
                fh.seek(start)
                if size - start <= head + tail:
                    return fh.read(head + tail)
                first = fh.read(head)
                fh.seek(size - tail)
                omitted = size - start - head - tail
                note = f"\n[gpuworker: {omitted} bytes of output omitted]\n".encode()
                return first + note + fh.read(tail)
        except Exception:
            return b""

    def log_tail(self, n: int = 3000) -> str:
        """Last *n* bytes of the container log, for failure messages. Read it before
        teardown removes ``root``."""
        try:
            with open(self.log_path, "rb") as fh:
                try:
                    fh.seek(-n, os.SEEK_END)
                except OSError:
                    fh.seek(0)
                return fh.read().decode("utf-8", "replace").strip()
        except Exception:
            return ""

    # ---- per-job staging (each job gets its own key under the fixed mounts) ----------

    @staticmethod
    def workspace(pj: PendingJob) -> str:
        """The name of one attempt's directories under the mounts, sent as the run frame's
        job_id. The broker's uid and attempt, never the client's job_id: two bundles can
        use the same job_id, and one must never see what the other left behind."""
        return f"{pj.uid}-{pj.attempts}"

    def stage_in(self, pj: PendingJob) -> None:
        """Copy one job's input bundle to bundle_root/<key>/ and create out_root/<key>/,
        both new: a leftover directory with that name is an error, not something to reuse."""
        key = self.workspace(pj)
        dst = bundle.safe_join(self.bundle_root, key)
        if os.path.isdir(pj.batch.work_dir):
            shutil.copytree(pj.batch.work_dir, dst)
        else:
            os.makedirs(dst)
        os.makedirs(bundle.safe_join(self.out_root, key))

    def collect_out(self, pj: PendingJob, max_bytes: int | None = None) -> list[str]:
        """Copy declared outputs from out_root/<key>/ into the batch out dir, where the
        result bundle is packed from. Returns the relpaths found. The container is still
        running and owns out_root, so every path is opened without following symlinks
        (from out_root itself, so a replaced <key> dir is caught too); a symlink or
        non-regular output raises bundle.UnsafeMember, and one over *max_bytes* raises
        bundle.OutputTooLarge."""
        key = self.workspace(pj)
        found: list[str] = []
        for rel in pj.manifest.outputs:
            try:
                bundle.copy_regular(
                    self.out_root,
                    f"{key}/{rel}",
                    bundle.safe_join(pj.batch.out_dir, rel),
                    max_bytes,
                )
            except (FileNotFoundError, NotADirectoryError):
                continue
            found.append(rel)
        return found

    def cleanup_job(self, key: str) -> bool:
        """Remove one attempt's directories. Returns False if anything is left, in which
        case the caller retires the container rather than serve more jobs beside it."""
        ok = True
        for root in (self.bundle_root, self.out_root):
            path = bundle.safe_join(root, key)
            shutil.rmtree(path, ignore_errors=True)
            if os.path.lexists(path):
                ok = False
        return ok


def run_serve_manager(sc: ServeContainer, ctx: WorkerContext) -> None:
    """Manager thread body for one serve container."""
    jt = sc.jt
    launcher = ctx.launcher
    cfg = ctx.cfg
    proc: Any = None
    spec: LaunchSpec | None = None
    session: serve.ServeSession | None = None
    # Where the container log stood at the last launch, until a job takes it: the first
    # job after a launch also gets what the container printed while it loaded its model.
    load_from: int | None = None

    def fail_queued(msg: str, infra: bool = True, not_started: bool = False) -> None:
        # Every job still queued here never started: the container failed under it, so
        # these are infrastructure failures and eligible for a retry (unless *infra* is
        # False: a configuration error would fail the same way on every attempt). With
        # *not_started*, the container died on another job, so these spend no retry; a
        # load failure does not set it, because it would fail them all again.
        while True:
            try:
                item = sc.jobq.get_nowait()
            except queue.Empty:
                return
            if item is UNLOAD:
                continue
            ctx.completions.put(
                Completion(
                    item,
                    "failed",
                    msg,
                    [],
                    serve_epoch=sc.epoch,
                    infra=infra,
                    not_started=not_started,
                )
            )

    def launch_and_ready(spec: LaunchSpec) -> tuple[Any, serve.ServeSession]:
        """(Re)launch the container, connect, and wait for `ready`. Used for the first load
        and for every recycle: for a model that accumulates state across requests, a full
        process restart is the only reliable reset."""
        nonlocal load_from
        launcher.terminate_by_name(sc.name)  # clear a stale orphan
        try:
            os.unlink(sc.host_socket)  # a dead worker's bound socket would EADDRINUSE
        except OSError:
            pass
        load_from = sc.log_size()
        p = launcher.run(spec)
        try:
            conn = serve.connect(sc.host_socket, timeout=cfg.serve_connect_timeout_s)
            sess = serve.ServeSession(conn)
            sess.wait_ready(timeout=cfg.serve_ready_timeout_s)
        except Exception:  # a failed connect/ready must not leak the container
            launcher.terminate(p, spec)
            raise
        return p, sess

    try:
        # Configuration, not infrastructure: an unsealed weight cache or an unusable image
        # fails identically on every attempt, so these are final (as for oneshot jobs).
        weights_dir = ctx.weights_for(jt)
        image = module.resolve_image(jt, cfg.sandbox_image, cfg.allow_unpinned_images)
    except Exception as exc:
        fail_queued(f"serve config error: {exc}", infra=False)
        _teardown(sc, ctx, launcher, proc, spec)
        return
    try:
        for d in (sc.bundle_root, sc.out_root, sc.sockdir):
            os.makedirs(d, exist_ok=True)
        spec = LaunchSpec(
            job_type=jt,
            bundle_dir=sc.bundle_root,
            out_dir=sc.out_root,
            weights_dir=weights_dir,
            serve_socket_host=sc.host_socket,
            name=sc.name,
            extra_env=dict(jt.fixed_env),
            image=image,
            tmpfs_mb=cfg.tmpfs_mb,
            user=cfg.container_user,
            log_path=sc.log_path,
        )
        proc, session = launch_and_ready(spec)
    except Exception as exc:  # load/connect failed
        tail = sc.log_tail()
        msg = f"serve load failed: {exc}"
        if tail:
            msg += f"\n--- container log (tail) ---\n{tail}"
        fail_queued(msg)
        _teardown(sc, ctx, launcher, proc, spec)
        return

    jobs_since_reset = 0
    try:
        while not sc.stop.is_set():
            try:
                item = sc.jobq.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is UNLOAD:
                break
            pj: PendingJob = item
            jm = pj.manifest
            # Recycle before this job if it asks (options.reset_serve) or the per-type cap
            # fired (serve_max_jobs), but never on a container that has not run a job yet.
            # The ledger slot is kept throughout (no serve_gone).
            want_reset = bool((jm.options or {}).get("reset_serve")) or (
                jt.serve_max_jobs and jobs_since_reset >= jt.serve_max_jobs
            )
            if want_reset and jobs_since_reset > 0:
                # The loop bounds the reload as it bounds the first load.
                ctx.completions.put(
                    Completion(None, kind="serve_reloading", serve_epoch=sc.epoch, jt_id=jt.id)
                )
                try:
                    session.unload()
                except Exception:
                    pass
                try:
                    launcher.terminate(proc, spec)  # free VRAM before reloading
                    proc.wait(timeout=15)
                except Exception:
                    pass
                reclaimed = True
                try:
                    # Wait for the old model's VRAM to actually return before relaunching:
                    # the driver's release lags the process exit (the reclaim gate's race).
                    reclaimed = gpu.wait_for_free(
                        ctx.monitor, sc.respawn_free_target_mb, cfg.reclaim_timeout_s
                    )
                    if not reclaimed:
                        raise RuntimeError(
                            f"VRAM not reclaimed within {cfg.reclaim_timeout_s}s "
                            "before serve respawn"
                        )
                    proc, session = launch_and_ready(spec)
                    jobs_since_reset = 0
                    ctx.completions.put(
                        Completion(None, kind="serve_respawn", serve_epoch=sc.epoch, jt_id=jt.id)
                    )
                except Exception as exc:  # respawn failed = serve dead
                    rmsg = f"serve respawn failed: {exc}"
                    ctx.completions.put(
                        Completion(
                            pj,
                            "failed",
                            rmsg,
                            [],
                            serve_epoch=sc.epoch,
                            infra=True,
                            container_gone=True,
                            reclaim_timeout=not reclaimed,
                        )
                    )
                    fail_queued(rmsg)
                    break
            ctx.completions.put(Completion(pj, kind="serve_started", serve_epoch=sc.epoch))
            # A staging or collect error fails only this job; the container is healthy. A
            # socket error means the container is wedged or dead: fail the job and the
            # whole backlog, and tear down.
            fail_msg: str | None = None
            died: str | None = None
            timed_out = False
            abnormal = False
            oom = False
            cuda_error = False
            status, message, outputs = "failed", "", []
            if load_from is None:
                sc.cap_log()
                log_from = sc.log_size()
            else:
                log_from, load_from = load_from, None
            try:
                sc.stage_in(pj)
            except Exception as exc:
                fail_msg = f"serve stage-in failed: {exc}"
            if fail_msg is None:
                try:
                    res = session.run_job(
                        sc.workspace(pj), jm.to_dict(), timeout=serve_timeout(jm, jt)
                    )
                    status, message = res.status, res.message
                    if res.error == serve.ERROR_CUDA_OOM:
                        # The worker freed the job's memory and stays up (MODULE_GUIDE 3.2);
                        # the loop decides whether to retry the job on it.
                        oom = True
                        fail_msg = f"serve worker reported {res.error}: {res.message}"
                    elif res.error:
                        # The worker's CUDA state is suspect: replace the process and retry
                        # the job on a fresh one.
                        cuda_error = res.error == serve.ERROR_CUDA
                        died = f"serve worker reported {res.error}: {res.message}"
                        fail_msg = died
                except (serve.ProtocolError, OSError) as exc:
                    # A timeout also tears the container down, but the job overran: that is
                    # not an infrastructure failure, so it is not retried.
                    timed_out = isinstance(exc, serve.ServeTimeout)
                    abnormal = True
                    tail = sc.log_tail()
                    died = f"serve container error: {exc}"
                    fail_msg = died + (f"\n--- container log (tail) ---\n{tail}" if tail else "")
            if fail_msg is None:
                try:
                    outputs = sc.collect_out(pj, cfg.max_output_mb * 1024 * 1024)
                except Exception as exc:
                    fail_msg = f"serve collect-out failed: {exc}"
            found = set(outputs)
            missing = [rel for rel in jm.outputs if rel not in found]
            if fail_msg is None and status == "done" and missing:
                status, message = "failed", f"did not produce {name_paths(missing)}"
            # Ship this job's slice of the container log in its own bundle, before the
            # completion (the loop packs the result as soon as it drains it).
            ctx.write_worker_log(
                pj,
                {"stdout": sc.log_slice(log_from)},
                fail_msg or ("" if status == "done" else message),
            )
            if fail_msg is not None:
                ctx.completions.put(
                    Completion(
                        pj,
                        "failed",
                        fail_msg,
                        [],
                        serve_epoch=sc.epoch,
                        infra=(died is not None and not timed_out) or oom,
                        container_gone=died is not None,
                        cuda_oom=oom,
                        cuda_error=cuda_error,
                        abnormal_exit=abnormal,
                    )
                )
            else:
                ctx.completions.put(Completion(pj, status, message, outputs, serve_epoch=sc.epoch))
            if not sc.cleanup_job(sc.workspace(pj)) and died is None:
                died = "could not remove the job's directories from the serve container"
                ctx.log.event(jm.job_id, "serve_cleanup_failed", container=sc.name)
            jobs_since_reset += 1
            if died is not None:  # the container is gone: fail the backlog and exit
                fail_queued(died, not_started=True)
                break
    finally:
        try:
            if session is not None:
                session.unload()
        except Exception:
            pass  # the container is torn down below whatever the unload did
        fail_queued("serve container evicted", not_started=True)
        _teardown(sc, ctx, launcher, proc, spec)


def _teardown(
    sc: ServeContainer, ctx: WorkerContext, launcher: Launcher, proc: Any, spec: LaunchSpec | None
) -> None:
    """Stop the container (freeing VRAM) before posting serve_gone, so the loop's reclaim
    gate sees the memory actually return."""
    try:
        if proc is not None and spec is not None:
            launcher.terminate(proc, spec)
            try:
                proc.wait(timeout=10)
            except Exception:
                pass
        else:
            launcher.terminate_by_name(sc.name)
    except Exception:
        pass
    if spec is not None:
        launcher.cleanup(spec)
    shutil.rmtree(sc.root, ignore_errors=True)
    ctx.completions.put(Completion(None, kind="serve_gone", serve_epoch=sc.epoch, jt_id=sc.jt.id))
