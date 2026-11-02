"""The broker scheduler (SPEC 3, 5, 9).

A single-writer loop owns all admission + VRAM-ledger state. Jobs run off the loop in
supervisor threads that only report terminal events back via a thread-safe queue; the
loop reacts and never blocks on a job. So a ``cpu`` job can run while a ``vram`` model is
resident, and the accounting needs no locks.

Ordering (SPEC 5): submitter priority first, then a job whose model is already loaded,
then the cold model with the most queued work times its load cost, then intake order.
Heavy vram jobs serialize; co-resident small models pack; cpu jobs run concurrently up to
cpu_slots.
"""

from __future__ import annotations

import datetime
import json
import os
import queue as _queue
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import replace

from . import bundle, gpu, worker_log
from . import oneshot as oneshot_mod
from . import weights as weights_mod
from .config import Config
from .footprints import Footprints
from .jobs import UNLOAD, Attempt, Batch, Completion, PendingJob, WorkerContext
from .launcher import (
    JOB_NAME_PREFIX,
    SERVE_NAME_PREFIX,
    Launcher,
    LocalLauncher,
    job_name_prefix,
    serve_container_name,
)
from .ledger import Ledger
from .lock import BrokerLock
from .manifest import RESERVED_OUTPUT_DIRS, STATUS_QUEUED, JobManifest, now_iso
from .module import LANE_CPU, LANE_VRAM, JobType, Registry
from .obs import JobLogger, write_status
from .oneshot import oneshot_timeout, run_oneshot, stop_attempt
from .serve_manager import ServeContainer, run_serve_manager, serve_timeout
from .share import ClaimRejected, QueueTree, discover_queues

# How often a result bundle that could not be written is retried, and how many failed
# writes it gets before the batch is given up (SPEC 10).
RESULT_RETRY_S = 30.0
RESULT_ATTEMPTS = 10
# Jobs a warm serve container is handed beyond the one it is running. The rest stay
# pending, so a later higher-priority job still goes ahead of them (SPEC 5); one queued
# job keeps the container from idling for a loop interval between jobs.
SERVE_QUEUE_AHEAD = 1
# A oneshot's watchdog fires this long after its timeout: the supervisor stops the worker
# at the timeout and may wait STOP_WAIT_S for it to exit, plus the runtime's own teardown.
ONESHOT_WATCHDOG_SLACK_S = oneshot_mod.STOP_WAIT_S + 60.0
# The longest an evicted serve container's teardown holds back a reclaim-gate deadline:
# its unload, stop and removal are each bounded, so this is only a backstop.
SERVE_TEARDOWN_BOUND_S = 120.0
# How often the loop asks the GPU monitor for hardware faults while nothing reports one.
HEALTH_POLL_S = 60.0
# Longest pause between loop iterations after repeated loop errors (the backoff cap).
TICK_ERROR_BACKOFF_MAX_S = 30.0

__all__ = ["Batch", "PendingJob", "Scheduler"]


def _inside(path: str, root: str) -> bool:
    path, root = os.path.realpath(path), os.path.realpath(root)
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _name_slug(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "-" for c in name)[:60] or "batch"


BROKER_ID_FILE = "broker_id"


def _broker_id(state_dir: str) -> str:
    """The id kept in *state_dir*, made on first use. A file that is unreadable or not an
    id is replaced, which only means a crashed run's containers are not found by name."""
    path = os.path.join(state_dir, BROKER_ID_FILE)
    try:
        with open(path, encoding="utf-8") as fh:
            found = fh.read().strip()
        if found.isalnum() and found.isascii() and len(found) <= 32:
            return found
    except OSError:
        pass
    new = uuid.uuid4().hex[:8]
    os.makedirs(state_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(new + "\n")
    return new


class Scheduler:
    def __init__(
        self,
        config: Config,
        registry: Registry,
        launcher: Launcher,
        monitor: gpu.GpuMonitor,
        logger: JobLogger | None = None,
        queues: list[QueueTree] | None = None,
        clock=time.monotonic,
        serve_name_prefix: str = SERVE_NAME_PREFIX,
    ):
        self.cfg = config
        self.serve_name_prefix = serve_name_prefix  # startup reaps only this prefix
        # Oneshot container names. A fresh id here; run() switches to the one kept in
        # state_dir, so a restarted broker can find the containers it left running.
        self.job_name_prefix = job_name_prefix(uuid.uuid4().hex[:8])
        self.reg = registry
        self.launcher = launcher
        self._local = LocalLauncher()  # for host job types
        self.monitor = monitor
        self.log = logger or JobLogger.null()
        self.clock = clock
        self.ledger = Ledger(config.admit_budget_mb(), config.cpu_slots)
        self.footprints = Footprints(config.footprints_path)
        self.queues = queues if queues is not None else discover_queues(config.queue_roots)
        # Working files live here, never in a client-writable queue tree (SPEC 6.3).
        self.state_dir = os.path.abspath(config.state_dir)
        for q in self.queues:
            if _inside(self.state_dir, q.root) or _inside(q.root, self.state_dir):
                raise ValueError(
                    f"state_dir {self.state_dir} and queue root {q.root} must not contain "
                    "each other: clients can write their queue tree"
                )
        self.pending: list[PendingJob] = []
        self.batches: list[Batch] = []
        self.completions: _queue.Queue[Completion] = _queue.Queue()
        self._stop = threading.Event()
        self._draining = False
        # Set by a reclaim gate timeout, a worker that did not stop or a GPU that did not
        # answer; holds vram admission until recovery (SPEC 9). Set through _degrade.
        self._gpu_degraded = False
        self._degraded_reason = ""
        # A hardware fault seen after a CUDA error. It keeps the GPU degraded until the
        # broker restarts: free memory says nothing about a failing card.
        self._gpu_fault = ""
        # Idle device-wide VRAM use (on WSL2 it includes the desktop); see _recover_degraded.
        self._used_baseline: int | None = None
        # Loop-thread only. Per-job state is keyed by PendingJob.uid, since a job_id is
        # unique only within its bundle.
        self._inflight: dict[str, tuple[PendingJob, float]] = {}  # watchdog deadline
        self._baseline: dict[str, int] = {}  # pre-load NVML used, for attribution
        # Parked at the reclaim gate: (job, free_target_mb, deadline).
        self._pending_reclaim: dict[str, tuple[PendingJob, int, float]] = {}
        self._inflight_epoch: dict[str, int] = {}  # serve epoch, so the watchdog spares a successor
        # Running oneshot attempts, so the watchdog and shutdown can stop their process.
        self._attempts: dict[str, tuple[PendingJob, Attempt]] = {}
        # (root, filename) of tracked batches: live, so reclaim_stale skips them.
        self._active_bundles: set[tuple[str, str]] = set()
        # (root, filename) requeued by reclaim_stale; intake scrubs every other trace.
        self._reclaimed: set[tuple[str, str]] = set()
        self._serve: dict[str, ServeContainer] = {}  # warm containers by jt.id
        self._serve_baseline: dict[str, int] = {}  # sole-resident NVML baseline
        self._serve_load_peak: dict[str, int] = {}  # loop-sampled peak (footprint)
        self._serve_epoch = 0
        # Evicted serve containers still tearing down (epoch -> when evicted); the reclaim
        # gate's deadline starts only once they are gone.
        self._tearing_down: dict[int, float] = {}
        # Serve containers still loading their model: jt.id -> (epoch, deadline). No job
        # watchdog is armed until the first job starts, so this one bounds the load.
        self._serve_loading: dict[str, tuple[int, float]] = {}
        self._reaped_orphans = False
        self._seq = 0  # intake counter: PendingJob.seq
        # (batch, next retry) for results that could not be written yet (SPEC 10).
        self._undelivered: list[tuple[Batch, float]] = []
        self._next_health_poll = 0.0
        # The last exception that escaped a loop iteration, for status.json.
        self.last_loop_error: dict | None = None

    def _publish_limits(self) -> None:
        """Write the broker limits a client needs, into each client's own queue tree.

        A client sizing bundles against reclaim_after_s reads it here instead of hard-coding
        it. Written by run(), not the constructor, so `gpuworker status` never touches client
        dirs; next to the queue dirs, never inside inbox/. Best-effort: a broker that cannot
        write it must still serve.
        """
        payload = {
            "schema": "gpuworker-broker-limits/v1",
            "generated_at": now_iso(),
            "reclaim_after_s": self.cfg.reclaim_after_s,
            "default_timeout_s": self.cfg.default_timeout_s,
            "job_type_timeouts_s": {
                jt.id: self._type_timeout(jt) for jt in self.reg.job_types.values()
            },
        }
        data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        for q in self.queues:
            try:
                q.ensure()
                bundle.write_new_bytes(q.root, "broker_limits.json", data)
            # Only OSError is expected; programming errors propagate and fail at startup.
            except OSError as exc:
                self.log.event(
                    "broker",
                    "publish_limits_failed",
                    queue=q.root,
                    error=f"{type(exc).__name__}: {exc}",
                )

    def _eff(self, jt: JobType) -> JobType:
        """*jt* with vram_mb set to the effective footprint (max of declared and worst
        observed), so admission trusts reality over an optimistic spec."""
        if jt.resource_class != LANE_VRAM:
            return jt
        eff = self.footprints.effective(jt, self._image_sig(jt))
        return jt if eff == jt.vram_mb else replace(jt, vram_mb=eff)

    def _type_timeout(self, jt: JobType) -> float:
        """The longest a job of *jt* may run; its manifest can only lower it. A
        serve-capable type always runs serve, everything else oneshot."""
        if jt.supports("serve"):
            return jt.serve_job_timeout
        return jt.oneshot_job_timeout_s or self.cfg.default_timeout_s

    def _job_timeout(self, pj: PendingJob) -> float:
        if pj.jt.supports("serve"):
            return serve_timeout(pj.manifest, pj.jt)
        return oneshot_timeout(pj.manifest, pj.jt, self.cfg)

    def _image_sig(self, jt: JobType) -> str:
        """The image a footprint was observed under, so a rebuild invalidates it."""
        if not jt.is_container:
            return ""
        return jt.image or self.cfg.sandbox_image

    def _weights_for(self, jt: JobType) -> str | None:
        """Return a verified cache path for *jt*, or fail before process launch.

        A sealed cache is always verified; with ``weights_required`` the seal must exist
        (SPEC 6.2).
        """
        cache = os.path.join(self.cfg.weights_dir, jt.id)
        manifest = os.path.join(cache, weights_mod.MANIFEST_NAME)
        if jt.weights_required:
            if not os.path.isdir(cache):
                raise RuntimeError(f"required weight cache missing: {cache}")
            if not os.path.isfile(manifest):
                raise RuntimeError(
                    f"required weight cache is unsealed: missing {weights_mod.MANIFEST_NAME}"
                )
        if os.path.isfile(manifest):
            ok, problems = weights_mod.verify(cache)
            if not ok:
                raise RuntimeError(f"weight cache verification failed: {problems[:3]}")
        return cache if os.path.isdir(cache) else None

    # ---- queue intake ------------------------------------------------------

    def reclaim(self) -> None:
        """Requeue stale bundles and archive delivered ones still in processing/, skipping
        the bundles of batches this broker is tracking."""
        for q in self.queues:
            active = {fn for (root, fn) in self._active_bundles if root == q.root}
            try:
                reclaimed = q.reclaim_stale(self.cfg.reclaim_after_s, skip=active)
            except OSError as exc:
                self.log.event("broker", "queue_error", queue=q.root, error=str(exc))
                continue
            try:
                for fn in q.archive_leftovers(skip=active):
                    self.log.event(fn, "archived_late")
            except OSError as exc:
                self.log.event("broker", "queue_error", queue=q.root, error=str(exc))
            for fn in reclaimed:
                # Intake scrubs every other sign of a requeue; this mark is the record.
                self._reclaimed.add((q.root, fn))
                self.log.event(
                    fn, "reclaimed_stale", requeued=True, age_limit_s=self.cfg.reclaim_after_s
                )

    def _log_left_in_processing(self) -> None:
        """Log each bundle an earlier run left in processing/ and when stale reclaim will run
        it again, so the wait after a crash is visible."""
        for q in self.queues:
            try:
                left = q.left_in_processing()
            except OSError as exc:
                self.log.event("broker", "queue_error", queue=q.root, error=str(exc))
                continue
            for fn, claimed in left:
                when = datetime.datetime.fromtimestamp(
                    claimed + self.cfg.reclaim_after_s, datetime.UTC
                )
                self.log.event(
                    fn,
                    "left_in_processing",
                    queue=q.root,
                    runs_again_after=when.replace(microsecond=0).isoformat(),
                )

    def intake(self) -> None:
        """Claim, unpack and enqueue new bundles; an unpackable one still reports every job
        failed (SPEC 7, 10). Bounded per tick so a flooded inbox cannot starve the loop."""
        if self._draining:
            return
        budget = self.cfg.max_claims_per_tick
        for q in self.queues:
            for fn in q.list_inbox():
                if budget <= 0:
                    return
                try:
                    claimed = q.claim(fn)
                except ClaimRejected as exc:
                    budget -= 1
                    self.log.event(fn, "claim_rejected", queue=q.root, reason=str(exc))
                    continue
                except OSError as exc:  # for example, a queue directory was replaced
                    self.log.event(fn, "claim_failed", queue=q.root, error=str(exc))
                    break
                if claimed is None:
                    continue
                budget -= 1
                self._ingest_bundle(q, claimed)

    def _ingest_bundle(self, q: QueueTree, processing_path: str) -> None:
        fn = os.path.basename(processing_path)
        root = ""
        try:
            batches = os.path.join(self.state_dir, "batches")
            os.makedirs(batches, exist_ok=True)
            root = tempfile.mkdtemp(prefix=_name_slug(fn) + "-", dir=batches)
            work_dir = os.path.join(root, "work")
            out_dir = os.path.join(root, "out")
            run_dir = os.path.join(root, "run")
            os.mkdir(out_dir)
            # Opened once: a second open by path could follow a link or FIFO swapped in.
            with q.open_processing(fn) as fh:
                members = bundle.unpack(
                    fh,
                    work_dir,
                    max_members=self.cfg.max_bundle_members,
                    max_bytes=self.cfg.max_bundle_mb * 1024 * 1024,
                )
            # Validate the manifests a worker will read: the unpacked files, not the zip.
            raw = bundle.read_unpacked_manifests(work_dir, members)
            if not raw:
                raise ValueError("bundle contains no job manifests")
            batch = Batch(
                queue=q,
                filename=fn,
                work_dir=work_dir,
                out_dir=out_dir,
                run_dir=run_dir,
                root=root,
            )
            claimed_outputs: dict[str, tuple[str, str]] = {}
            claimed_dirs: dict[str, tuple[str, str]] = {}
            priority_cap = self.cfg.max_priority.get(q.root)
            for name, rm in raw:
                if not isinstance(rm, dict):
                    raise ValueError("a job manifest is not a JSON object")
                jm = JobManifest.from_dict(rm)
                if name != f"{jm.job_id}.json":
                    # Workers find their manifest by job_id, so it must be this file. This also
                    # rules out a duplicate job_id: unpack refused two members of one name.
                    raise ValueError(
                        f"{bundle.PENDING_PREFIX}{name[:200]} holds job_id {jm.job_id!r}; "
                        "a manifest file is named <job_id>.json"
                    )
                # The broker owns lifecycle state: a submitted terminal status would be
                # dropped by the is_terminal guards with no result. Not done in from_dict,
                # which also parses results client-side, where it would hang --wait.
                jm.status = STATUS_QUEUED
                jm.result = None
                jm.finished_utc = None
                jm.message = ""
                jm.submitted_utc = now_iso()  # informational; ordering uses PendingJob.seq
                if priority_cap is not None and jm.priority > priority_cap:
                    self.log.event(
                        jm.job_id, "priority_capped", requested=jm.priority, cap=priority_cap
                    )
                    jm.priority = priority_cap
                self._claim_outputs(jm, claimed_outputs, claimed_dirs)
                if len(claimed_outputs) > self.cfg.max_bundle_members:
                    # Each output is a member of the result bundle.
                    raise ValueError(
                        f"the jobs declare more than {self.cfg.max_bundle_members} outputs"
                    )
                jt = self.reg.get(jm.job_type)  # raises on unknown -> batch fails
                bound = self._type_timeout(jt)
                if jm.timeout_s > bound:
                    # A client may shorten its job's timeout, never lengthen it.
                    self.log.event(jm.job_id, "timeout_capped", requested=jm.timeout_s, cap=bound)
                self._seq += 1
                batch.jobs.append(PendingJob(manifest=jm, jt=jt, batch=batch, seq=self._seq))
            # The worker reads its manifest from disk: give it the validated one, in the
            # broker's own serialization, so it cannot see anything intake did not check.
            pending_dir = os.path.join(work_dir, "jobs", "pending")
            for pj in batch.jobs:
                with open(
                    os.path.join(pending_dir, f"{pj.manifest.job_id}.json"), "w", encoding="utf-8"
                ) as out:
                    json.dump(pj.manifest.to_dict(), out, indent=2, sort_keys=True)
            # A multi-job batch can outlive reclaim_after_s even when every job fits, and
            # _active_bundles does not survive a restart, so say so.
            try:
                budget = sum(self._job_timeout(pj) for pj in batch.jobs)
                if budget > self.cfg.reclaim_after_s:
                    self.log.event(
                        fn,
                        "batch_may_outlive_reclaim_window",
                        batch_timeout_budget_s=budget,
                        reclaim_after_s=self.cfg.reclaim_after_s,
                        jobs=len(batch.jobs),
                    )
            except Exception:  # diagnostic only, never fail intake
                pass
            self.batches.append(batch)
            self._active_bundles.add((q.root, fn))
            self.pending.extend(batch.jobs)
            was_requeue = (q.root, fn) in self._reclaimed
            self._reclaimed.discard((q.root, fn))
            for pj in batch.jobs:
                self.log.event(
                    pj.manifest.job_id,
                    "claimed",
                    job_type=pj.jt.id,
                    **({"requeued_from_stale": True} if was_requeue else {}),
                )
        except Exception as exc:  # any failure -> report the whole batch
            if root:
                shutil.rmtree(root, ignore_errors=True)
            reason = f"batch failed: {exc}"[:2000]
            try:
                reported = q.fail_batch(
                    processing_path, reason, max_members=self.cfg.max_bundle_members
                )
            except Exception as report_exc:
                # The bundle stays in processing/ for stale reclaim; the tick goes on.
                self.log.event(fn, "batch_fail_report_failed", reason=reason, error=str(report_exc))
                return
            self.log.event(fn, "batch_failed", reason=reason, jobs=reported)

    @staticmethod
    def _claim_outputs(
        jm: JobManifest, claimed: dict[str, tuple[str, str]], dirs: dict[str, tuple[str, str]]
    ) -> None:
        """Reject a declared output that is reserved or overlaps another output in the same
        bundle (the jobs share one result tree), including one of the same job.

        *claimed* maps each output so far to (output, job), *dirs* each proper parent of
        one to that (output, job). Both are keyed case-folded, since on a case-insensitive
        filesystem ``out/A`` and ``out/a`` are one file. Two paths overlap when they are
        equal or one is a parent of the other, so each check is a few lookups and the whole
        bundle is linear in the total length of its paths. from_dict has already checked
        that each is canonical."""
        for rel in jm.outputs:
            parts = rel.casefold().split("/")
            if parts[0] in RESERVED_OUTPUT_DIRS:
                raise ValueError(f"job {jm.job_id!r}: output {rel!r} is under a reserved prefix")
            key = "/".join(parts)
            parents = ["/".join(parts[:i]) for i in range(1, len(parts))]
            hit = claimed.get(key) or dirs.get(key)
            if hit is None:
                hit = next((claimed[p] for p in parents if p in claimed), None)
            if hit is not None:
                raise ValueError(
                    f"output {rel[:200]!r} of job {jm.job_id!r} overlaps {hit[0][:200]!r} "
                    f"of job {hit[1]!r}"
                )
            claimed[key] = (rel, jm.job_id)
            for parent in parents:
                dirs.setdefault(parent, (rel, jm.job_id))

    # ---- scheduling decision ----------------------------------------------

    def _runnable_mode(self, jt) -> bool:
        """True if *jt* supports oneshot or serve; one with neither fails instead of hanging."""
        return jt.supports("oneshot") or jt.supports("serve")

    def _unschedulable(self, jt: JobType) -> str | None:
        """Why a vram job can never run, or None: its footprint exceeds the budget or the
        card's idle free memory, so the gate would time out and mark a healthy card degraded."""
        if jt.resource_class != LANE_VRAM:
            return None
        need = self._eff(jt).vram_mb
        if need > self.ledger.admit_budget_mb:
            return (
                f"job_type {jt.id} footprint {need}MB exceeds VRAM budget "
                f"{self.ledger.admit_budget_mb}MB; unschedulable"
            )
        if self._used_baseline is not None:
            try:
                idle_free = self.monitor.total_mb() - self._used_baseline
            except Exception:
                return None
            if need + self.cfg.headroom_mb > idle_free:
                return (
                    f"job_type {jt.id} footprint {need}MB plus headroom "
                    f"{self.cfg.headroom_mb}MB exceeds the {idle_free}MB this card has free "
                    "when idle; unschedulable"
                )
        return None

    def _admittable(self, pj: PendingJob) -> bool:
        if pj.uid in self._pending_reclaim:  # _service_reclaim drives it
            return False
        jt = pj.jt
        if jt.resource_class == LANE_CPU:
            return self.ledger.can_admit_cpu()
        if self._gpu_degraded:
            return False
        # A warm container takes a job while busy, up to SERVE_QUEUE_AHEAD queued. A dying
        # one takes none, and no successor may start until its serve_gone.
        if jt.supports("serve") and jt.id in self._serve:
            return self._warm_serve(jt) is not None and self._serve_has_room(jt)
        # A oneshot container loads its own model copy, so same-type jobs serialize.
        if self.ledger.is_resident(jt.id):
            return not self.ledger.residents[jt.id].busy
        # One job at the reclaim gate at a time: cold jobs parked together on an empty card
        # would all pass the eviction plan, then time out once the first is admitted.
        if self._pending_reclaim:
            return False
        return self.ledger.eviction_plan(self._eff(jt)) is not None

    def _serve_has_room(self, jt: JobType) -> bool:
        r = self.ledger.residents.get(jt.id)
        return r is None or r.running <= SERVE_QUEUE_AHEAD

    def _pending_depth(self) -> dict[str, int]:
        """Pending jobs per job_type id."""
        depth: dict[str, int] = {}
        for pj in self.pending:
            depth[pj.jt.id] = depth.get(pj.jt.id, 0) + 1
        return depth

    def pick_next(self) -> PendingJob | None:
        """Choose the next job to run (ordering in SPEC 5)."""
        # Over-budget vram jobs are never _admittable; listing them gets them failed, not stuck.
        candidates = [
            pj
            for pj in self.pending
            if self._admittable(pj) or self._unschedulable(pj.jt) is not None
        ]
        # Hold lower-priority work for a loaded model while a higher-priority cold job waits
        # for it to go idle and be evicted.
        waiting = [
            pj.manifest.priority
            for pj in self.pending
            if pj.jt.resource_class == LANE_VRAM
            and not self.ledger.is_resident(pj.jt.id)
            and pj not in candidates
        ]
        if waiting:
            top = max(waiting)
            candidates = [
                pj
                for pj in candidates
                if not (self.ledger.is_resident(pj.jt.id) and pj.manifest.priority < top)
            ]
        if not candidates:
            return None
        depth = self._pending_depth()

        def key(pj: PendingJob):
            jt = pj.jt
            resident = jt.resource_class == LANE_VRAM and self.ledger.is_resident(jt.id)
            amort = (depth.get(jt.id, 0) * jt.load_cost_s) if jt.resource_class == LANE_VRAM else 0
            return (
                -pj.manifest.priority,  # 1. submitter priority dominates
                0 if resident else 1,  # 2. then use the already-loaded model
                -amort,  # 3. then amortize the costliest cold load
                pj.seq,  # then intake order
            )

        return min(candidates, key=key)

    # ---- admission + dispatch ---------------------------------------------

    def _admit_and_run(self, pj: PendingJob) -> None:
        jt = pj.jt
        now = self.clock()

        if not self._runnable_mode(jt):
            self._fail(pj, f"job_type {jt.id} has no runnable mode (oneshot|serve)")
            return
        refusal = self._unsandboxed_refusal(jt)
        if refusal:
            self._fail(pj, refusal)
            return

        warm = self._warm_serve(jt)
        if warm is not None:
            self._dispatch_serve_warm(pj, warm, now)
            return

        if jt.resource_class == LANE_VRAM:
            reason = self._unschedulable(jt)
            if reason:
                self._fail(pj, reason)
                return
            if not self.ledger.is_resident(jt.id):
                # Evict idle residents, then park at the reclaim gate (SPEC 9).
                eff = self._eff(jt)
                for victim in self.ledger.eviction_plan(eff) or []:
                    self._evict_resident(victim)
                    self.log.event(jt.id, "evicted", victim=victim)
                free_target = eff.vram_mb + self.cfg.headroom_mb
                self._pending_reclaim[pj.uid] = (
                    pj,
                    free_target,
                    now + self.cfg.reclaim_timeout_s,
                )
                self.log.event(pj.manifest.job_id, "reclaim_pending", free_target=free_target)
                return
            self._spoil_attribution()
            self.ledger.admit_vram(self._eff(jt), now)
        else:
            self._spoil_attribution()
            self.ledger.admit_cpu()
        self._spawn(pj, now)

    def _unsandboxed_refusal(self, jt: JobType) -> str | None:
        """Why a job type must not run under this launcher, or None.

        The local launcher would run it as a plain host process with no isolation, so it is
        refused unless the operator opts in (development only). Under a container runtime,
        a host job type is refused unless allow_host_jobs is set."""
        if not jt.is_container:
            if isinstance(self.launcher, LocalLauncher) or self.cfg.allow_host_jobs:
                return None
            return (
                f'job_type {jt.id} is env = "host" and would run outside a container; set '
                "allow_host_jobs = true to run host job types beside a container runtime"
            )
        if not isinstance(self.launcher, LocalLauncher):
            return None
        if self.cfg.allow_unsandboxed_containers:
            return None
        return (
            f"job_type {jt.id} ({jt.resource_class} lane, sandbox {jt.sandbox}) needs a "
            'container, but container_runtime = "local" would run it as a host process '
            "with no isolation; use podman or docker, or set "
            "allow_unsandboxed_containers = true for development only"
        )

    def _attribution_clean(self) -> bool:
        """True if nothing else is on the GPU. Running cpu jobs count: a host-process job can
        use the GPU from any lane."""
        return not (self.ledger.residents or self.ledger.cpu_active)

    def _spoil_attribution(self) -> None:
        """Another job is starting on the GPU: drop every pending baseline."""
        self._baseline.clear()
        self._serve_baseline.clear()

    def _admit_resolved(self, pj: PendingJob) -> None:
        """Commit a non-resident vram job whose reclaim gate has passed."""
        now = self.clock()
        # An earlier parked job of this type may have created the warm container meanwhile.
        if pj.jt.supports("serve"):
            sc = self._warm_serve(pj.jt)
            if sc is None:
                self._spawn_serve(pj, now)
            elif self._serve_has_room(pj.jt):
                self._dispatch_serve_warm(pj, sc, now)
            # else it stays pending until the container has room
            return
        if self._attribution_clean():
            try:
                self._baseline[pj.uid] = self.monitor.used_mb()
            except Exception:
                pass
        else:
            self._spoil_attribution()
        self.ledger.admit_vram(self._eff(pj.jt), now)
        self._spawn(pj, now)

    def _spawn(self, pj: PendingJob, now: float) -> None:
        """Mark running, arm the watchdog, and start the supervisor thread."""
        self.pending.remove(pj)
        pj.manifest.mark_running()
        timeout = oneshot_timeout(pj.manifest, pj.jt, self.cfg)
        self._inflight[pj.uid] = (pj, now + timeout + ONESHOT_WATCHDOG_SLACK_S)
        attempt = Attempt(pj.attempts)
        self._attempts[pj.uid] = (pj, attempt)
        self.log.event(pj.manifest.job_id, "running", job_type=pj.jt.id)
        try:
            # Sample the peak only if the job had the GPU to itself at load (SPEC 9).
            sample = pj.jt.resource_class == LANE_VRAM and pj.uid in self._baseline
            threading.Thread(
                target=run_oneshot,
                args=(pj, self._worker_ctx(pj.jt), sample, attempt),
                daemon=True,
            ).start()
        except RuntimeError as exc:  # thread exhaustion
            self._attempts.pop(pj.uid, None)
            self.completions.put(
                Completion(pj, "failed", f"could not start worker: {exc}", [], infra=True)
            )

    # ---- serve mode (warm container, streamed jobs) -----------------------

    def _dispatch_serve_warm(self, pj: PendingJob, sc: ServeContainer, now: float) -> None:
        """Stream a job to a warm serve container. It enters _inflight only at serve_started,
        so a deep backlog never trips the watchdog."""
        self.pending.remove(pj)
        pj.manifest.mark_running()
        self.ledger.start_serve_job(pj.jt.id, now)
        self.log.event(pj.manifest.job_id, "running", job_type=pj.jt.id, serve=True)
        sc.jobq.put(pj)

    def _spawn_serve(self, pj: PendingJob, now: float) -> None:
        """Cold first job: reserve VRAM, create the warm container and its manager thread.
        Its directory is created first, so a full or unwritable state dir fails the job
        before anything is reserved for it."""
        jt = pj.jt
        try:
            serve_dir = os.path.join(self.state_dir, "serve")
            os.makedirs(serve_dir, exist_ok=True)
            root = tempfile.mkdtemp(prefix=_name_slug(jt.id) + "-", dir=serve_dir)
        except OSError as exc:
            self._fail(pj, f"could not create the serve container's directory: {exc}", infra=True)
            return
        self.pending.remove(pj)
        pj.manifest.mark_running()
        if self._attribution_clean():
            try:
                self._serve_baseline[jt.id] = self.monitor.used_mb()
            except Exception:
                pass
        else:
            self._spoil_attribution()
        self.ledger.admit_vram(self._eff(jt), now)  # resident, running=1
        self._serve_epoch += 1
        sc = ServeContainer(
            jt=jt,
            epoch=self._serve_epoch,
            name=serve_container_name(jt.id, self._serve_epoch, self.serve_name_prefix),
            root=root,
            jobq=_queue.Queue(),
            stop=threading.Event(),
            respawn_free_target_mb=self._eff(jt).vram_mb + self.cfg.headroom_mb,
        )
        self._serve[jt.id] = sc
        self._serve_loading[jt.id] = (sc.epoch, now + self._serve_load_window())
        sc.jobq.put(pj)
        self.log.event(pj.manifest.job_id, "running", job_type=jt.id, serve=True, load=True)
        try:
            sc.thread = threading.Thread(
                target=run_serve_manager, args=(sc, self._worker_ctx(jt)), daemon=True
            )
            sc.thread.start()
        except RuntimeError as exc:  # thread exhaustion
            self._serve.pop(jt.id, None)
            self.ledger.evict(jt.id)
            self._serve_baseline.pop(jt.id, None)
            self._serve_load_peak.pop(jt.id, None)
            shutil.rmtree(sc.root, ignore_errors=True)
            self.completions.put(
                Completion(
                    pj,
                    "failed",
                    f"could not start serve manager: {exc}",
                    [],
                    serve_epoch=sc.epoch,
                    infra=True,
                )
            )

    def _serve_load_window(self) -> float:
        """How long a serve container may take to come up: the manager's own connect and
        ready bounds plus slack. Past it the loop retires the container, whatever the
        manager thread is stuck on."""
        return self.cfg.serve_connect_timeout_s + self.cfg.serve_ready_timeout_s + 30.0

    def _evict_resident(self, jt_id: str) -> None:
        """Evict an idle resident. A serve container is torn down async; the reclaim gate
        guards the actual VRAM free."""
        r = self.ledger.residents.get(jt_id)
        if r is not None and r.busy:
            return  # callers pick idle residents; re-checked defensively
        sc = self._serve.pop(jt_id, None)
        self._serve_loading.pop(jt_id, None)
        self.ledger.evict(jt_id)
        self._serve_baseline.pop(jt_id, None)
        self._serve_load_peak.pop(jt_id, None)
        if sc is not None:
            self._tearing_down[sc.epoch] = self.clock()
            sc.stop.set()

    def _sample_serve_peaks(self) -> None:
        """Track a serve container's peak VRAM while it is the sole resident."""
        if not self._serve or len(self.ledger.residents) != 1:
            return
        if self.ledger.cpu_active:
            return
        jt_id = next(iter(self.ledger.residents))
        if jt_id not in self._serve or jt_id not in self._serve_baseline:
            return
        try:
            u = self.monitor.used_mb()
        except Exception:
            return
        if u > self._serve_load_peak.get(jt_id, 0):
            self._serve_load_peak[jt_id] = u

    def _shutdown_serve(self) -> None:
        """Signal every warm container to tear down."""
        for sc in list(self._serve.values()):
            sc.stop.set()

    def shutdown(self, timeout_s: float = 20.0) -> None:
        """Graceful teardown on broker exit: stop every warm container, join its manager,
        name-kill stragglers, give running oneshot jobs until the same deadline and then
        stop them, and drain completions so in-flight batches ship a terminal result. Call
        only after the loop has stopped (it drains on this thread). Times itself with
        time.monotonic, not self.clock, so a test clock cannot wedge it."""
        self._draining = True
        self._shutdown_serve()
        deadline = time.monotonic() + timeout_s
        for sc in list(self._serve.values()):
            if sc.thread is not None:
                sc.thread.join(max(0.0, deadline - time.monotonic()))
        self._stop_oneshots(deadline)
        # Name-kill only stragglers: it drops a blocked socket call, but cannot unblock a
        # manager still in serve.connect's pre-bind poll.
        for sc in list(self._serve.values()):
            if sc.thread is not None and sc.thread.is_alive():
                (self.launcher if sc.jt.is_container else self._local).terminate_by_name(sc.name)
                sc.thread.join(2.0)
        drain_deadline = time.monotonic() + 2.0
        while time.monotonic() < drain_deadline:
            self._drain_completions()
            if not self._serve:
                break
            time.sleep(0.02)
        self._drain_completions()
        if self._undelivered:
            # Last attempt; a batch still undelivered is re-run after a restart.
            self._retry_delivery(force=True)

    def _stop_oneshots(self, deadline: float) -> None:
        """Wait until *deadline* for running oneshot jobs to report, then stop the rest
        and wait for their processes to exit, so each gets a result and none keeps running
        after the broker."""
        while self._attempts and time.monotonic() < deadline:
            self._drain_completions()
            time.sleep(0.02)
        stoppers = []
        for pj, attempt in list(self._attempts.values()):
            msg = "the broker shut down before the job finished; resubmit it"
            t = threading.Thread(
                target=stop_attempt,
                args=(pj, attempt, self._worker_ctx(pj.jt), msg),
                daemon=True,
            )
            t.start()
            stoppers.append(t)
        for t in stoppers:
            t.join()  # bounded by stop_attempt's own wait
        # A watchdog stop already under way reports from its own thread.
        end = time.monotonic() + oneshot_mod.STOP_WAIT_S
        while self._attempts and time.monotonic() < end:
            self._drain_completions()
            time.sleep(0.02)

    def _service_reclaim(self) -> None:
        """Advance jobs parked at the reclaim gate (SPEC 9): admit at most one per tick,
        fail any past its deadline. Runs after completions drain, so freed VRAM shows."""
        if not self._pending_reclaim:
            return
        now = self.clock()
        try:
            free = self.monitor.free_mb()
        except Exception:
            free = -1
        admitted = False
        tearing_down = any(now - t < SERVE_TEARDOWN_BOUND_S for t in self._tearing_down.values())
        for uid in list(self._pending_reclaim):
            pj, target, deadline = self._pending_reclaim[uid]
            if tearing_down:
                # An evicted container is still unloading, which can take longer than the
                # gate allows: its time counts from when the last one is gone.
                deadline = now + self.cfg.reclaim_timeout_s
                self._pending_reclaim[uid] = (pj, target, deadline)
            # Require both NVML room and the ledger: NVML lags admission, so it alone would
            # over-commit against a job admitted on a prior tick.
            eff = self._eff(pj.jt)
            ledger_ok = self.ledger.is_resident(pj.jt.id) or self.ledger.can_fit(eff)
            # A warm serve container needs no new VRAM, and its loaded model keeps free low.
            warm_serve = self._warm_serve(pj.jt) is not None
            dying = pj.jt.id in self._serve and not warm_serve
            if pj.manifest.is_terminal:
                self._pending_reclaim.pop(uid, None)
            elif dying:
                continue  # re-check once serve_gone lands
            elif not admitted and ledger_ok and (warm_serve or free >= target):
                self._pending_reclaim.pop(uid, None)
                self._admit_resolved(pj)
                admitted = True
            elif not ledger_ok:
                # The ledger, not the card, holds it back, so a timeout would wrongly
                # degrade the GPU. Unpark it; admission retries when it fits.
                self._pending_reclaim.pop(uid, None)
                self.log.event(pj.manifest.job_id, "reclaim_unparked", reason="ledger_full")
            elif now >= deadline:
                self._pending_reclaim.pop(uid, None)
                self.log.event(pj.jt.id, "reclaim_timeout", free_target=target)
                self._degrade("the reclaim gate timed out waiting for VRAM to be free")
                self._fail(pj, "VRAM reclaim gate timed out; GPU marked degraded", infra=True)

    def _warm_serve(self, jt: JobType) -> ServeContainer | None:
        """The warm container that can take a job of *jt* now, if any."""
        sc = self._serve.get(jt.id) if jt.supports("serve") else None
        return sc if sc is not None and not sc.dying else None

    def _module_for(self, jt):
        """The Module a job type came from, or None."""
        for mod in self.reg.modules:
            if mod.name == jt.module_name:
                return mod
        return None

    def _write_worker_log(self, pj: PendingJob, streams: dict[str, bytes], note: str) -> str | None:
        """Write the job's worker log into its result bundle. Called from worker threads;
        never raises, so a log failure cannot fail a job."""
        try:
            return worker_log.write_worker_log(pj, streams, note, self._module_for(pj.jt))
        except Exception as exc:
            self.log.event(pj.manifest.job_id, "worker_log_failed", error=str(exc))
            return None

    def _worker_ctx(self, jt: JobType) -> WorkerContext:
        """The collaborators a worker thread for *jt* gets (never loop-owned state)."""
        return WorkerContext(
            cfg=self.cfg,
            launcher=self.launcher if jt.is_container else self._local,
            monitor=self.monitor,
            completions=self.completions,
            log=self.log,
            weights_for=self._weights_for,
            write_worker_log=self._write_worker_log,
            job_name_prefix=self.job_name_prefix,
        )

    def _drain_completions(self) -> None:
        while True:
            try:
                c = self.completions.get_nowait()
            except _queue.Empty:
                return
            self._on_complete(c)

    def _on_complete(self, c: Completion) -> None:
        now = self.clock()
        # Serve control messages finalize no job.
        if c.kind == "serve_started":
            # Arm the watchdog only now, so a queued backlog cannot false-fire it.
            pj = c.pending
            assert pj is not None
            sc = self._serve.get(pj.jt.id)
            if c.attempt != pj.attempts:  # stale attempt
                return
            if sc is not None and sc.epoch == c.serve_epoch:
                if self._serve_loading.get(pj.jt.id, (None,))[0] == sc.epoch:
                    del self._serve_loading[pj.jt.id]
                self._inflight[pj.uid] = (
                    pj,
                    now + self.cfg.serve_stage_slack_s + serve_timeout(pj.manifest, pj.jt) + 30.0,
                )
                self._inflight_epoch[pj.uid] = sc.epoch
            return
        if c.kind == "serve_gone":
            jt_id = c.jt_id
            assert jt_id is not None
            self._tearing_down.pop(c.serve_epoch or 0, None)
            sc = self._serve.get(jt_id)
            if sc is not None and sc.epoch == c.serve_epoch:  # live (crash teardown)
                self._serve.pop(jt_id, None)
                self._serve_loading.pop(jt_id, None)
                self.ledger.evict(jt_id)
                self._serve_baseline.pop(jt_id, None)
                self._serve_load_peak.pop(jt_id, None)
                self.log.event(jt_id, "serve_gone")
                # Jobs dispatched during the teardown window, after the manager's fail_queued,
                # would never be read; fail them here or their batch never terminates.
                while True:
                    try:
                        item = sc.jobq.get_nowait()
                    except _queue.Empty:
                        break
                    if item is UNLOAD or item.manifest.is_terminal:
                        continue
                    self._fail(
                        item,
                        "serve container died before the job started",
                        infra=True,
                        not_started=True,
                    )
            return  # stale epoch: the evict already cleaned up
        if c.kind == "serve_reloading":  # a recycle: waits for VRAM, then loads again
            sc = self._serve.get(c.jt_id or "")
            if sc is not None and sc.epoch == c.serve_epoch:
                window = self.cfg.reclaim_timeout_s + self._serve_load_window()
                self._serve_loading[sc.jt.id] = (sc.epoch, now + window)
            return
        if c.kind == "serve_respawn":  # same container slot, fresh model
            self.log.event(c.jt_id or "", "serve_respawn")
            return

        pj = c.pending
        assert pj is not None, "a job completion always carries its PendingJob"
        jt = pj.jt
        # A watchdog failure and a late real completion can both arrive; take the first.
        if pj.manifest.is_terminal:
            return
        # A retried attempt's accounting is already settled; never touch the ledger twice.
        if c.attempt != pj.attempts:
            return
        self._inflight.pop(pj.uid, None)
        self._inflight_epoch.pop(pj.uid, None)
        self._attempts.pop(pj.uid, None)
        baseline = self._baseline.pop(pj.uid, None)  # oneshot-vram only
        infra, message = c.infra, c.message
        if c.still_running:
            # Its slot is released below so the job can finish, but the memory may still be
            # in use: hold vram admission until the card shows it free (_recover_degraded).
            self.log.event(pj.manifest.job_id, "worker_not_stopped", job_type=jt.id)
            message += "\n[the worker did not exit when stopped"
            if jt.resource_class == LANE_VRAM:
                self._degrade("a stopped worker did not exit")
                message += "; the GPU is marked degraded until its memory is free"
            message += "]"
        if c.cuda_oom:
            peak = c.peak_used_mb if c.serve_epoch is None else self._serve_load_peak.get(jt.id)
            alone = baseline if c.serve_epoch is None else self._serve_baseline.get(jt.id)
            own = self._oom_is_own(alone)
            self.log.event(
                pj.manifest.job_id,
                "cuda_oom",
                job_type=jt.id,
                declared_mb=jt.vram_mb,
                effective_mb=self._eff(jt).vram_mb,
                peak_used_mb=peak,
                retried=infra and not own,
            )
            if own:
                # A retry would meet the same free memory, so the failure is the job's own.
                infra = False
                message += (
                    "\n[not retried: the job had the GPU to itself, so it needs more memory "
                    "than the card has; ask for less in its options or raise vram_mb]"
                )
        # After a CUDA error, or a vram job that timed out, was killed or died on a signal,
        # the card itself may be at fault.
        if c.cuda_error or (c.abnormal_exit and jt.resource_class == LANE_VRAM):
            fault = self._check_gpu_health()
            if fault:
                infra = False
                message += f"\n[not retried: GPU fault: {fault}]"
        if c.reclaim_timeout:
            # A serve recycle waited for VRAM that never returned: treat it like the gate.
            self.log.event(jt.id, "reclaim_timeout", during="serve_respawn")
            self._degrade("a serve restart waited for VRAM that did not become free")
        # Branch on the epoch stamp: a oneshot run of a dual-mode type has serve_epoch=None
        # and must take the vram branch, or its ledger slot is never released.
        if jt.supports("serve") and c.serve_epoch is not None:
            sc = self._serve.get(jt.id)
            if sc is not None and sc.epoch == c.serve_epoch:  # live container
                # A CUDA OOM alone leaves the worker running (MODULE_GUIDE 3.2).
                if c.container_gone or (c.infra and not c.cuda_oom):
                    sc.dying = True  # its manager is exiting; take no new work
                # After every finished job, since a model's peak can grow on later jobs.
                # OOM and infra peaks measure the card, not the model.
                if c.status == "done":
                    self._reconcile_footprint(
                        jt, self._serve_baseline.get(jt.id), self._serve_load_peak.get(jt.id)
                    )
                eff = self._eff(jt)
                sc.respawn_free_target_mb = eff.vram_mb + self.cfg.headroom_mb
                resident = self.ledger.residents.get(jt.id)
                if resident is not None:
                    resident.jt = eff  # the ledger counts what the model was seen to use
                self.ledger.finish_serve_job(jt.id, now, jt.keep_warm_s)
            # Stale epoch: already evicted, so leave the ledger alone but finalize the job.
        elif jt.resource_class == LANE_VRAM:
            if not c.cuda_oom:  # an OOM peak measures the card, not the model
                self._reconcile_footprint(jt, baseline, c.peak_used_mb)
            self.ledger.finish_vram(jt.id, now, jt.keep_warm_s)
            # A oneshot exit frees its VRAM, so an idle slot is released at once.
            r = self.ledger.residents.get(jt.id)
            if r is not None and not r.busy:
                self.ledger.evict(jt.id)
        else:
            self.ledger.finish_cpu()
        if c.status != "done" and infra and self._retry(pj, message, c.not_started):
            return
        if c.status == "done":
            pj.manifest.mark_done(result={"outputs": c.outputs}, message=message)
        else:
            message = self._quarantine(pj, message) if infra else message
            pj.manifest.mark_failed(message)
        self.log.event(pj.manifest.job_id, c.status, message=message)
        self._finish_job(pj)

    def _oom_is_own(self, baseline: int | None) -> bool:
        """True if a CUDA OOM belongs to the job: no other job used the GPU during the
        attempt (*baseline*, its pre-load reading, survives only then) and nothing outside
        the ledger was using memory when it started. Unknown counts as not its own."""
        if baseline is None or self._used_baseline is None:
            return False
        return baseline <= self._used_baseline + max(self.cfg.headroom_mb, 512)

    def _degrade(self, reason: str) -> None:
        """Hold vram admission until _recover_degraded sees the card idle again, or after a
        hardware fault until a restart. *reason* is shown in status.json."""
        self._gpu_degraded = True
        self._degraded_reason = reason

    def _check_gpu_health(self) -> str:
        """Ask the monitor whether the GPU still answers and reports no hardware fault.
        Returns the fault, or "" if there is none. Called after a job failure that may be
        the card's, and every HEALTH_POLL_S by the loop.

        A reported fault, including a monitor saying the GPU is lost, marks the GPU degraded
        until the broker restarts. A query that fails is asked once more; if it fails again
        the GPU is marked degraded only until it answers, as a slow ``nvidia-smi`` right
        after a hung job is not a hardware fault."""
        self._next_health_poll = self.clock() + HEALTH_POLL_S
        try:
            try:
                faults = self.monitor.hardware_faults()
            except Exception:
                faults = self.monitor.hardware_faults()
        except Exception as exc:
            if not self._gpu_degraded:
                self.log.event("broker", "gpu_unanswered", error=type(exc).__name__)
            self._degrade("the GPU did not answer the monitor")
            return ""
        fault = "; ".join(faults)
        if fault:
            self._degrade(f"hardware fault: {fault}")
            self._gpu_fault = fault
            self.log.event("broker", "gpu_fault", fault=fault)
        return fault

    def _retry(self, pj: PendingJob, message: str, not_started: bool = False) -> bool:
        """Re-queue *pj* after an infrastructure failure if it has retries left (SPEC 10);
        False means the caller fails it.

        A job that *not_started* is re-queued without spending a retry. This stays finite:
        each container death is charged to the job that was running."""
        if self._draining:
            return False  # nothing would run it
        if not_started:
            pj.requeues += 1
        elif self._retries_spent(pj) >= self.cfg.max_retries:
            return False
        pj.attempts += 1  # also the stale-report guard
        pj.manifest.status = STATUS_QUEUED
        pj.manifest.message = f"retrying after infrastructure failure: {message[:300]}"
        self.pending.append(pj)
        self.log.event(
            pj.manifest.job_id,
            "retry",
            attempt=pj.attempts + 1,
            max_retries=self.cfg.max_retries,
            retry_spent=not not_started,
            reason=message[:500],
        )
        return True

    @staticmethod
    def _retries_spent(pj: PendingJob) -> int:
        return pj.attempts - pj.requeues

    def _quarantine(self, pj: PendingJob, message: str) -> str:
        """Final infrastructure failure. A job that used a retry is quarantined, so one that
        reliably breaks the infrastructure cannot wedge the scheduler."""
        spent = self._retries_spent(pj)
        if self._draining and spent < self.cfg.max_retries:
            return f"{message}\n[not retried: the broker is shutting down]"
        if spent == 0:
            return message
        self.log.event(pj.manifest.job_id, "quarantined", attempts=spent + 1)
        return (
            f"{message}\n[quarantined: infrastructure failure on all "
            f"{spent + 1} attempts; not retried again]"
        )

    def _reconcile_footprint(
        self, jt: JobType, baseline: int | None, peak_used: int | None
    ) -> None:
        """Record an observed footprint from a clean baseline and peak (SPEC 9)."""
        if baseline is None or peak_used is None:
            return
        observed = peak_used - baseline
        if observed <= 0:
            return
        if observed > self.ledger.admit_budget_mb:
            # Recording it would make the type unschedulable on one sample, so it is only
            # reported (`gpuworker footprints reset` clears a bad recorded value).
            self.log.event(
                jt.id,
                "footprint_rejected",
                observed_mb=observed,
                admit_budget_mb=self.ledger.admit_budget_mb,
            )
            return
        if self.footprints.record(jt.id, observed, self._image_sig(jt)):
            declared = jt.vram_mb
            over = observed > declared * (1.0 + self.cfg.footprint_variance_tol)
            self.log.event(
                jt.id,
                "footprint_variance" if over else "footprint_observed",
                declared_mb=declared,
                observed_mb=observed,
            )

    def _fail(
        self, pj: PendingJob, message: str, infra: bool = False, not_started: bool = False
    ) -> None:
        """Fail a job that holds no ledger slot; with *infra* it may be retried instead."""
        if pj in self.pending:
            self.pending.remove(pj)
        self._inflight.pop(pj.uid, None)
        self._attempts.pop(pj.uid, None)
        self._pending_reclaim.pop(pj.uid, None)
        if infra:
            if self._retry(pj, message, not_started):
                return
            message = self._quarantine(pj, message)
        pj.manifest.mark_failed(message)
        self.log.event(pj.manifest.job_id, "failed", message=message)
        self._finish_job(pj)

    def _finish_job(self, pj: PendingJob) -> None:
        b = pj.batch
        b.results[pj.manifest.job_id] = pj.manifest.to_dict()
        if b.done:
            self._finish_batch(b)

    def _finish_batch(self, b: Batch) -> None:
        if self._deliver(b):
            self._retire(b)
            return
        # Keep the outputs and the place in _active_bundles, so stale reclaim skips the
        # bundle meanwhile. No job reads the unpacked input or an attempt directory again.
        for d in (b.work_dir, b.run_dir):
            shutil.rmtree(d, ignore_errors=True)
        self._undelivered.append((b, self.clock() + RESULT_RETRY_S))

    def _deliver(self, b: Batch) -> bool:
        """Write the batch's result bundle (manifests, declared outputs, worker logs).
        Returns False only if not even a manifest-only result could be written."""
        results = [b.results[j.manifest.job_id] for j in b.jobs]
        out_files: list[str] = []
        seen: set[str] = set()
        for j in b.jobs:
            res = b.results[j.manifest.job_id].get("result") or {}
            for rel in res.get("outputs", []):
                if rel in seen:
                    continue
                if not bundle.is_regular_member(b.out_dir, rel):
                    self.log.event(j.manifest.job_id, "unsafe_output_skipped", rel=rel)
                    continue
                seen.add(rel)
                out_files.append(rel)
        # Worker logs come from a reserved prefix, not the declared outputs, so a log that
        # always exists cannot satisfy the missing-outputs check.
        for j in b.jobs:
            rel = f"{worker_log.WORKER_LOG_PREFIX}{j.manifest.job_id}.log"
            if rel not in seen and bundle.is_regular_member(b.out_dir, rel):
                seen.add(rel)
                out_files.append(rel)
        try:
            b.queue.write_result(b.filename, results, files_root=b.out_dir, files=out_files)
            return True
        except Exception as exc:
            self.log.event(b.filename, "result_pack_failed", error=str(exc))
            pack_error = exc
        # Ship terminal manifests anyway, but a job whose outputs never arrived is not `done`.
        degraded = [self._undelivered_result(r, pack_error) for r in results]
        try:
            b.queue.write_result(b.filename, degraded)
            return True
        except Exception as exc:
            b.delivery_attempts += 1
            b.delivery_error = str(exc)
            self.log.event(
                b.filename,
                "result_write_failed",
                error=str(exc),
                attempt=b.delivery_attempts,
                retry_in_s=RESULT_RETRY_S,
            )
            return False

    @staticmethod
    def _undelivered_result(result: dict, error: Exception) -> dict:
        outputs = (result.get("result") or {}).get("outputs") or []
        if result.get("status") != "done" or not outputs:
            return result
        return {
            **result,
            "status": "failed",
            "result": None,
            "message": (
                f"the job finished, but its outputs could not be delivered: {error}. "
                "Resubmit to produce them again."
            ),
        }

    def _retry_delivery(self, force: bool = False) -> None:
        now = self.clock()
        waiting, self._undelivered = self._undelivered, []
        for b, due in waiting:
            if not force and now < due:
                self._undelivered.append((b, due))
            elif self._deliver(b):
                self._retire(b)
            elif b.delivery_attempts >= RESULT_ATTEMPTS:
                self._abandon(b)
            else:
                self._undelivered.append((b, now + RESULT_RETRY_S))

    def _retire(self, b: Batch) -> None:
        """A result is in the outbox: archive the input and drop the working state."""
        try:
            b.queue.archive(b.filename)
        except OSError as exc:
            self.log.event(b.filename, "archive_failed", error=str(exc))
        self._drop(b)
        self.log.event(b.filename, "batch_done", jobs=len(b.jobs))

    def _abandon(self, b: Batch) -> None:
        """No result could be written in RESULT_ATTEMPTS tries: move the input to failed/
        with the reason and drop the batch. Left in processing/, stale reclaim would run it
        again. If even that move fails, the bundle stays claimed until the broker exits."""
        reason = f"no result could be written in {b.delivery_attempts} attempts: {b.delivery_error}"
        moved = True
        try:
            b.queue.move_to_failed(b.filename, reason)
        except OSError as exc:
            moved = False
            self.log.event(b.filename, "move_to_failed_failed", error=str(exc))
        self._drop(b, release=moved)
        self.log.event(b.filename, "result_abandoned", jobs=len(b.jobs), message=reason)

    def _drop(self, b: Batch, release: bool = True) -> None:
        for d in (b.work_dir, b.out_dir, b.run_dir):
            shutil.rmtree(d, ignore_errors=True)
        if b.root:
            shutil.rmtree(b.root, ignore_errors=True)
        if b in self.batches:
            self.batches.remove(b)
        if release:
            self._active_bundles.discard((b.queue.root, b.filename))

    # ---- keep-warm eviction ------------------------------------------------

    def evict_expired(self) -> None:
        now = self.clock()
        for jt_id in self.ledger.expired_warm(now):
            self._evict_resident(jt_id)
            self.log.event(jt_id, "evicted_idle")

    def check_watchdog(self) -> None:
        """Fail any admitted job that never reported by its deadline (SPEC 10). A oneshot
        job's process is stopped first, off the loop, and the job is reported (and its slot
        released) once the process has exited."""
        now = self.clock()
        overdue = [uid for uid, (_pj, dl) in self._inflight.items() if now >= dl]
        msg = "watchdog: no completion within deadline"
        for uid in overdue:
            pj, _ = self._inflight.pop(uid)
            epoch = self._inflight_epoch.pop(uid, None)
            self.log.event(pj.manifest.job_id, "watchdog_timeout")
            running = self._attempts.get(uid)
            if epoch is None and running is not None:
                threading.Thread(
                    target=stop_attempt,
                    args=(pj, running[1], self._worker_ctx(pj.jt), f"{msg}; worker stopped"),
                    daemon=True,
                ).start()
                continue
            # The epoch stamp routes it to the serve branch of _on_complete.
            self.completions.put(
                Completion(pj, "failed", msg, [], serve_epoch=epoch, abnormal_exit=True)
            )
            sc = self._serve.get(pj.jt.id)
            if epoch is not None and sc is not None and sc.epoch == epoch:
                self._retire_hung_serve(sc)
        for jt_id, (epoch, dl) in list(self._serve_loading.items()):
            sc = self._serve.get(jt_id)
            if sc is None or sc.epoch != epoch:
                del self._serve_loading[jt_id]
            elif now >= dl:
                self.log.event(jt_id, "serve_load_timeout", container=sc.name)
                self._retire_hung_serve(sc, reason="load_timeout")

    def _retire_hung_serve(self, sc: ServeContainer, reason: str = "watchdog") -> None:
        """A serve job outlived its watchdog, or the container never finished loading, so
        its manager is stuck. Free its slot, fail or re-queue its waiting jobs, and
        force-remove it off the loop thread. The reclaim gate still waits for its memory.

        After a hung job the waiting jobs never started, so they spend no retry. After a
        load timeout they do: a model that never loads would otherwise be retried forever."""
        jt = sc.jt
        self._serve.pop(jt.id, None)
        self._serve_loading.pop(jt.id, None)
        self.ledger.evict(jt.id)
        self._serve_baseline.pop(jt.id, None)
        self._serve_load_peak.pop(jt.id, None)
        sc.dying = True
        sc.stop.set()
        if reason == "load_timeout" and jt.resource_class == LANE_VRAM:
            self._check_gpu_health()  # after a hung job, the check runs when its failure lands
        while True:
            try:
                item = sc.jobq.get_nowait()
            except _queue.Empty:
                break
            if item is UNLOAD or item.manifest.is_terminal:
                continue
            if reason == "load_timeout":
                self._fail(
                    item,
                    f"serve container did not load within {self._serve_load_window():.0f}s",
                    infra=True,
                )
            else:
                self._fail(
                    item,
                    "serve container stopped after a job hung",
                    infra=True,
                    not_started=True,
                )
        launcher = self.launcher if jt.is_container else self._local
        threading.Thread(target=launcher.terminate_by_name, args=(sc.name,), daemon=True).start()
        self.log.event(jt.id, "serve_retired", reason=reason, container=sc.name)

    def _recover_degraded(self) -> None:
        if not self._gpu_degraded or self._gpu_fault:
            return
        try:
            used = self.monitor.used_mb()
        except Exception:
            return
        # Recover at the idle baseline plus residents. A free-memory test never passes on
        # WSL2, where NVML counts the desktop, so it is only the pre-baseline fallback.
        if self._used_baseline is not None:
            slack = max(self.cfg.headroom_mb, 512)
            healthy = used <= self._used_baseline + self.ledger.resident_vram_mb + slack
        else:
            free = self.monitor.total_mb() - used
            healthy = free >= self.cfg.admit_budget_mb() - self.ledger.resident_vram_mb
        if healthy:
            self._gpu_degraded = False
            self._degraded_reason = ""
            self.log.event("broker", "gpu_recovered")

    # ---- main loop ---------------------------------------------------------

    def tick(self) -> None:
        """One scheduler iteration. Public for deterministic tests."""
        # Sanity-gated so a stray model or unreaped orphan cannot poison the idle baseline.
        if self._used_baseline is None and not self._gpu_degraded and self._attribution_clean():
            try:
                u = self.monitor.used_mb()
                if u < self.cfg.admit_budget_mb() // 2:
                    self._used_baseline = u
            except Exception:
                pass
        self._drain_completions()
        self.check_watchdog()
        self._drain_completions()  # finalize watchdog failures now
        if not self._gpu_fault and self.clock() >= self._next_health_poll:
            self._check_gpu_health()
        self._recover_degraded()
        self._sample_serve_peaks()
        # Parked jobs go before new picks, now that drained completions show freed VRAM.
        self._service_reclaim()
        if self._undelivered:
            self._retry_delivery()
        self.reclaim()
        self.intake()
        while True:
            pj = self.pick_next()
            if pj is None:
                break
            self._admit_and_run(pj)
        self.evict_expired()

    def write_status(self) -> None:
        """Refresh the live status snapshot. Best-effort: it must never disturb scheduling."""
        if not self.cfg.status_path:
            return
        try:
            write_status(self.cfg.status_path, self)
        except Exception:
            pass

    def _clear_state(self) -> None:
        """Remove batch and serve dirs a previous run left; none of them is live."""
        for sub in ("batches", "serve"):
            shutil.rmtree(os.path.join(self.state_dir, sub), ignore_errors=True)

    def run(self, interval_s: float = 1.0) -> None:
        """Run the loop until stop(). Raises lock.BrokerRunning, before touching any state,
        if another broker uses the same state_dir."""
        lock = BrokerLock(self.state_dir)
        lock.acquire()
        try:
            self._run(interval_s)
        finally:
            lock.release()

    def _run(self, interval_s: float) -> None:
        self.job_name_prefix = job_name_prefix(_broker_id(self.state_dir))
        self.log.event("broker", "start", budget_mb=self.cfg.admit_budget_mb())
        # Create and pin every queue tree now, before anything can move a subdirectory.
        for q in self.queues:
            q.ensure()
        self._log_left_in_processing()
        try:
            total = self.monitor.total_mb()
        except Exception:
            total = None
        if total is not None and total < self.cfg.gpu_budget_mb:
            # Say up front why such jobs will later fail as unschedulable.
            self.log.event(
                "broker",
                "gpu_budget_exceeds_card",
                gpu_budget_mb=self.cfg.gpu_budget_mb,
                card_total_mb=total,
            )
        self._clear_state()
        self._publish_limits()
        # Reap serve containers a crashed broker leaked: they hold VRAM and collide on name.
        # And oneshot containers, which run without --rm: this broker's own, running or not
        # (the state_dir lock means no other broker has its id), and any exited one.
        if not self._reaped_orphans:
            self._reaped_orphans = True
            try:
                self.launcher.reap_serve_orphans(self.serve_name_prefix)
                self.launcher.reap_jobs(self.job_name_prefix, exited_only=False)
                self.launcher.reap_jobs(JOB_NAME_PREFIX, exited_only=True)
            except (OSError, subprocess.SubprocessError) as exc:
                self.log.event("broker", "reap_orphans_failed", error=str(exc))
        errors = 0  # consecutive failed iterations
        try:
            while not self._stop.is_set():
                try:
                    self.tick()
                    errors = 0
                    self.last_loop_error = None  # the log keeps it
                except Exception as exc:  # the loop must never die
                    errors += 1
                    self._loop_error(exc, errors)
                self.write_status()
                # Back off while every iteration fails, so a lasting error does not flood
                # the log. stop() still ends the wait at once.
                wait = interval_s * 2 ** min(errors, 6) if errors else interval_s
                self._stop.wait(max(interval_s, min(wait, TICK_ERROR_BACKOFF_MAX_S)))
        finally:
            self.shutdown()

    def _loop_error(self, exc: Exception, count: int) -> None:
        """Log an exception that escaped a loop iteration with its traceback, and keep it
        for status.json."""
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self.last_loop_error = {
            "ts": now_iso(),
            "error": f"{type(exc).__name__}: {exc}"[:2000],
            "consecutive": count,
        }
        self.log.event(
            "broker",
            "tick_error",
            error=self.last_loop_error["error"],
            consecutive=count,
            traceback=tb[-8000:],
        )

    def stop(self) -> None:
        self._stop.set()
