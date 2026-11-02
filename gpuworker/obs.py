"""Observability (SPEC 11): structured per-job log + a status snapshot."""

from __future__ import annotations

import json
import os
import sys
import threading
from typing import Any, TextIO

from .manifest import now_iso


class JobLogger:
    """Structured JSON-lines logger keyed by job_id. Thread-safe (jobs report from
    supervisor threads)."""

    def __init__(self, stream: TextIO | None):
        self._stream = stream
        self._lock = threading.Lock()

    @classmethod
    def null(cls) -> JobLogger:
        return cls(None)

    @classmethod
    def to_path(cls, path: str) -> JobLogger:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        return cls(open(path, "a", encoding="utf-8"))

    @classmethod
    def to_stdout(cls) -> JobLogger:
        return cls(sys.stdout)

    def event(self, job_id: str, event: str, **fields: Any) -> None:
        if self._stream is None:
            return
        # Observability must never break the caller: the scheduler loop's last-resort
        # error handler itself logs, so an unguarded disk-full / share write error here would
        # escape the loop's except and exit the broker. Swallow all write failures.
        try:
            rec = {"ts": now_iso(), "job_id": job_id, "event": event, **fields}
            line = json.dumps(rec, sort_keys=True)
            with self._lock:
                self._stream.write(line + "\n")
                self._stream.flush()
        except Exception:
            pass

    def close(self) -> None:
        # Don't close shared streams (stdout); only file streams we opened.
        if self._stream is not None and self._stream not in (sys.stdout, sys.stderr):
            try:
                self._stream.close()
            except Exception:
                pass


def write_status(path: str, sched) -> None:
    """Snapshot GPU + queue + lane + per-module memory to ``status.json`` (SPEC
    11). Surfaces live occupied/free VRAM and, per job type, the declared footprint,
    the observed peak (footprint reconciliation), and the effective number admission
    uses, so the operator can see required-vs-actual memory at a glance."""
    monitor, ledger = sched.monitor, sched.ledger
    try:
        used, total, free = monitor.used_mb(), monitor.total_mb(), monitor.free_mb()
    except Exception:
        used = total = free = -1

    modules = []
    for jt_id, jt in sorted(sched.reg.job_types.items()):
        modules.append(
            {
                "job_type": jt_id,
                "lane": jt.resource_class,
                "declared_mb": jt.vram_mb,
                # The same image-stamped values admission uses (Scheduler._eff).
                "observed_mb": sched.footprints.observed(jt_id, sched._image_sig(jt)),
                "effective_mb": sched.footprints.effective(jt, sched._image_sig(jt)),
                "vram_profile": jt.vram_profile,
                "co_resident": jt.co_resident,
                "packable": jt.packable,
                "sandbox": jt.sandbox,
            }
        )

    snap = {
        "ts": now_iso(),
        "gpu": {"total_mb": total, "used_mb": used, "free_mb": free},
        "ledger": {
            "admit_budget_mb": ledger.admit_budget_mb,
            "resident": [
                {"job_type": r.jt.id, "vram_mb": r.jt.vram_mb, "running": r.running, "busy": r.busy}
                for r in ledger.residents.values()
            ],
            "resident_vram_mb": ledger.resident_vram_mb,
            "cpu_active": ledger.cpu_active,
            "degraded": sched._gpu_degraded,
            "degraded_reason": sched._degraded_reason if sched._gpu_degraded else "",
            "gpu_fault": sched._gpu_fault,
        },
        "modules": modules,
        "queues": [{"root": q.root, "inbox_depth": len(q.list_inbox())} for q in sched.queues],
        "pending": len(sched.pending),
        # The last exception that escaped a loop iteration (its traceback is in the log).
        "last_loop_error": sched.last_loop_error,
    }
    # Per-process temp name: the running broker writes this same status_path on its
    # own loop, so a concurrent `gpuworker status` sharing one "<path>.tmp" loses the
    # race: whichever os.replace lands second finds its temp already renamed away and
    # dies with FileNotFoundError.
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(snap, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        # Don't strand a partial temp file. Still raises: the CLI should report a failed
        # snapshot, and Scheduler.write_status already swallows this for the broker loop.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
