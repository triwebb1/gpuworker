"""VRAM readings and GPU health (SPEC 9, 10).

The broker never imports torch. Its truth source for VRAM is NVML (pynvml) or
``nvidia-smi`` polling. After a worker reports a CUDA error, or a vram job times out, is
killed or dies on a signal, the scheduler asks the monitor for
:meth:`GpuMonitor.hardware_faults`, and it also polls them on a timer, so a failing card
is marked degraded instead of being handed the next job. ``torch.cuda.empty_cache`` is
the *worker's* obligation during graceful unload, not something the broker calls.

The reclaim gate itself is in the scheduler (``Scheduler._service_reclaim``): a cold
job waits there until NVML shows its memory free, because a driver releases a previous
CUDA context some time after the process exits. :func:`wait_for_free` is the blocking
form of the same check, used only when a serve container restarts its own model.
"""

from __future__ import annotations

import subprocess
import time
from typing import Protocol


class GpuMonitor(Protocol):
    """Reports total/used/free VRAM in MB for the managed GPU."""

    def total_mb(self) -> int: ...
    def used_mb(self) -> int: ...
    def free_mb(self) -> int: ...

    def hardware_faults(self) -> list[str]:
        """Hardware faults seen since the monitor started, as short descriptions; empty
        when there are none or the backend cannot tell. A GPU the driver reports as lost
        is a fault. Raises if the GPU does not answer for any other reason."""
        ...


# NVML's error code for a GPU that has fallen off the bus (NVML_ERROR_GPU_IS_LOST), and
# the message nvidia-smi prints for it.
NVML_ERROR_GPU_IS_LOST = 15
GPU_IS_LOST = "gpu is lost"

# Xid codes that NVIDIA documents as hardware faults or as needing a GPU reset. Most
# other Xids are raised by a faulting application, which a fresh process can survive.
HARDWARE_XIDS = frozenset({48, 64, 74, 79, 95, 119, 120})


class NvidiaSmiMonitor:
    """VRAM via ``nvidia-smi`` query. Works without pynvml; one subprocess per poll.

    The fallback when pynvml is not installed (:func:`make_monitor` tries
    :class:`NvmlMonitor` first)."""

    def __init__(self, index: int = 0):
        self.index = index
        self._total: int | None = None
        self._ecc_start: int | None = None
        try:
            self._ecc_start = self._optional("ecc.errors.uncorrected.volatile.total")
        except Exception:
            pass

    def _query(self, field: str) -> int:
        return int(self._raw(field))

    def _raw(self, field: str) -> str:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu={field}",
                "--format=csv,noheader,nounits",
                "-i",
                str(self.index),
            ],
            text=True,
            stderr=subprocess.PIPE,
            timeout=10,
        )
        return out.strip().splitlines()[0].strip()

    def total_mb(self) -> int:
        if self._total is None:
            self._total = self._query("memory.total")
        return self._total

    def used_mb(self) -> int:
        return self._query("memory.used")

    def free_mb(self) -> int:
        return self.total_mb() - self.used_mb()

    def _optional(self, field: str) -> int | None:
        """A counter or flag the card may not report: None for "[N/A]" or a field this
        driver's nvidia-smi does not know. A Yes/No flag reads as 1/0."""
        try:
            value = self._raw(field)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, IndexError):
            return None  # the GPU answered used_mb just before; this counter is optional
        flag = {"yes": 1, "no": 0}.get(value.lower())
        if flag is not None:
            return flag
        return int(value) if value.isdigit() else None

    def hardware_faults(self) -> list[str]:
        """Checks that the GPU answers, then the counters this nvidia-smi reports: new
        uncorrected ECC errors and row remapping. Xid events need NVML
        (:class:`NvmlMonitor`)."""
        try:
            self.used_mb()  # raises if the GPU does not answer
        except subprocess.CalledProcessError as exc:
            if GPU_IS_LOST in f"{exc.output or ''} {exc.stderr or ''}".lower():
                return ["nvidia-smi reports the GPU is lost"]
            raise
        faults: list[str] = []
        ecc = self._optional("ecc.errors.uncorrected.volatile.total")
        if ecc is not None and self._ecc_start is not None and ecc > self._ecc_start:
            faults.append(f"{ecc - self._ecc_start} new uncorrected ECC errors")
        if self._optional("remapped_rows.failure"):
            faults.append("row remapping failed")
        elif self._optional("remapped_rows.pending"):
            faults.append("a row remap is pending a GPU reset")
        return faults


class NvmlMonitor:
    """VRAM via pynvml (NVML). Lazy-imports so the package works without it.

    It also reads the GPU's fault counters where the card reports them: uncorrected ECC
    errors, row-remapping failures, and critical Xid events. Consumer cards and WSL2
    support few of these; an unsupported counter is skipped."""

    def __init__(self, index: int = 0):
        import pynvml  # noqa: F401  (raises ImportError if absent -> caller falls back)

        self._pynvml = pynvml
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        self._ecc_start = self._uncorrected_ecc()
        self._events = None
        try:
            events = pynvml.nvmlEventSetCreate()
            pynvml.nvmlDeviceRegisterEvents(
                self._handle, pynvml.nvmlEventTypeXidCriticalError, events
            )
            self._events = events
        except Exception:
            pass

    def _uncorrected_ecc(self) -> int | None:
        p = self._pynvml
        try:
            return int(
                p.nvmlDeviceGetTotalEccErrors(
                    self._handle, p.NVML_MEMORY_ERROR_TYPE_UNCORRECTED, p.NVML_VOLATILE_ECC
                )
            )
        except Exception:
            return None

    def hardware_faults(self) -> list[str]:
        p = self._pynvml
        try:
            self._info()  # raises if the GPU stopped answering
        except Exception as exc:
            if getattr(exc, "value", None) == getattr(
                p, "NVML_ERROR_GPU_IS_LOST", NVML_ERROR_GPU_IS_LOST
            ):
                return ["NVML reports the GPU is lost"]
            raise
        faults: list[str] = []
        ecc = self._uncorrected_ecc()
        if ecc is not None and self._ecc_start is not None and ecc > self._ecc_start:
            faults.append(f"{ecc - self._ecc_start} new uncorrected ECC errors")
        try:
            _corr, _unc, pending, failed = p.nvmlDeviceGetRemappedRows(self._handle)
            if failed:
                faults.append("row remapping failed")
            elif pending:
                faults.append("a row remap is pending a GPU reset")
        except Exception:
            pass
        while self._events is not None:
            try:
                data = p.nvmlEventSetWait(self._events, 0)
            except Exception:  # NVML_ERROR_TIMEOUT: no more events
                break
            if int(data.eventData) in HARDWARE_XIDS:
                faults.append(f"Xid {int(data.eventData)}")
        return faults

    def _info(self):
        return self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)

    def total_mb(self) -> int:
        return self._info().total // (1024 * 1024)

    def used_mb(self) -> int:
        return self._info().used // (1024 * 1024)

    def free_mb(self) -> int:
        return self._info().free // (1024 * 1024)


class MockGpu:
    """In-memory GPU for tests and dry-runs. Mutate ``used`` to simulate load."""

    def __init__(self, total_mb: int = 16384, used_mb: int = 800):
        self._total = total_mb
        self.used = used_mb
        self.faults: list[str] = []  # what hardware_faults reports
        self.unresponsive = False  # make every query raise, as a GPU that stopped answering

    def _answer(self) -> None:
        if self.unresponsive:
            raise RuntimeError("mock GPU does not answer")

    def total_mb(self) -> int:
        self._answer()
        return self._total

    def used_mb(self) -> int:
        self._answer()
        return self.used

    def free_mb(self) -> int:
        self._answer()
        return self._total - self.used

    def hardware_faults(self) -> list[str]:
        self._answer()
        return list(self.faults)


def make_monitor(index: int = 0) -> GpuMonitor:
    """Best-available real monitor: NVML if present, else nvidia-smi."""
    try:
        return NvmlMonitor(index)
    except Exception:
        return NvidiaSmiMonitor(index)


def wait_for_free(
    monitor: GpuMonitor,
    needed_mb: int,
    timeout_s: float,
    poll_s: float = 0.25,
    sleep=time.sleep,
    clock=time.monotonic,
) -> bool:
    """Poll until free VRAM >= *needed_mb* or *timeout_s* elapses.

    Returns True if the memory became available, False on timeout (caller must then
    refuse admission / mark the GPU degraded; SPEC 9). ``sleep``/``clock`` are
    injectable for deterministic tests."""
    deadline = clock() + timeout_s
    while True:
        try:
            if monitor.free_mb() >= needed_mb:
                return True
        except Exception:  # transient monitor hiccup (such as an nvidia-smi timeout):
            pass  # keep polling rather than abort the whole reclaim
        if clock() >= deadline:
            return False
        sleep(poll_s)
