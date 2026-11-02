"""The broker's contract for CUDA failures (MODULE_GUIDE 3.6): a oneshot worker exits with
one of these codes, and a serve worker names the failure in its result.

The speech and audiogen modules carry this same file, since each runs in its own venv
from its own src/ (which its runtime fingerprint hashes). tests/test_example_modules.py
checks that the copies stay identical. Standard library only.
"""

from __future__ import annotations

EXIT_CUDA_OOM = 75
EXIT_CUDA_ERROR = 76
CUDA_OOM = "cuda_oom"
CUDA_ERROR = "cuda_error"
_CUDA_ERROR_MARKERS = (
    "CUDA error",
    "CUDA failed with error",  # CTranslate2 (faster-whisper)
    "CUBLAS_STATUS_",
    "CUDNN_STATUS_",
    "cuDNN error",
)
# cuBLAS could not allocate its workspace: out of memory, though it reads like an error.
_CUDA_OOM_MARKERS = ("CUBLAS_STATUS_ALLOC_FAILED",)


def _kind_of(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    if (
        type(exc).__name__ == "OutOfMemoryError"
        or ("out of memory" in lowered and "cuda" in lowered)
        or any(m in text for m in _CUDA_OOM_MARKERS)
    ):
        return CUDA_OOM
    if type(exc).__name__ == "AcceleratorError" or any(m in text for m in _CUDA_ERROR_MARKERS):
        return CUDA_ERROR
    return ""


def cuda_error_kind(exc: BaseException) -> str:
    """Classify an exception from model code: CUDA_OOM when CUDA ran out of memory,
    CUDA_ERROR for any other CUDA runtime error, after which the process's CUDA context
    cannot be trusted, and "" for everything else.

    The exceptions *exc* was raised from or while handling count too, since a library often
    wraps a CUDA failure in its own exception. If the chain holds both kinds, CUDA_ERROR
    wins: replacing a process whose context is fine costs less than keeping a broken one."""
    kinds: set[str] = set()
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        kinds.add(_kind_of(cur))
        cur = cur.__cause__ or cur.__context__
    return CUDA_ERROR if CUDA_ERROR in kinds else CUDA_OOM if CUDA_OOM in kinds else ""


def exit_code_for(exc: BaseException) -> int:
    """A oneshot worker's exit code for an unexpected exception."""
    kind = cuda_error_kind(exc)
    return EXIT_CUDA_OOM if kind == CUDA_OOM else EXIT_CUDA_ERROR if kind == CUDA_ERROR else 1
