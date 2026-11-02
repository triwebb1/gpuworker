"""Job I/O shared by the speech workers: bundle-relative paths, option checks, WAV output
and the runtime fingerprint. Only write_wav needs numpy; nothing here needs the model."""

from __future__ import annotations

import hashlib
import os
import wave
from typing import Any

from .cuda_errors import (  # noqa: F401  (the workers import these from here)
    CUDA_ERROR,
    CUDA_OOM,
    EXIT_CUDA_ERROR,
    EXIT_CUDA_OOM,
    cuda_error_kind,
    exit_code_for,
)

SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def src_fingerprint(src: str = SRC_ROOT) -> str:
    """The marker line for MODULE_GUIDE 1.5: a digest of the source this process runs,
    computed the way the broker hashes the module's src/ (gpuworker.module)."""
    h, count = hashlib.sha256(), 0
    for root, dirs, files in os.walk(src):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for fn in sorted(files):
            if fn.endswith((".pyc", ".pyo")):
                continue
            full = os.path.join(root, fn)
            h.update(os.path.relpath(full, src).replace(os.sep, "/").encode("utf-8") + b"\0")
            try:
                with open(full, "rb") as fh:
                    h.update(fh.read())
            except OSError:
                h.update(b"<unreadable>")
            count += 1
    return f"GPUWORKER_RUNTIME_MODULE src_sha256={h.hexdigest()} files={count}"


class BadJob(ValueError):
    """The job's own input is wrong. The worker fails that job and keeps running."""


def job_path(root: str, rel: str) -> str:
    """Join a bundle-relative path under root, refusing anything that leaves it.

    In serve mode one container sees the inputs of every job it serves, so a path like
    ../<other job>/input.wav must not resolve.
    """
    if not isinstance(rel, str) or not rel or os.path.isabs(rel) or "\\" in rel:
        raise BadJob(f"bad input path {rel!r}")
    parts = rel.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise BadJob(f"bad input path {rel!r}")
    return os.path.join(root, *parts)


def pick_output(outputs: list[str], suffix: str, required: bool = True) -> str | None:
    """The first declared output with this suffix."""
    for rel in outputs:
        if rel.lower().endswith(suffix):
            return rel
    if required:
        raise BadJob(f"declare an output ending in {suffix}")
    return None


def opt_str(options: dict, key: str, default: str | None = None) -> str | None:
    value = options.get(key, default)
    if value is not None and not isinstance(value, str):
        raise BadJob(f"options.{key} must be a string")
    return value


def opt_num(options: dict, key: str, default: Any, lo: float, hi: float, kind: type) -> Any:
    """A bounded int or float option. bool is refused even though it is an int. A numeric
    string is accepted, because `gpuworker run-once --option k=v` passes strings."""
    value = options.get(key, default)
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = kind(value)
        except ValueError:
            raise BadJob(f"options.{key} must be a {kind.__name__}") from None
    ok = isinstance(value, kind) or (kind is float and isinstance(value, int))
    if not ok or isinstance(value, bool) or not lo <= value <= hi:
        raise BadJob(f"options.{key} must be a {kind.__name__} in [{lo}, {hi}]")
    return kind(value)


def cpu_budget() -> int:
    """Threads to use: the broker's budget, not os.cpu_count(), which ignores --cpus."""
    try:
        return max(1, int(os.environ.get("GPUWORKER_CPU_BUDGET", "")))
    except ValueError:
        return 1


def write_wav(path: str, samples: Any, sample_rate: int) -> float:
    """Write float samples in [-1, 1] as 16-bit PCM and return the duration in seconds.

    samples is a numpy array shaped (frames,) or (frames, channels).
    """
    import numpy as np

    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    if data.size == 0 or not np.all(np.isfinite(data)):
        raise RuntimeError("generated audio is empty or not finite")
    pcm = (np.clip(data, -1.0, 1.0) * 32767.0).round().astype("<i2")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with wave.open(path, "wb") as fh:
        fh.setnchannels(pcm.shape[1])
        fh.setsampwidth(2)
        fh.setframerate(int(sample_rate))
        fh.writeframes(pcm.tobytes())
    return pcm.shape[0] / float(sample_rate)
