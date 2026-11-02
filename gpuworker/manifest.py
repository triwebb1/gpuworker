"""Work-package job manifest: the per-job JSON that names a module job type.

Module-agnostic on purpose (SPEC 4.2): the footprint/env/image come from the module
job type, not the job. Per-model knobs ride in `options`. Every job reaches a
terminal state with a message (the reliability contract, SPEC 10).
"""

from __future__ import annotations

import datetime as _dt
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any

# Manifest schema version (SPEC 4.3). Readers reject newer-major manifests rather
# than silently mis-handling them.
MANIFEST_SCHEMA = 1

# A job_id becomes a path component on the broker host (per-job work dirs, worker logs,
# serve staging), and a manifest may be malformed (SPEC 6.3). So it is a plain
# name: no separators, no leading dot, nothing that could resolve outside its parent.
_JOB_ID_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def is_valid_job_id(job_id: object) -> bool:
    return isinstance(job_id, str) and bool(_JOB_ID_RE.match(job_id))


# Characters that some filesystem or reader treats as more than part of a name: a
# backslash is a separator to a zip reader and to Windows, a colon names a Windows drive
# or alternate data stream, and the rest are refused or special on Windows.
_NON_PORTABLE = frozenset('\\:*?"<>|')
_WINDOWS_DEVICES = frozenset(
    {"CON", "PRN", "AUX", "NUL"} | {f"{p}{i}" for p in ("COM", "LPT") for i in range(1, 10)}
)
MAX_PATH_CHARS = 1024


def is_bundle_path(rel: object) -> bool:
    """A path inside a bundle, in the one spelling every reader agrees on (SPEC 6.3):
    relative, forward slashes only, no empty, ``.`` or ``..`` part, no control or
    non-portable character, and no part ending in a dot or a space, which Windows drops.
    Any other spelling could name the same file as a path the broker checked, so it is
    refused rather than normalized. On Windows, device names such as ``NUL`` are refused
    too."""
    if not isinstance(rel, str) or not rel or len(rel) > MAX_PATH_CHARS:
        return False
    if rel.startswith("/") or any(c in _NON_PORTABLE or ord(c) < 32 for c in rel):
        return False
    for part in rel.split("/"):
        if part in ("", ".", "..") or part[-1] in ". ":
            return False
        if os.name == "nt" and part.split(".")[0].upper() in _WINDOWS_DEVICES:
            return False
    return True


# Top-level result-bundle directories the broker writes itself: job manifests
# (bundle.PENDING_PREFIX) and worker logs (worker_log.WORKER_LOG_PREFIX). A declared
# output may not live under either.
RESERVED_OUTPUT_DIRS = ("jobs", "_gpuworker")

# A job lists at most this many inputs and this many outputs. Each output becomes a
# member of the result bundle, which the broker's member limit also bounds per bundle.
MAX_PATHS_PER_JOB = 10_000

# Submitter priority is a small signed int. The bound keeps it a ranking, not a lever;
# a broker can cap it further per queue (Config.max_priority).
PRIORITY_LIMIT = 1000


# Terminal + non-terminal statuses. `done`/`failed` are the only terminal ones.
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
TERMINAL_STATUSES = frozenset({STATUS_DONE, STATUS_FAILED})


def now_iso() -> str:
    """UTC timestamp, ISO-8601, second precision."""
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat()


class ManifestError(ValueError):
    """A job manifest is malformed or fails validation."""


@dataclass
class JobManifest:
    """One job. Mirrors SPEC 4.2; `result` is filled on completion."""

    job_id: str
    job_type: str
    schema: int = MANIFEST_SCHEMA
    priority: int = 0
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)
    deliver_to: str | None = None
    timeout_s: int = 0
    # Lifecycle (broker-managed):
    status: str = STATUS_QUEUED
    submitted_utc: str = field(default_factory=now_iso)
    finished_utc: str | None = None
    result: dict[str, Any] | None = None
    message: str = ""

    # ---- (de)serialization -------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> JobManifest:
        """Validate and build a manifest from a parsed dict. Strict: a mistaken or
        malformed job should fail here with a reason (SPEC 6.3), not later."""
        if not isinstance(d, dict):
            raise ManifestError("manifest is not an object")
        schema = d.get("schema", MANIFEST_SCHEMA)
        if not isinstance(schema, int) or schema > MANIFEST_SCHEMA:
            raise ManifestError(f"unsupported manifest schema: {schema!r}")
        job_id = d.get("job_id")
        if not isinstance(job_id, str) or not job_id.strip():
            raise ManifestError("job_id must be a non-empty string")
        if not _JOB_ID_RE.match(job_id.strip()):
            raise ManifestError(
                f"job_id must match [A-Za-z0-9][A-Za-z0-9._-]{{0,127}}, got {job_id!r}"
            )
        job_type = d.get("job_type")
        if not isinstance(job_type, str) or not job_type.strip():
            raise ManifestError("job_type must be a non-empty string")

        def _paths(key: str) -> list[str]:
            v = d.get(key, [])
            if v is None:
                return []
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                raise ManifestError(f"{key} must be a list of strings")
            if len(v) > MAX_PATHS_PER_JOB:
                raise ManifestError(f"{key}: more than {MAX_PATHS_PER_JOB} paths")
            for rel in v:
                if not is_bundle_path(rel):
                    raise ManifestError(
                        f"{key}: {rel[:200]!r} is not a plain relative path (SPEC 6.3)"
                    )
            return list(v)

        priority = d.get("priority", 0)
        if (
            not isinstance(priority, int)
            or isinstance(priority, bool)
            or abs(priority) > PRIORITY_LIMIT
        ):
            raise ManifestError(f"priority must be an int in [-{PRIORITY_LIMIT}, {PRIORITY_LIMIT}]")
        timeout_s = d.get("timeout_s", 0)
        # Upper-bound it: a mistyped value (SPEC 6.3) far out of range would
        # overflow socket.settimeout's C timeval in the serve lane, escaping run_job's
        # (ProtocolError, OSError) catch and leaving a job with no completion and a watchdog
        # deadline that never fires. 1e7 s (~116 days) is far past any real job.
        if (
            not isinstance(timeout_s, int)
            or isinstance(timeout_s, bool)
            or timeout_s < 0
            or timeout_s > 10_000_000
        ):
            raise ManifestError("timeout_s must be a non-negative int <= 10_000_000")
        options = d.get("options", {})
        if not isinstance(options, dict):
            raise ManifestError("options must be an object")
        deliver_to = d.get("deliver_to")
        if deliver_to is not None and not isinstance(deliver_to, str):
            raise ManifestError("deliver_to must be a string or null")
        # Lifecycle fields are broker-owned (intake overwrites them), but they still arrive
        # from the submitter, so a wrong type is rejected rather than carried into state.
        for key in ("status", "submitted_utc", "finished_utc", "message"):
            if d.get(key) is not None and not isinstance(d[key], str):
                raise ManifestError(f"{key} must be a string or null")
        if d.get("result") is not None and not isinstance(d["result"], dict):
            raise ManifestError("result must be an object or null")

        return cls(
            job_id=job_id.strip(),
            job_type=job_type.strip(),
            schema=schema,
            priority=priority,
            inputs=_paths("inputs"),
            outputs=_paths("outputs"),
            options=options,
            deliver_to=deliver_to,
            timeout_s=timeout_s,
            status=d.get("status", STATUS_QUEUED),
            submitted_utc=d.get("submitted_utc") or now_iso(),
            finished_utc=d.get("finished_utc"),
            result=d.get("result"),
            message=d.get("message", "") or "",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # ---- transitions -------------------------------------------------------

    def mark_running(self) -> None:
        self.status = STATUS_RUNNING
        self.message = "running"

    def mark_done(self, result: dict[str, Any] | None = None, message: str = "done") -> None:
        self.status = STATUS_DONE
        self.result = result
        self.message = message
        self.finished_utc = now_iso()

    def mark_failed(self, message: str) -> None:
        self.status = STATUS_FAILED
        self.result = None
        self.message = message
        self.finished_utc = now_iso()

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES
