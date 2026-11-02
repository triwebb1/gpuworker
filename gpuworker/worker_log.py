"""The per-job worker log shipped inside each result bundle.

A client that unzips a result gets the terminal manifests, the declared outputs, and
``_gpuworker/<job_id>.log``: the worker's own stdout/stderr plus a header saying which
module source the broker staged and, if the worker reports it, which source actually ran.
"""

from __future__ import annotations

import os
import re

from . import bundle
from .jobs import PendingJob
from .module import Module

# Reserved bundle prefix for the per-job worker log. It is deliberately not a declared
# output: a job type's `outputs` list drives the "exited 0 but produced no declared outputs"
# false-success check, and a log that always exists would defeat it.
WORKER_LOG_PREFIX = "_gpuworker/"
# Per stream. Well above the 2000-char tail in gpuworker.log, because this copy is the
# client's only view of the worker, and still bounded so a chatty job cannot inflate a
# result bundle without limit.
WORKER_LOG_CAP = 200_000

_RUNTIME_SHA_RE = re.compile(rb'src_sha256["\'\s:=]+([0-9a-f]{64})')


def clip_both_ends(text: str, cap: int) -> str:
    """Keep the head and the tail when clipping. On a chained traceback the tail holds the
    last exception and the head holds the original one, which is usually the root cause."""
    if len(text) <= cap:
        return text
    head = cap // 2
    tail = cap - head
    return f"{text[:head]}\n... [{len(text) - cap} chars elided] ...\n{text[-tail:]}"


def runtime_vs_staged(mod: Module, streams: dict[str, bytes]) -> str:
    """Compare the fingerprint of the module source the broker staged against the one the
    worker reported at runtime. They can differ by a whole container image: the broker
    resolves the sandbox by pinned digest, so rebuilding and retagging an image does not
    change what runs. Putting both on one line makes that visible.

    A worker opts in by printing its digest, computed with the same recipe as
    :func:`gpuworker.module.source_fingerprint`, for example::

        GPUWORKER_RUNTIME_MODULE src_sha256=<64 hex> files=<n>

    A worker that prints no digest gets a line saying so, not a warning: reporting it is
    optional.
    """
    try:
        blob = b"\n".join(streams.get(k) or b"" for k in ("stdout", "stderr"))
        m = _RUNTIME_SHA_RE.search(blob)
        if not m:
            return "# runtime module: not reported by this worker (see MODULE_GUIDE 1.5)"
        runtime = m.group(1).decode()
        if not mod.src_fingerprint:
            return f"# runtime module: src_sha256={runtime} (broker staged no src/)"
        if runtime == mod.src_fingerprint:
            return f"# runtime module: MATCHES staged ({runtime[:16]}...)"
        return (
            "# runtime module: *** MISMATCH *** the code this job EXECUTED is not "
            f"the code the broker staged.\n#   staged  {mod.src_fingerprint}\n"
            f"#   runtime {runtime}\n"
            "#   Usually a stale container image: the broker may resolve the "
            "sandbox by pinned digest, so re-baking and retagging is invisible to "
            "it. Check the resolved image, not the tag you built."
        )
    except Exception:  # a diagnostic line must never fail a job
        return "# runtime module: comparison failed (diagnostic only)"


def write_worker_log(
    pj: PendingJob, streams: dict[str, bytes], note: str, mod: Module | None
) -> str:
    """Write the worker's stdout/stderr into the job's out dir, where the result bundle
    packs it. Returns the bundle-relative path. Raises on I/O errors; the caller decides
    that a log it cannot write must not fail an otherwise successful job."""
    rel = f"{WORKER_LOG_PREFIX}{pj.manifest.job_id}.log"
    dest = bundle.safe_join(pj.batch.out_dir, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    parts = [f"# gpuworker worker log job_id={pj.manifest.job_id} job_type={pj.jt.id}"]
    if mod is not None:
        parts.append(runtime_vs_staged(mod, streams))
        parts.append(
            f"# module: {mod.name} v{mod.version} "
            f"src_sha256={mod.src_fingerprint or '(no src/)'} "
            f"files={mod.src_files} newest={mod.src_newest_mtime or '-'}"
        )
    if note:
        parts.append(f"# outcome: {note}")
    for name in ("stdout", "stderr"):
        text = (streams.get(name) or b"").decode("utf-8", "replace").strip()
        parts.append(f"\n===== {name}: {len(text)} chars =====")
        parts.append(clip_both_ends(text, WORKER_LOG_CAP) if text else "(empty)")
    if not streams:
        # No streams at all means they were never read (the launch raised before the
        # process existed, or a kill's salvage read failed), not that the worker was silent.
        parts.append("\n(worker streams were never captured; see 'outcome' above)")
    with open(dest, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts) + "\n")
    return rel
