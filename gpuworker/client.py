"""Client-side submit helper (SPEC 7). A client uses this to drop a job bundle into a
queue inbox and optionally wait for the terminal result; the broker is a separate
process. Preserves the reliability contract: a ``--wait`` submit always returns a
terminal manifest (or times out with a clear error), never hangs silently.
"""

from __future__ import annotations

import os
import shutil
import time
import uuid

from . import bundle
from .manifest import JobManifest
from .share import QueueTree


def submit(
    queue_root: str,
    manifests: list[JobManifest],
    inputs_root: str = ".",
    extra_members: list[str] | None = None,
) -> str:
    """Pack *manifests* (and any *extra_members* input files relative to
    *inputs_root*) into a bundle and drop it into the queue inbox atomically.
    Returns the bundle filename."""
    q = QueueTree(queue_root).ensure()
    # First job id + UTC time + a random suffix: readable in a directory listing, and
    # unique, so resubmitting a job_id can never collide with (or be answered by) an
    # earlier bundle's result still sitting in outbox/.
    base = manifests[0].job_id if manifests else "batch"
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    fn = f"{base}-{stamp}-{uuid.uuid4().hex[:8]}.zip"
    tmp = os.path.join(q.dirs["inbox"], fn + ".tmp")
    bundle.pack(
        tmp, [m.to_dict() for m in manifests], root=inputs_root, extra_members=extra_members
    )
    final = os.path.join(q.dirs["inbox"], fn)
    os.replace(tmp, final)
    return fn


def wait_result(
    queue_root: str, filename: str, timeout_s: float = 600.0, poll_s: float = 0.5
) -> list[JobManifest]:
    """Block until the broker writes the result bundle for *filename* to outbox,
    then return the terminal manifests. Raises TimeoutError if it never lands."""
    q = QueueTree(queue_root)
    out_zip = os.path.join(q.dirs["outbox"], filename)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if os.path.exists(out_zip):
            raw = bundle.read_pending_manifests(out_zip)
            mans = [JobManifest.from_dict(r) for r in raw]
            if mans and all(m.is_terminal for m in mans):
                return mans
        time.sleep(poll_s)
    raise TimeoutError(f"no terminal result for {filename} within {timeout_s}s")


def _atomic_copy(src: str, dst: str) -> None:
    """Copy src->dst atomically (tmp + replace) so a crash mid-copy can't leave a
    truncated external file. Idempotent: re-running lands the same bytes."""
    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    tmp = dst + ".tmp"
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def collect(
    queue_root: str, filename: str, dest_root: str, timeout_s: float = 600.0, poll_s: float = 0.5
) -> list[JobManifest]:
    """Full collect: wait for the terminal result, extract the output files
    from the result bundle into *dest_root* (at their declared relpaths), and honor
    each job's ``deliver_to`` by copying its primary output to that external path
    (atomic, idempotent). Delivery happens here on the client, where an external
    host path is meaningful (the broker runs sandboxed and can't resolve it; SPEC
    4.2, 10). Returns the terminal manifests, with ``result['delivered']`` annotated
    where a deliver_to was honored. The standard collect path (no deliver_to) just
    unzips the outputs into *dest_root*."""
    mans = wait_result(queue_root, filename, timeout_s, poll_s)
    out_zip = os.path.join(QueueTree(queue_root).dirs["outbox"], filename)
    bundle.extract_outputs(out_zip, dest_root)
    for m in mans:
        outputs = (m.result or {}).get("outputs") if m.result else None
        if m.deliver_to and outputs and m.result is not None:
            primary = os.path.join(dest_root, outputs[0])
            if os.path.isfile(primary):
                _atomic_copy(primary, m.deliver_to)
                m.result["delivered"] = m.deliver_to
    return mans
