"""Serve-mode wire protocol: length-prefixed JSON frames over a bind-mounted unix
socket. The protocol is specified in SPEC 8 and the worker's side of it in
docs/MODULE_GUIDE.md section 3.2; this module implements both ends.

The broker side is connect -> ``ServeSession(conn)`` -> ``wait_ready`` -> ``run_job``
per job -> ``unload``. Workers implement their side themselves (see
modules/echo/src/serve_worker.py); the frame helpers here are shared with the tests.

This module is transport-agnostic: ``send_frame``/``recv_frame`` operate on any
object with ``sendall``/``recv`` (a connected socket) so the framing is unit-testable
over a socketpair without containers.
"""

from __future__ import annotations

import json
import os
import socket
import stat
import struct
import time
from typing import Any, NamedTuple

from . import PROTOCOL_VERSION

_HEADER = struct.Struct(">I")
MAX_FRAME = 8 * 1024 * 1024  # control frames are tiny; guard against a bad length
MAX_MESSAGE_CHARS = 64 * 1024  # a result message is a summary, not a log
UNLOAD_TIMEOUT_S = 5.0  # the container is removed next either way

# Optional `error` values a worker may put in a failed result (MODULE_GUIDE 3.2). After
# cuda_oom the container stays warm, and the job is retried on it unless it had the GPU to
# itself. After cuda_error the broker replaces the container and retries the job there.
ERROR_CUDA_OOM = "cuda_oom"  # CUDA ran out of memory; the worker frees it and stays up
ERROR_CUDA = "cuda_error"  # any other CUDA error: the process's CUDA context is unusable
ERROR_KINDS = frozenset({ERROR_CUDA_OOM, ERROR_CUDA})


class ProtocolError(Exception):
    """Malformed frame, oversized frame, or unexpected EOF."""


class ServeTimeout(ProtocolError):
    """A serve job did not answer within its timeout. Still a ProtocolError (the
    container is torn down either way), but the broker must not retry it: a job that
    overran once is not an infrastructure failure (SPEC 10)."""


class ServeResult(NamedTuple):
    """A validated `result` frame."""

    status: str  # "done" | "failed"
    outputs: list[str]  # informational; the broker collects the declared outputs itself
    message: str
    error: str = ""  # "" or one of ERROR_KINDS


# ---- framing ---------------------------------------------------------------


def send_frame(sock, obj: dict[str, Any]) -> None:
    payload = json.dumps(obj).encode("utf-8")
    if len(payload) > MAX_FRAME:
        raise ProtocolError(f"frame too large: {len(payload)} bytes")
    sock.sendall(_HEADER.pack(len(payload)) + payload)


def _recv_exact(sock, n: int, deadline: float | None = None) -> bytes:
    """Read exactly *n* bytes. With *deadline* (a time.monotonic() value) the whole read
    must finish by then, raising TimeoutError otherwise: a socket timeout alone bounds
    each recv, so a peer sending a byte at a time could stretch it without limit."""
    buf = bytearray()
    while len(buf) < n:
        if deadline is not None:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError("serve socket read passed its deadline")
            sock.settimeout(left)
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ProtocolError("unexpected EOF on serve socket")
        buf.extend(chunk)
    return bytes(buf)


def recv_frame(sock, deadline: float | None = None) -> dict[str, Any]:
    """Read one frame, by *deadline* if given (see :func:`_recv_exact`)."""
    (length,) = _HEADER.unpack(_recv_exact(sock, _HEADER.size, deadline))
    if length > MAX_FRAME:
        raise ProtocolError(f"frame too large: {length} bytes")
    try:
        obj = json.loads(_recv_exact(sock, length, deadline).decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        # A garbled frame is a protocol violation like any other, and the broker's
        # handlers catch ProtocolError; a JSONDecodeError, or the RecursionError of a
        # deeply nested document, would escape them.
        raise ProtocolError(f"frame is not UTF-8 JSON: {type(exc).__name__}") from exc
    if not isinstance(obj, dict) or "type" not in obj:
        raise ProtocolError("frame is not a typed object")
    return obj


# ---- message constructors --------------------------------------------------


def ready_msg() -> dict[str, Any]:
    return {"type": "ready", "protocol": PROTOCOL_VERSION}


def run_msg(job_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
    return {"type": "run", "job_id": job_id, "manifest": manifest}


def result_msg(
    job_id: str,
    status: str,
    outputs: list[str] | None = None,
    message: str = "",
    error: str = "",
) -> dict[str, Any]:
    msg: dict[str, Any] = {
        "type": "result",
        "job_id": job_id,
        "status": status,
        "outputs": outputs or [],
        "message": message,
    }
    if error:
        msg["error"] = error
    return msg


def parse_result(msg: dict[str, Any], job_id: str) -> ServeResult:
    """Validate a `result` frame for *job_id*. Every field is type-checked, because the
    values end up in the loop's state and the client's result, and a buggy worker
    should not corrupt them: a bad one raises ProtocolError, like any other protocol
    violation."""
    if msg.get("type") != "result":
        raise ProtocolError(f"expected 'result', got {msg.get('type')!r}")
    if msg.get("job_id") != job_id:
        raise ProtocolError(f"result job_id mismatch: {msg.get('job_id')!r}")
    status = msg.get("status")
    if status not in ("done", "failed"):
        raise ProtocolError(f"result status must be 'done' or 'failed', got {status!r}")
    outputs = msg.get("outputs", [])
    if not isinstance(outputs, list) or not all(isinstance(o, str) for o in outputs):
        raise ProtocolError("result outputs must be a list of strings")
    message = msg.get("message", "")
    if not isinstance(message, str):
        raise ProtocolError("result message must be a string")
    error = msg.get("error", "")
    if error and error not in ERROR_KINDS:
        raise ProtocolError(f"result error must be one of {sorted(ERROR_KINDS)}, got {error!r}")
    if error and status != "failed":
        raise ProtocolError("a result with an error must have status 'failed'")
    return ServeResult(status, list(outputs), message[:MAX_MESSAGE_CHARS], error or "")


def unload_msg() -> dict[str, Any]:
    return {"type": "unload"}


def check_ready(msg: dict[str, Any]) -> None:
    """Validate a 'ready' frame and its protocol version (SPEC 4.3)."""
    if msg.get("type") != "ready":
        raise ProtocolError(f"expected 'ready', got {msg.get('type')!r}")
    peer = msg.get("protocol")
    if peer != PROTOCOL_VERSION:
        raise ProtocolError(f"serve protocol mismatch: broker={PROTOCOL_VERSION} module={peer}")


# ---- broker-side driver ----------------------------------------------------


# Linux: open the socket's path once with O_PATH and connect through /proc/self/fd, so the
# inode that was checked is the one connected to.
_O_PATH = getattr(os, "O_PATH", 0)


def _connect_checked(sock: socket.socket, sock_path: str) -> None:
    """Connect *sock* to *sock_path*, refusing anything that is not a socket. The socket's
    directory is writable by the container's model code, so what sits at the path may not
    be the socket the worker bound.

    On Linux the path is opened once with ``O_PATH | O_NOFOLLOW``, checked with ``fstat``
    and connected through ``/proc/self/fd/<n>``, so the inode checked is the one connected
    to. Elsewhere it is an ``lstat`` and then a connect by path, two steps (SPEC 6.3)."""
    if _O_PATH and os.path.isdir("/proc/self/fd"):
        fd = os.open(sock_path, _O_PATH | os.O_NOFOLLOW)  # a symlink opens as itself
        try:
            if not stat.S_ISSOCK(os.fstat(fd).st_mode):
                raise ProtocolError(f"serve socket {sock_path} is not a socket")
            sock.connect(f"/proc/self/fd/{fd}")
        finally:
            os.close(fd)
        return
    # Elsewhere the check and the connect are two steps (SPEC 6.3, known limits).
    if not stat.S_ISSOCK(os.lstat(sock_path).st_mode):
        raise ProtocolError(f"serve socket {sock_path} is not a socket")
    sock.connect(sock_path)


def connect(sock_path: str, timeout: float = 60.0, poll_s: float = 0.1):
    """Broker side: connect to the serve worker's unix socket, retrying until the worker
    has bound and listened on it (it should bind before loading its model). Returns a
    connected socket, or raises ProtocolError if it never comes up within *timeout*, or
    at once if the path is not a socket."""
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            _connect_checked(sock, sock_path)
            return sock
        except ProtocolError:
            sock.close()
            raise
        except OSError as exc:
            last_err = exc
            sock.close()
            time.sleep(poll_s)
    raise ProtocolError(f"serve socket {sock_path} not reachable in {timeout}s: {last_err}")


class ServeSession:
    """Broker side of a warm serve container. Wraps a connected socket; the scheduler
    keeps one per resident serve job type and streams `run` frames through it."""

    def __init__(self, sock):
        self.sock = sock

    def wait_ready(self, timeout: float | None = None) -> None:
        """Block for the worker's `ready` frame. *timeout* (seconds) bounds the whole
        model-load wait; on expiry raise ProtocolError so the broker can tear the container
        down."""
        deadline = None if timeout is None else time.monotonic() + timeout
        try:
            check_ready(recv_frame(self.sock, deadline))
        except TimeoutError as exc:
            raise ProtocolError("serve worker did not become ready in time") from exc
        finally:
            self.sock.settimeout(None)

    def run_job(
        self, job_id: str, manifest: dict[str, Any], timeout: float | None = None
    ) -> ServeResult:
        """Send one job, block for its result (bounded by *timeout*), and return it
        validated (:func:`parse_result`). On timeout raise ServeTimeout: a job that wedges
        the model can't be trusted, so the caller tears the container down."""
        deadline = None if timeout is None else time.monotonic() + timeout
        self.sock.settimeout(timeout)
        try:
            send_frame(self.sock, run_msg(job_id, manifest))
            msg = recv_frame(self.sock, deadline)
        except TimeoutError as exc:
            raise ServeTimeout(f"serve job {job_id} timed out after {timeout}s") from exc
        finally:
            self.sock.settimeout(None)
        return parse_result(msg, job_id)

    def unload(self) -> None:
        """Ask the worker to unload its model. Best effort, within UNLOAD_TIMEOUT_S: a
        worker that has stopped reading its socket leaves the send blocked on a full
        buffer, and the teardown after this must still run."""
        try:
            self.sock.settimeout(UNLOAD_TIMEOUT_S)
            send_frame(self.sock, unload_msg())
        except OSError:
            pass
