"""The broker's serve protocol (MODULE_GUIDE 3.2), as a loop a model worker plugs into.
Same framing as modules/echo/src/serve_worker.py: a 4-byte big-endian length, then JSON."""

from __future__ import annotations

import gc
import json
import os
import socket
import struct
import sys
from collections.abc import Callable

from .jobio import CUDA_ERROR, CUDA_OOM, cuda_error_kind

PROTOCOL = 1
_HEADER = struct.Struct(">I")

# handle(job_id, manifest) -> (status, outputs, message)
Handler = Callable[[str, dict], tuple[str, list[str], str]]


def send(sock: socket.socket, obj: dict) -> None:
    payload = json.dumps(obj).encode("utf-8")
    sock.sendall(_HEADER.pack(len(payload)) + payload)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise EOFError
        buf += chunk
    return buf


def recv(sock: socket.socket) -> dict:
    (n,) = _HEADER.unpack(_recv_exact(sock, _HEADER.size))
    return json.loads(_recv_exact(sock, n).decode("utf-8"))


def run(load: Callable[[], Handler], release: Callable[[], None] | None = None) -> int:
    """Bind, load the model, report ready, then serve run frames until unload or EOF.

    The socket is bound before the model loads so the broker's short connect timeout is
    met while the load is still running. A load error propagates and the process exits
    non-zero before `ready`, which is what the broker expects from a broken model.

    A job that fails on a CUDA error says so in its result (`error`). After running out
    of memory the loop returns the job's cached blocks to the driver and stays up; the
    broker keeps the container and may retry the job on it. After any other CUDA error
    the context is unusable, so the loop exits non-zero and the broker replaces the
    process.
    """
    path = os.environ["GPUWORKER_SERVE_SOCKET"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(1)
    handle = load()
    conn, _ = srv.accept()
    send(conn, {"type": "ready", "protocol": PROTOCOL})
    try:
        while True:
            try:
                msg = recv(conn)
            except (EOFError, OSError):
                return 0
            if msg.get("type") == "unload":
                return 0
            job_id = str(msg.get("job_id", ""))
            error = ""
            if msg.get("type") != "run":
                status, outputs, message = "failed", [], f"bad frame {msg.get('type')!r}"
            else:
                try:
                    status, outputs, message = handle(job_id, msg.get("manifest") or {})
                except Exception as exc:  # one bad job must not take the warm model down
                    status, outputs, message = "failed", [], f"{type(exc).__name__}: {exc}"
                    error = cuda_error_kind(exc)
            if error == CUDA_OOM:
                _free_cached_memory()
            print(f"[{job_id}] {status}: {message}", file=sys.stderr, flush=True)
            result = {
                "type": "result",
                "job_id": job_id,
                "status": status,
                "outputs": outputs,
                "message": message,
            }
            if error:
                result["error"] = error
            send(conn, result)
            if error == CUDA_ERROR:
                return 1
    finally:
        if release is not None:
            release()


def _free_cached_memory() -> None:
    """Give the memory a failed job left in torch's cache back to the driver, so the next
    job, or another model, can use it. Collect first: tensors the failed job still holds
    through reference cycles (a saved traceback, for one) are not in the cache yet. Best
    effort: this module does not need torch."""
    try:
        import torch

        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass
