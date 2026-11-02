#!/usr/bin/env python3
"""Serve-mode echo worker: a self-contained reference for the gpuworker serve protocol
(SPEC 8 / MODULE_GUIDE 3.2). A real module's serve worker can be modelled on this. It loads no
model; it echoes options.text to each job's declared outputs to prove the warm-container
lifecycle (load once, stream many jobs over the socket).

It is deliberately standalone (no gpuworker import), so it is also a complete example of
the framing. Inputs are read from GPUWORKER_BUNDLE/<key>/ and outputs written to
GPUWORKER_OUT/<key>/, where <key> is the frame's job_id (MODULE_GUIDE 3.2).
"""

import json
import os
import socket
import struct

PROTOCOL = 1
_H = struct.Struct(">I")


def _send(sock, obj):
    p = json.dumps(obj).encode("utf-8")
    sock.sendall(_H.pack(len(p)) + p)


def _recvn(sock, n):
    buf = b""
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        if not c:
            raise EOFError
        buf += c
    return buf


def _recv(sock):
    (n,) = _H.unpack(_recvn(sock, 4))
    return json.loads(_recvn(sock, n).decode("utf-8"))


def handle(job_id, manifest):
    """Run one job. Echo options.text into each declared output under OUT/<key>/."""
    out_root = os.environ.get("GPUWORKER_OUT", "/job/out")
    odir = os.path.realpath(os.path.join(out_root, job_id))
    text = (manifest.get("options") or {}).get("text", "")
    written = []
    for rel in manifest.get("outputs") or []:
        p = os.path.realpath(os.path.join(odir, rel))
        if os.path.isabs(rel) or not p.startswith(odir + os.sep):
            raise ValueError(f"output outside the job's output directory: {rel!r}")
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        written.append(rel)
    return "done", written, ""


def main():
    path = os.environ["GPUWORKER_SERVE_SOCKET"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        os.unlink(path)  # clear a stale socket
    except OSError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(1)
    # (a real worker loads its model here, before sending `ready`.)
    conn, _ = srv.accept()
    _send(conn, {"type": "ready", "protocol": PROTOCOL})
    while True:
        try:
            msg = _recv(conn)
        except (EOFError, OSError):
            return  # broker closed -> exit cleanly
        mtype = msg.get("type")
        if mtype == "unload":
            return
        if mtype != "run":
            _send(
                conn,
                {
                    "type": "result",
                    "job_id": msg.get("job_id", ""),
                    "status": "failed",
                    "outputs": [],
                    "message": f"bad frame {mtype!r}",
                },
            )
            continue
        jid = msg.get("job_id", "")
        try:
            status, outs, message = handle(jid, msg.get("manifest", {}))
        except Exception as exc:  # one job's failure must not kill serve
            status, outs, message = "failed", [], str(exc)
        _send(
            conn,
            {
                "type": "result",
                "job_id": jid,
                "status": status,
                "outputs": outs,
                "message": message,
            },
        )


if __name__ == "__main__":
    main()
