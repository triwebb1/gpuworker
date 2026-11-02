"""Serve protocol broker-side + worker-side drivers over a socketpair."""

import socket
import threading

from gpuworker import serve

from helpers import serve_worker


def test_serve_session_runs_jobs_then_unloads():
    broker_sock, worker_sock = socket.socketpair()

    def handler(job_id, manifest):
        return "done", [f"out/{job_id}.bin"], "ok"

    t = threading.Thread(target=serve_worker, args=(worker_sock, handler), daemon=True)
    t.start()
    try:
        sess = serve.ServeSession(broker_sock)
        sess.wait_ready()
        res = sess.run_job("j1", {"k": 1})
        assert res.status == "done" and res.outputs == ["out/j1.bin"] and res.error == ""
        assert sess.run_job("j2", {}).status == "done"
        sess.unload()
    finally:
        t.join(timeout=2)
        broker_sock.close()
        worker_sock.close()


def test_serve_worker_survives_job_failure():
    broker_sock, worker_sock = socket.socketpair()

    def handler(job_id, manifest):
        if job_id == "bad":
            raise RuntimeError("job blew up")
        return "done", [], "ok"

    t = threading.Thread(target=serve_worker, args=(worker_sock, handler), daemon=True)
    t.start()
    try:
        sess = serve.ServeSession(broker_sock)
        sess.wait_ready()
        res = sess.run_job("bad", {})
        assert res.status == "failed" and "job blew up" in res.message
        # worker stays up for the next job
        assert sess.run_job("good", {}).status == "done"
        sess.unload()
    finally:
        t.join(timeout=2)
        broker_sock.close()
        worker_sock.close()
