"""Shared test helpers: configs that keep every state file under tmp_path, small
registries, and a containerless serve runtime.

The fake serve runtime pairs a socketpair per ``run()``: an in-process echo serve worker
sits on one end and the broker gets the other through a monkeypatched ``serve.connect``
(socketpair works on Windows and Linux), so serve mode is tested without containers.
"""

import os
import socket
import threading
import time

from gpuworker import gpu, serve
from gpuworker.config import Config
from gpuworker.launcher import Launcher, LocalLauncher
from gpuworker.module import JobType, Registry
from gpuworker.scheduler import Scheduler


def make_config(tmp_path, **overrides) -> Config:
    """A Config whose queue, weights, footprints and status files all live under
    *tmp_path*, so no test writes broker state into the working directory."""
    kw = dict(
        queue_roots=[str(tmp_path / "Q")],
        registry_dir="modules",
        container_runtime="local",
        weights_dir=str(tmp_path / "w"),
        footprints_path=str(tmp_path / "footprints.json"),
        status_path=str(tmp_path / "status.json"),
        state_dir=str(tmp_path / "state"),
        sandbox_image="gpuworker/sandbox:cuda12.8",
        allow_unpinned_images=True,
    )
    kw.update(overrides)
    return Config(**kw)


def make_scheduler(
    tmp_path, reg: Registry, launcher: Launcher | None = None, monitor=None, **cfg
) -> Scheduler:
    """A Scheduler over *reg* with a host-process launcher and a mock GPU by default."""
    return Scheduler(
        make_config(tmp_path, **cfg), reg, launcher or LocalLauncher(), monitor or gpu.MockGpu()
    )


def registry(*jts: JobType) -> Registry:
    reg = Registry()
    for jt in jts:
        reg.job_types[jt.id] = jt
    return reg


def cpu_jt(code="pass", **kw) -> JobType:
    """A host-process cpu job type running ``python -c <code>``."""
    return JobType(
        id=kw.pop("id", "ex.cpu"),
        resource_class="cpu",
        vram_mb=0,
        env="host",
        entrypoint=["python", "-c", code],
        sandbox="none",
        **kw,
    )


# ---- a controllable in-process serve worker + fake container runtime ------------------


def echo_serve_worker(conn, out_root, crash_on=None, hang_on=None, error_once=None):
    """Ready -> loop: echo options.text to OUT/<key>/<rel>; 'unload' or EOF -> exit.
    If a run's manifest job_id is in *crash_on*, close the socket mid-job (simulates death).
    If it is in *hang_on*, never reply (block until the broker drops the socket); this exercises
    the broker watchdog. If it is in *error_once* (job_id -> serve error kind), the first
    run of it reports that error."""
    serve.send_frame(conn, serve.ready_msg())
    while True:
        try:
            msg = serve.recv_frame(conn)
        except serve.ProtocolError:
            return
        if msg.get("type") == "unload":
            return
        if msg.get("type") != "run":
            continue
        # The frame's job_id names the job's directories; the client's id is in the manifest.
        jid, man = msg["job_id"], msg["manifest"]
        cid = man.get("job_id")
        if crash_on and cid in crash_on:
            conn.close()
            return
        if error_once and cid in error_once:
            kind = error_once.pop(cid)
            serve.send_frame(conn, serve.result_msg(jid, "failed", [], f"simulated {kind}", kind))
            continue
        if hang_on and cid in hang_on:
            while True:  # never send a result: wait for the broker's EOF
                try:
                    serve.recv_frame(conn)
                except serve.ProtocolError:
                    return
        odir = os.path.join(out_root, jid)
        text = (man.get("options") or {}).get("text", "")
        written = []
        for rel in man.get("outputs") or []:
            p = os.path.join(odir, rel)
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
            written.append(rel)
        serve.send_frame(conn, serve.result_msg(jid, "done", written, ""))


class FakeProc:
    def __init__(self, thread, ends):
        self.thread, self.ends, self.returncode = thread, ends, 0

    def wait(self, timeout=None):
        self.thread.join(timeout)

    def poll(self):
        return None if self.thread.is_alive() else 0


class FakeServe:
    def __init__(self, crash_on=None, hang_on=None, error_once=None):
        self.runs = 0
        self.crash_on = crash_on or set()
        self.hang_on = hang_on or set()
        self.error_once = error_once if error_once is not None else {}
        self._broker_ends: dict[str, socket.socket] = {}
        self._by_name: dict[str, tuple[socket.socket, socket.socket]] = {}
        self.terminated = 0
        self.killed_by_name: list[str] = []
        self.cleaned = 0

    def run(self, spec):
        self.runs += 1
        broker, worker = socket.socketpair()
        out_root = spec.out_dir
        crash = self.crash_on
        hang = self.hang_on
        errors = self.error_once

        def go():
            try:
                echo_serve_worker(worker, out_root, crash, hang, errors)
            except OSError:
                pass  # broker closed our end mid-op (respawn/evict/death races): expected
            finally:
                try:
                    worker.close()
                except OSError:
                    pass

        t = threading.Thread(target=go, daemon=True)
        t.start()
        self._broker_ends[spec.serve_socket_host] = broker
        self._by_name[spec.name] = (broker, worker)
        return FakeProc(t, (broker, worker))

    def terminate(self, proc, spec):
        self.terminated += 1
        for end in proc.ends:
            try:
                end.close()
            except OSError:
                pass

    def terminate_by_name(self, name):
        # A forced remove ends the "container": close both socket ends, as a real kill would.
        ends = self._by_name.pop(name, None)
        if ends is not None:
            self.killed_by_name.append(name)
            for end in ends:
                try:
                    end.close()
                except OSError:
                    pass

    def reap_serve_orphans(self, prefix=""):
        pass

    def reap_jobs(self, prefix="", exited_only=True):
        pass

    def cleanup(self, spec):
        self.cleaned += 1


def fake_connect(launcher):
    def connect(path, timeout=60.0, poll_s=0.1):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path in launcher._broker_ends:
                return launcher._broker_ends.pop(path)
            time.sleep(0.005)
        raise serve.ProtocolError(f"no broker end for {path}")

    return connect


def serve_jt(id="ex.serve", vram=5000, keep_warm=100.0, job_timeout=10.0, max_jobs=0):
    return JobType(
        id=id,
        resource_class="vram",
        vram_mb=vram,
        env="container",
        entrypoint=["python", "-c", "pass"],
        modes=("serve",),
        serve_socket="/run/gpuworker/job.sock",
        sandbox="standard",
        keep_warm_s=keep_warm,
        serve_job_timeout=job_timeout,
        serve_max_jobs=max_jobs,
    )


def inflight(sched, job_id):
    """The (PendingJob, deadline) watchdog entry for *job_id*, or None. Loop state is
    keyed by the broker's uid, so tests look jobs up by their submitted id here."""
    return next((v for v in sched._inflight.values() if v[0].manifest.job_id == job_id), None)


def parked_ids(sched) -> set[str]:
    """job_ids parked at the reclaim gate."""
    return {p.manifest.job_id for (p, _, _) in sched._pending_reclaim.values()}


def drive(sched, qr, fn, ticks=400):
    out = os.path.join(qr, "outbox", fn)
    for _ in range(ticks):
        sched.tick()
        if os.path.exists(out):
            sched.tick()
            return True
        time.sleep(0.005)
    return False


def serve_worker(sock, handler) -> None:
    """A minimal worker side of the serve protocol, for tests. *handler(job_id, manifest)*
    runs one job and returns (status, outputs, message). Loops until an `unload` frame or
    EOF, staying up across individual job failures (MODULE_GUIDE 3.2)."""
    serve.send_frame(sock, serve.ready_msg())
    while True:
        try:
            msg = serve.recv_frame(sock)
        except serve.ProtocolError:
            return  # EOF / socket closed -> exit cleanly
        mtype = msg.get("type")
        if mtype == "unload":
            return
        if mtype != "run":
            serve.send_frame(
                sock,
                serve.result_msg(
                    msg.get("job_id", ""), "failed", [], f"unexpected frame: {mtype!r}"
                ),
            )
            continue
        job_id = msg.get("job_id", "")
        try:
            status, outputs, message = handler(job_id, msg.get("manifest", {}))
        except Exception as exc:  # one job's failure must not kill serve
            status, outputs, message = "failed", [], str(exc)
        serve.send_frame(sock, serve.result_msg(job_id, status, outputs, message))
