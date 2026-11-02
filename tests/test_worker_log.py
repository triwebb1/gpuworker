"""The worker's own output must reach the client, inside the result bundle.

Without it the client sees only a 500-char stderr tail in the manifest message on
failure, and nothing on success; the root cause of a chained traceback is often at the
head, which that tail drops.

These tests validate the delivered artifact: they unzip the outbox bundle and read the
log out of it, rather than trusting an exit code or a broker-side log line.
"""

import os
import socket
import threading
import time

from gpuworker import bundle, gpu, serve
from gpuworker.client import submit, wait_result
from gpuworker.launcher import make_launcher
from gpuworker.manifest import JobManifest
from gpuworker.module import JobType, Registry
from gpuworker.scheduler import Batch, PendingJob, Scheduler
from gpuworker.serve_manager import ServeContainer
from gpuworker.worker_log import WORKER_LOG_CAP, WORKER_LOG_PREFIX, clip_both_ends

from helpers import make_config


def _jt(entrypoint, id="cpu.log"):
    return JobType(
        id=id,
        resource_class="cpu",
        vram_mb=0,
        env="host",
        entrypoint=entrypoint,
        modes=("oneshot",),
        sandbox="none",
    )


def _sched(tmp_path, jt):
    reg = Registry()
    reg.job_types[jt.id] = jt
    cfg = make_config(
        tmp_path,
        queue_roots=[str(tmp_path / "Q")],
        registry_dir="modules",
        container_runtime="local",
        weights_dir=str(tmp_path / "w"),
    )
    return Scheduler(cfg, reg, make_launcher("local"), gpu.MockGpu())


def _run(tmp_path, jt, manifest):
    """Submit one job, drive the scheduler to a shipped result, return (results, members)."""
    queue_root = str(tmp_path / "Q")
    fn = submit(queue_root, [manifest])
    sched = _sched(tmp_path, jt)
    out_zip = os.path.join(queue_root, "outbox", fn)
    # The worker runs on its own thread, so the ticks have to be paced: a tight spin
    # finishes long before a python subprocess has even started.
    for _ in range(200):
        sched.tick()
        if os.path.exists(out_zip):
            break
        time.sleep(0.05)
    assert os.path.exists(out_zip), "broker never shipped a result bundle"
    results = wait_result(queue_root, fn, timeout_s=5)
    dest = str(tmp_path / "collected")
    members = bundle.extract_outputs(out_zip, dest)
    return results, members, dest


def _log_text(members, dest, job_id):
    rel = f"{WORKER_LOG_PREFIX}{job_id}.log"
    assert rel in members, f"worker log missing from the bundle; got {members}"
    with open(os.path.join(dest, *rel.split("/")), encoding="utf-8") as fh:
        return fh.read()


def test_success_ships_the_worker_log(tmp_path):
    """A job that succeeds and prints still ships its log."""
    jt = _jt(
        ["python", "-c", "import sys; print('HELLO-STDOUT'); sys.stderr.write('HELLO-STDERR\\n')"]
    )
    m = JobManifest.from_dict({"job_id": "ok1", "job_type": jt.id})
    results, members, dest = _run(tmp_path, jt, m)
    assert [r.status for r in results] == ["done"]
    text = _log_text(members, dest, "ok1")
    assert "HELLO-STDOUT" in text
    assert "HELLO-STDERR" in text
    assert "job_id=ok1" in text


def test_failure_ships_the_cause_the_message_tail_drops(tmp_path):
    """The cause is at the head of the traceback and the symptom at the tail, so the
    500-char message tail carries only the symptom. The bundled log has to carry both, or
    the client is diagnosing from the wrong exception."""
    jt = _jt(
        [
            "python",
            "-c",
            "import sys;"
            " sys.stderr.write('OSError: [Errno 28] No space left on device\\n');"
            " sys.stderr.write('x' * 900 + '\\n');"
            " sys.stderr.write('OSError: Unable to load vocabulary from file\\n');"
            " sys.exit(1)",
        ]
    )
    m = JobManifest.from_dict({"job_id": "bad1", "job_type": jt.id})
    results, members, dest = _run(tmp_path, jt, m)
    assert [r.status for r in results] == ["failed"]
    msg = results[0].message
    # the manifest message is the tail only; it names the symptom, not the cause
    assert "Unable to load vocabulary" in msg
    assert "No space left on device" not in msg
    # the bundled log is what closes the gap
    text = _log_text(members, dest, "bad1")
    assert "No space left on device" in text  # the cause the tail dropped
    assert "Unable to load vocabulary" in text  # and the symptom


def test_log_is_not_listed_as_an_output(tmp_path):
    """It rides in the bundle but is not a declared output: the manifest's outputs list is
    the job type's contract and the broker's false-success check, so a broker-authored file
    must not appear in it."""
    jt = _jt(
        [
            "python",
            "-c",
            "import os;"
            " open(os.path.join(os.environ['GPUWORKER_OUT'], 'made.txt'), 'w')"
            ".write('real output');"
            " print('WORKER-CHATTER')",
        ]
    )
    m = JobManifest.from_dict({"job_id": "made1", "job_type": jt.id, "outputs": ["made.txt"]})
    results, members, dest = _run(tmp_path, jt, m)
    assert [r.status for r in results] == ["done"]
    assert (results[0].result or {}).get("outputs") == ["made.txt"]
    # both travel, but only one is claimed as an output
    assert "made.txt" in members
    assert "WORKER-CHATTER" in _log_text(members, dest, "made1")


def test_false_success_check_still_bites(tmp_path):
    """A log that always exists must not let "exited 0 but did not produce its declared outputs"
    pass. The job still fails, and the log still ships, because a silent false success is
    exactly when you need to read the worker."""
    jt = _jt(["python", "-c", "print('ran but wrote nothing')"])
    m = JobManifest.from_dict(
        {"job_id": "silent1", "job_type": jt.id, "outputs": ["out/missing.txt"]}
    )
    results, members, dest = _run(tmp_path, jt, m)
    assert [r.status for r in results] == ["failed"]
    assert "did not produce" in results[0].message
    assert "ran but wrote nothing" in _log_text(members, dest, "silent1")


def _pj(tmp_path, out_dir, job_id="j1"):
    jt = _jt(["python", "-c", "pass"])
    sched = _sched(tmp_path, jt)
    b = Batch(queue=None, filename="b.zip", work_dir=str(tmp_path / "wk"), out_dir=out_dir)
    pj = PendingJob(JobManifest.from_dict({"job_id": job_id, "job_type": jt.id}), jt, b)
    b.jobs.append(pj)
    return sched, pj


def test_streams_are_clipped_at_both_ends(tmp_path):
    sched, pj = _pj(tmp_path, str(tmp_path / "out"))
    huge = ("CAUSE-AT-HEAD" + "x" * (WORKER_LOG_CAP * 2) + "SYMPTOM-AT-TAIL").encode()
    rel = sched._write_worker_log(pj, {"stdout": huge, "stderr": b""}, "")
    assert rel == f"{WORKER_LOG_PREFIX}j1.log"
    with open(os.path.join(str(tmp_path / "out"), *rel.split("/")), encoding="utf-8") as fh:
        text = fh.read()
    assert "CAUSE-AT-HEAD" in text
    assert "SYMPTOM-AT-TAIL" in text
    assert "elided" in text
    assert len(text) < WORKER_LOG_CAP * 2


def test_write_failure_degrades_and_never_raises(tmp_path):
    """A log we cannot write must not fail a job that otherwise succeeded."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    sched, pj = _pj(tmp_path, str(blocker / "out"))
    assert sched._write_worker_log(pj, {"stdout": b"hi", "stderr": b""}, "") is None


def test_missing_streams_say_so(tmp_path):
    """An empty log must not read as "the worker printed nothing" when the truth is that
    we never got to read its pipes (a launch that raised, or a kill whose salvage failed)."""
    sched, pj = _pj(tmp_path, str(tmp_path / "out"))
    rel = sched._write_worker_log(pj, {}, "job exceeded timeout 900s")
    with open(os.path.join(str(tmp_path / "out"), *rel.split("/")), encoding="utf-8") as fh:
        text = fh.read()
    assert "never captured" in text
    assert "job exceeded timeout 900s" in text


def _SC(root):
    return ServeContainer(jt=None, epoch=1, name="n", root=root, jobq=None, stop=None)


def test_serve_log_slice_is_this_jobs_bytes_only(tmp_path):
    """A warm container's serve.log is shared by every job it serves; the slice has to be
    the delta this job appended, not the whole file."""
    sched, _ = _pj(tmp_path, str(tmp_path / "out"))
    sc = _SC(str(tmp_path))
    path = os.path.join(sc.root, "serve.log")
    with open(path, "wb") as fh:
        fh.write(b"PREVIOUS-JOB-OUTPUT\n")
    start = sc.log_size()
    with open(path, "ab") as fh:
        fh.write(b"THIS-JOB-OUTPUT\n")
    got = sc.log_slice(start)
    assert got == b"THIS-JOB-OUTPUT\n"
    assert b"PREVIOUS" not in got


def test_serve_log_slice_is_empty_when_the_file_shrank(tmp_path):
    """A respawn/rotation resets the log; a stale offset must yield nothing rather than a
    wrong slice attributed to this job."""
    sched, _ = _pj(tmp_path, str(tmp_path / "out"))
    sc = _SC(str(tmp_path))
    with open(os.path.join(sc.root, "serve.log"), "wb") as fh:
        fh.write(b"short\n")
    assert sc.log_slice(10_000) == b""
    assert _SC(str(tmp_path / "nope")).log_size() == 0


# ---- serve mode, end to end -------------------------------------------------
# A warm serve container is where real model workloads run, and it is the harder case:
# every job it serves appends to one shared serve.log, so each bundle must carry that job's
# slice and nobody else's. Containerless fake: a socketpair per run(), an in-process worker
# that writes to spec.log_path exactly as the real container's inherited stdout does.


class _P:
    def __init__(self, thread):
        self.thread, self.returncode = thread, 0

    def wait(self, timeout=None):
        self.thread.join(timeout)

    def poll(self):
        return None if self.thread.is_alive() else 0


class _LoggingServe:
    def __init__(self):
        self.runs = 0
        self._ends: dict[str, socket.socket] = {}

    def run(self, spec):
        self.runs += 1
        broker, worker = socket.socketpair()
        log_path, out_root = spec.log_path, spec.out_dir

        def say(line):
            with open(log_path, "a", encoding="utf-8") as lg:
                lg.write(line + "\n")

        def go():
            try:
                # A model prints while it loads, before it is ready. Job 1's slice started
                # at the log size when the job started, so this was in no job's log.
                say("CONTAINER-READY")
                serve.send_frame(worker, serve.ready_msg())
                while True:
                    try:
                        msg = serve.recv_frame(worker)
                    except serve.ProtocolError:
                        return
                    if msg.get("type") == "unload":
                        return
                    if msg.get("type") != "run":
                        continue
                    jid, man = msg["job_id"], msg["manifest"]
                    say(f"CHATTER-FROM-{man['job_id']}")
                    written = []
                    for rel in man.get("outputs") or []:
                        p = os.path.join(out_root, jid, rel)
                        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
                        with open(p, "w", encoding="utf-8") as fh:
                            fh.write("payload\n")
                        written.append(rel)
                    serve.send_frame(worker, serve.result_msg(jid, "done", written, ""))
            except OSError:
                pass  # broker dropped our end, as expected on teardown
            finally:
                try:
                    worker.close()
                except OSError:
                    pass

        t = threading.Thread(target=go, daemon=True)
        t.start()
        self._ends[spec.serve_socket_host] = broker
        return _P(t)

    def terminate(self, proc, spec):
        pass

    def terminate_by_name(self, name):
        pass

    def reap_serve_orphans(self):
        pass

    def cleanup(self, spec):
        pass


def _serve_sched(tmp_path, launcher, jt):
    reg = Registry()
    reg.job_types[jt.id] = jt
    cfg = make_config(
        tmp_path,
        queue_roots=[str(tmp_path / "Q")],
        registry_dir="modules",
        container_runtime="docker",
        weights_dir=str(tmp_path / "w"),
        sandbox_image="gpuworker/sandbox:cuda12.8",
        allow_unpinned_images=True,
    )
    cfg.serve_connect_timeout_s = 5.0
    cfg.serve_ready_timeout_s = 5.0
    return Scheduler(cfg, reg, launcher, gpu.MockGpu())


def test_serve_job_ships_its_own_slice_of_the_shared_log(tmp_path, monkeypatch):
    launcher = _LoggingServe()

    def connect(path, timeout=60.0, poll_s=0.1):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path in launcher._ends:
                return launcher._ends.pop(path)
            time.sleep(0.005)
        raise serve.ProtocolError(f"no broker end for {path}")

    monkeypatch.setattr(serve, "connect", connect)

    jt = JobType(
        id="ex.serve",
        resource_class="vram",
        vram_mb=5000,
        env="container",
        entrypoint=["python", "-c", "pass"],
        modes=("serve",),
        serve_socket="/run/gpuworker/job.sock",
        sandbox="standard",
        keep_warm_s=100.0,
        serve_job_timeout=10.0,
    )
    sched = _serve_sched(tmp_path, launcher, jt)
    queue_root = str(tmp_path / "Q")

    texts = {}
    for jid in ("s1", "s2"):
        m = JobManifest.from_dict({"job_id": jid, "job_type": jt.id, "outputs": ["out/o.txt"]})
        fn = submit(queue_root, [m])
        out_zip = os.path.join(queue_root, "outbox", fn)
        for _ in range(400):
            sched.tick()
            if os.path.exists(out_zip):
                sched.tick()
                break
            time.sleep(0.005)
        assert os.path.exists(out_zip), f"{jid} never shipped"
        assert wait_result(queue_root, fn, timeout_s=5)[0].status == "done"
        dest = str(tmp_path / f"collected-{jid}")
        texts[jid] = _log_text(bundle.extract_outputs(out_zip, dest), dest, jid)

    assert launcher.runs == 1  # one warm container served both
    assert "CONTAINER-READY" in texts["s1"]  # cold-load output belongs to job 1
    assert "CHATTER-FROM-s1" in texts["s1"]
    # the property that matters: job 2 gets its own slice, not the whole shared log
    assert "CHATTER-FROM-s2" in texts["s2"]
    assert "CHATTER-FROM-s1" not in texts["s2"]
    assert "CONTAINER-READY" not in texts["s2"]


def test_clip_keeps_both_ends():
    """Tail-only clipping hides the root exception of a chained traceback."""
    text = "ORIGIN-ENOSPC" + ("x" * 5000) + "FINAL-vocab-error"
    out = clip_both_ends(text, 200)
    assert "ORIGIN-ENOSPC" in out  # the cause, which tail-only clipping dropped
    assert "FINAL-vocab-error" in out  # the symptom
    assert "elided" in out
    assert len(out) < len(text)


def test_clip_leaves_short_text_alone():
    assert clip_both_ends("short", 200) == "short"
