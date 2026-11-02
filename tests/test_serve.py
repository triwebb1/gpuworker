import json
import os
import socket

import pytest

from gpuworker import PROTOCOL_VERSION, serve


def test_run_job_timeout_raises_protocol_error():
    # A worker that never replies must time out the broker (so a wedged model can't hang
    # the serve manager forever); the broker then tears the container down.
    a, b = socket.socketpair()
    try:
        sess = serve.ServeSession(a)
        with pytest.raises(serve.ProtocolError, match="timed out"):
            sess.run_job("j1", {}, timeout=0.1)  # b never reads/replies
    finally:
        a.close()
        b.close()


def test_wait_ready_timeout_raises_protocol_error():
    a, b = socket.socketpair()
    try:
        sess = serve.ServeSession(a)
        with pytest.raises(serve.ProtocolError):
            sess.wait_ready(timeout=0.1)  # b never sends ready
    finally:
        a.close()
        b.close()


def test_frame_roundtrip_over_socketpair():
    a, b = socket.socketpair()
    try:
        serve.send_frame(a, serve.run_msg("j1", {"k": "v"}))
        got = serve.recv_frame(b)
        assert got["type"] == "run" and got["job_id"] == "j1"
        assert got["manifest"] == {"k": "v"}
    finally:
        a.close()
        b.close()


def test_multiple_frames_in_order():
    a, b = socket.socketpair()
    try:
        serve.send_frame(a, serve.ready_msg())
        serve.send_frame(a, serve.result_msg("j1", "done", ["out/x"], "ok"))
        m1 = serve.recv_frame(b)
        m2 = serve.recv_frame(b)
        assert m1["type"] == "ready"
        assert m2["type"] == "result" and m2["status"] == "done"
    finally:
        a.close()
        b.close()


def test_check_ready_version_match():
    serve.check_ready(serve.ready_msg())  # no raise


def test_check_ready_version_mismatch():
    with pytest.raises(serve.ProtocolError, match="protocol mismatch"):
        serve.check_ready({"type": "ready", "protocol": PROTOCOL_VERSION + 99})


def test_eof_raises():
    a, b = socket.socketpair()
    a.close()
    with pytest.raises(serve.ProtocolError):
        serve.recv_frame(b)
    b.close()


def test_oversized_length_rejected():
    a, b = socket.socketpair()
    try:
        import struct

        a.sendall(struct.pack(">I", serve.MAX_FRAME + 1))
        with pytest.raises(serve.ProtocolError, match="too large"):
            serve.recv_frame(b)
    finally:
        a.close()
        b.close()


@pytest.mark.parametrize("payload", [b"\xff\xfe not utf-8", b"{not json", b"[1, 2]"])
def test_garbled_frame_is_a_protocol_error(payload):
    # The broker's serve handlers catch ProtocolError; a raw JSONDecodeError or
    # UnicodeDecodeError would escape them and kill the manager thread.
    a, b = socket.socketpair()
    try:
        a.sendall(serve._HEADER.pack(len(payload)) + payload)
        with pytest.raises(serve.ProtocolError):
            serve.recv_frame(b)
    finally:
        a.close()
        b.close()


@pytest.mark.parametrize(
    "bad",
    [
        {"status": "ok"},
        {"status": None},
        {"outputs": "out/x"},
        {"outputs": [1, 2]},
        {"outputs": None},
        {"message": {"nested": True}},
        {"message": 5},
        {"error": "something_else"},
        {"error": "cuda_oom", "status": "done"},
    ],
)
def test_result_fields_are_validated(bad):
    # A worker's result reaches the loop and the client's result file: a wrong type is a
    # protocol violation, not something to pass along.
    msg = {**serve.result_msg("j1", "failed", ["out/x"], "m"), **bad}
    with pytest.raises(serve.ProtocolError):
        serve.parse_result(msg, "j1")


def test_result_parse_keeps_valid_fields_and_bounds_the_message():
    msg = serve.result_msg("j1", "failed", ["out/x"], "m" * 200_000, error=serve.ERROR_CUDA_OOM)
    res = serve.parse_result(msg, "j1")
    assert res.status == "failed" and res.outputs == ["out/x"]
    assert res.error == "cuda_oom" and len(res.message) == serve.MAX_MESSAGE_CHARS
    assert serve.parse_result(serve.result_msg("j1", "done"), "j1").error == ""
    with pytest.raises(serve.ProtocolError, match="mismatch"):
        serve.parse_result(serve.result_msg("j2", "done"), "j1")


# ---- connecting to the container's socket -------------------------------------------


needs_unix = pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="unix sockets")


def _listener(path):
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(1)
    return srv


@needs_unix
def test_connect_reaches_a_real_socket(tmp_path):
    path = str(tmp_path / "job.sock")
    srv = _listener(path)
    conn = serve.connect(path, timeout=2)
    peer, _ = srv.accept()
    conn.sendall(b"x")
    assert peer.recv(1) == b"x"
    for s in (conn, peer, srv):
        s.close()


@needs_unix
def test_connect_refuses_a_link_to_another_socket(tmp_path):
    """The container can write the socket's directory, so it could replace job.sock with
    a link to a host socket it wants the broker to talk to."""
    host = _listener(str(tmp_path / "host.sock"))
    sockdir = tmp_path / "run"
    sockdir.mkdir()
    os.symlink(tmp_path / "host.sock", sockdir / "job.sock")
    with pytest.raises(serve.ProtocolError, match="not a socket"):
        serve.connect(str(sockdir / "job.sock"), timeout=2)
    host.close()


def test_wait_ready_bounds_the_whole_wait_not_each_read():
    """A worker that sends its ready frame a byte at a time reset a per-read timeout on
    every byte and held the load phase as long as it liked."""
    import struct
    import threading
    import time

    a, b = socket.socketpair()
    payload = json.dumps(serve.ready_msg()).encode()
    frame = struct.pack(">I", len(payload)) + payload
    stop = threading.Event()

    def trickle():
        for byte in frame:
            if stop.wait(0.05):
                return
            try:
                b.sendall(bytes([byte]))
            except OSError:
                return

    t = threading.Thread(target=trickle, daemon=True)
    t.start()
    try:
        start = time.monotonic()
        with pytest.raises(serve.ProtocolError, match="ready in time"):
            serve.ServeSession(a).wait_ready(timeout=0.3)
        assert time.monotonic() - start < len(frame) * 0.05 / 2
    finally:
        stop.set()
        a.close()
        b.close()


def test_a_deeply_nested_frame_is_a_protocol_error():
    """json.loads raises RecursionError on deep nesting, which the serve manager does not
    catch: the job waited out its watchdog with nothing logged."""
    import struct

    a, b = socket.socketpair()
    try:
        bomb = b"[" * 200_000
        b.sendall(struct.pack(">I", len(bomb)) + bomb)
        with pytest.raises(serve.ProtocolError, match="RecursionError"):
            serve.recv_frame(a)
    finally:
        a.close()
        b.close()
