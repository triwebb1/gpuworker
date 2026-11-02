"""The model modules (speech, audiogen) without a GPU: their manifests load, and their
workers' logic runs against fake models. The model calls themselves need the GPU host."""

import json
import os
import socket
import sys
import threading
import wave

import pytest

from gpuworker.module import scan_registry, source_fingerprint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _mod in ("speech", "audiogen"):
    _src = os.path.join(ROOT, "modules", _mod, "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)

from audiogen import cuda_errors as audiogen_cuda  # noqa: E402
from audiogen import generate  # noqa: E402
from speech import check, jobio, serve, text, tts  # noqa: E402

np = pytest.importorskip("numpy")


def test_manifests_load():
    reg = scan_registry(os.path.join(ROOT, "modules"))
    tts_type = reg.get("speech.voxcpm2")
    assert tts_type.modes == ("serve",) and tts_type.sandbox == "strict"
    assert tts_type.weights_required  # strict implies it
    assert reg.get("speech.check").weights_required
    assert reg.get("audiogen.stableaudio").modes == ("oneshot",)


@pytest.mark.parametrize("mod,fingerprint", [("speech", jobio), ("audiogen", generate)])
def test_runtime_fingerprint_matches_the_brokers(mod, fingerprint):
    digest, count, _ = source_fingerprint(os.path.join(ROOT, "modules", mod))
    assert fingerprint.src_fingerprint() == (
        f"GPUWORKER_RUNTIME_MODULE src_sha256={digest} files={count}"
    )


@pytest.mark.parametrize("mod", ["_TEMPLATE", "speech", "audiogen"])
def test_provision_puts_src_on_the_path_instead_of_installing_it(mod):
    """The template pip-installed src/, so a module copied from it imported its code from
    site-packages and its runtime fingerprint hashed files the job never ran."""
    with open(os.path.join(ROOT, "modules", mod, "provision.sh"), encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if not ln.lstrip().startswith("#")]
    installs = [ln for ln in lines if "pip" in ln and " install " in ln]
    assert installs and all(
        ("-r " in ln or "--index-url" in ln or "--upgrade" in ln) for ln in installs
    ), installs
    assert any(ln.endswith('.pth"') and "src" in ln.lower() for ln in lines)


# --- text ----------------------------------------------------------------------------


def test_split_chunks_packs_whole_sentences():
    src = "One two. Three four five. Six! Seven eight nine ten."
    chunks = text.split_chunks(src, 22)
    assert chunks == ["One two.", "Three four five. Six!", "Seven eight nine ten."]
    assert text.split_chunks(src, 1000) == [src]


def test_split_chunks_breaks_long_sentences_and_keeps_every_word():
    src = "alpha beta, gamma delta, " + " ".join(f"w{i}" for i in range(30)) + "."
    chunks = text.split_chunks(src, 25)
    assert all(len(c) <= 25 for c in chunks)
    assert " ".join(chunks).split() == src.split()


def test_score_exact_case_and_punctuation_insensitive():
    r = text.score("Hello, world. It's here!", "hello world it’s here")
    assert r["wer"] == 0.0 and r["missing_tail_words"] == 0


def test_score_reports_a_missing_tail():
    expected = "the job ran to the very end of the text"
    r = text.score(expected, "the job ran to the very")
    assert r["missing_tail_words"] == 4
    assert r["word_errors"] == 4


def test_score_counts_substitutions_and_insertions():
    r = text.score("a b c d", "a x c d e")
    assert r["word_errors"] == 2 and r["missing_tail_words"] == 0
    assert r["wer"] == 0.5


# --- job I/O -------------------------------------------------------------------------


@pytest.mark.parametrize("rel", ["../other/in.wav", "/etc/passwd", "a/../../b", "a\\b", "", "./a"])
def test_job_path_refuses_escapes(rel):
    with pytest.raises(jobio.BadJob):
        jobio.job_path("/job/bundle/j1", rel)


def test_job_path_joins_nested():
    assert jobio.job_path("root", "in/ref.wav") == os.path.join("root", "in", "ref.wav")


def test_opt_num_bounds_types_and_strings():
    assert jobio.opt_num({"n": "7"}, "n", 1, 0, 10, int) == 7
    assert jobio.opt_num({}, "x", 2.0, 0, 5, float) == 2.0
    assert jobio.opt_num({"x": 3}, "x", 2.0, 0, 5, float) == 3.0
    for bad in ({"n": True}, {"n": 11}, {"n": "seven"}, {"n": 1.5}):
        with pytest.raises(jobio.BadJob):
            jobio.opt_num(bad, "n", 1, 0, 10, int)


def _read_wav(path):
    with wave.open(path, "rb") as fh:
        return fh.getnchannels(), fh.getframerate(), fh.getnframes()


def test_write_wav_refuses_non_finite(tmp_path):
    with pytest.raises(RuntimeError):
        jobio.write_wav(str(tmp_path / "x.wav"), np.array([0.0, np.nan]), 100)


# --- speech.voxcpm2 ------------------------------------------------------------------


class FakeTts:
    """Returns 0.1 s of audio per call and records the arguments."""

    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return np.full(100, 0.25, dtype=np.float32)


def _tts_job(tmp_path, options, outputs=("out/a.wav", "out/a.json")):
    bundle, out = tmp_path / "bundle", tmp_path / "out"
    (bundle / "j1" / "in").mkdir(parents=True)
    (bundle / "j1" / "in" / "ref.wav").write_bytes(b"RIFF")
    return {"options": options, "outputs": list(outputs)}, str(bundle), str(out)


def test_tts_renders_chunks_with_pauses(tmp_path):
    manifest, bundle, out = _tts_job(
        tmp_path,
        {
            "text": "First sentence here. Second one.",
            "max_chunk_chars": 20,
            "pause_ms": 50,
            "reference": "in/ref.wav",
            "reference_text": "what the clip says",
            "seed": 3,
        },
    )
    model, seeds = FakeTts(), []
    status, outs, _msg = tts.render("j1", manifest, model, 1000, bundle, out, seeds.append)
    assert status == "done" and outs == ["out/a.wav", "out/a.json"]
    assert seeds == [3]
    assert [c["text"] for c in model.calls] == ["First sentence here.", "Second one."]
    ref = os.path.join(bundle, "j1", "in", "ref.wav")
    assert model.calls[0]["reference_wav_path"] == ref
    assert model.calls[0]["prompt_wav_path"] == ref
    assert model.calls[0]["prompt_text"] == "what the clip says"
    # two 100-sample chunks and one 50 ms pause at 1000 Hz
    assert _read_wav(os.path.join(out, "j1", "out", "a.wav")) == (1, 1000, 250)
    sidecar = json.loads(open(os.path.join(out, "j1", "out", "a.json")).read())
    assert len(sidecar["chunks"]) == 2 and sidecar["settings"]["cfg_value"] == 2.0
    assert "reference_wav_path" not in sidecar["settings"]


def test_tts_reads_text_file_from_the_jobs_own_inputs(tmp_path):
    manifest, bundle, out = _tts_job(tmp_path, {"text_file": "in/script.txt"}, ["a.wav"])
    open(os.path.join(bundle, "j1", "in", "script.txt"), "w").write("Read from a file.")
    model = FakeTts()
    tts.render("j1", manifest, model, 1000, bundle, out)
    assert model.calls[0]["text"] == "Read from a file."
    assert "reference_wav_path" not in model.calls[0]


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"text": "a", "text_file": "b"},
        {"text": "   "},
        {"text": "hi", "reference": "../j2/in/ref.wav"},
        {"text": "hi", "reference": "in/missing.wav"},
        {"text": "hi", "reference_text": "no clip"},
        {"text": "hi", "cfg_value": 0},
    ],
)
def test_tts_bad_jobs(tmp_path, options):
    manifest, bundle, out = _tts_job(tmp_path, options)
    with pytest.raises(jobio.BadJob):
        tts.render("j1", manifest, FakeTts(), 1000, bundle, out)


def test_tts_needs_a_wav_output(tmp_path):
    manifest, bundle, out = _tts_job(tmp_path, {"text": "hi"}, ["a.json"])
    with pytest.raises(jobio.BadJob):
        tts.render("j1", manifest, FakeTts(), 1000, bundle, out)


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX") or os.name == "nt", reason="unix sockets")
def test_serve_loop_survives_a_bad_job_and_unloads(tmp_path, monkeypatch):
    path = str(tmp_path / "s" / "job.sock")
    monkeypatch.setenv("GPUWORKER_SERVE_SOCKET", path)
    released = []

    def handle(job_id, manifest):
        if job_id == "bad":
            raise jobio.BadJob("nope")
        return "done", manifest["outputs"], "ok"

    code = []
    t = threading.Thread(
        target=lambda: code.append(serve.run(lambda: handle, lambda: released.append(1)))
    )
    t.start()
    for _ in range(200):
        if os.path.exists(path):
            break
        threading.Event().wait(0.01)
    c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    c.connect(path)
    assert serve.recv(c) == {"type": "ready", "protocol": 1}
    serve.send(c, {"type": "run", "job_id": "bad", "manifest": {}})
    assert serve.recv(c)["status"] == "failed"
    serve.send(c, {"type": "run", "job_id": "good", "manifest": {"outputs": ["a.wav"]}})
    assert serve.recv(c) == {
        "type": "result",
        "job_id": "good",
        "status": "done",
        "outputs": ["a.wav"],
        "message": "ok",
    }
    serve.send(c, {"type": "unload"})
    t.join(5)
    assert code == [0] and released == [1]


class OutOfMemoryError(RuntimeError):
    """Stands in for torch.cuda.OutOfMemoryError, matched by name."""


def _wrapped(inner, outer=None, implicit=False):
    """*outer* raised from *inner* (``raise ... from``), or while handling it."""
    outer = outer or RuntimeError("generation failed")
    if implicit:
        outer.__context__ = inner
    else:
        outer.__cause__ = inner
    return outer


@pytest.mark.parametrize(
    "exc,kind",
    [
        (OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"), "cuda_oom"),
        (RuntimeError("CUDA failed with error out of memory"), "cuda_oom"),  # CTranslate2
        (RuntimeError("CUDA error: an illegal memory access was encountered"), "cuda_error"),
        (RuntimeError("CUBLAS_STATUS_EXECUTION_FAILED when calling cublasGemmEx"), "cuda_error"),
        (RuntimeError("CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate"), "cuda_oom"),
        # Only the top-level exception was read, so a wrapped CUDA failure counted as the job's.
        (_wrapped(OutOfMemoryError("CUDA out of memory")), "cuda_oom"),
        (
            _wrapped(RuntimeError("CUDA error: unspecified launch failure"), implicit=True),
            "cuda_error",
        ),
        (
            _wrapped(OutOfMemoryError("CUDA out of memory"), RuntimeError("CUDA error: x")),
            "cuda_error",
        ),
        (RuntimeError("shape mismatch"), ""),
        (MemoryError("host RAM"), ""),
        (jobio.BadJob("no text"), ""),
    ],
)
def test_cuda_error_kind(exc, kind):
    # Both modules, through the names their workers import.
    assert jobio.cuda_error_kind(exc) == kind
    expected = {"cuda_oom": 75, "cuda_error": 76, "": 1}[kind]
    assert jobio.exit_code_for(exc) == expected
    assert generate.exit_code_for(exc) == expected


def test_both_modules_carry_the_same_cuda_error_classifier():
    """Each module runs from its own venv and src/, so each has a copy; they had drifted
    apart (audiogen did not know CTranslate2's message)."""
    copies = [
        open(os.path.join(ROOT, "modules", m, "src", m, "cuda_errors.py"), "rb").read()
        for m in ("speech", "audiogen")
    ]
    assert copies[0] == copies[1]


class _FakeCuda:
    def __init__(self):
        self.emptied = 0

    def empty_cache(self):
        self.emptied += 1


@pytest.fixture
def fake_torch(monkeypatch):
    import types

    torch = types.ModuleType("torch")
    torch.cuda = _FakeCuda()
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX") or os.name == "nt", reason="unix sockets")
def test_serve_loop_reports_cuda_errors_and_exits_after_a_poisoned_context(
    tmp_path, monkeypatch, fake_torch
):
    path = str(tmp_path / "s" / "job.sock")
    monkeypatch.setenv("GPUWORKER_SERVE_SOCKET", path)

    def handle(job_id, manifest):
        if job_id == "oom":
            raise OutOfMemoryError("CUDA out of memory")
        raise RuntimeError("CUDA error: device-side assert triggered")

    code = []
    t = threading.Thread(target=lambda: code.append(serve.run(lambda: handle)))
    t.start()
    for _ in range(200):
        if os.path.exists(path):
            break
        threading.Event().wait(0.01)
    c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    c.connect(path)
    serve.recv(c)  # ready
    serve.send(c, {"type": "run", "job_id": "oom", "manifest": {}})
    res = serve.recv(c)
    assert res["status"] == "failed" and res["error"] == "cuda_oom"
    assert fake_torch.cuda.emptied == 1  # the worker stays up, with the job's memory freed
    serve.send(c, {"type": "run", "job_id": "assert", "manifest": {}})
    assert serve.recv(c)["error"] == "cuda_error"
    t.join(5)  # exits on its own, without an unload
    assert code == [1]


# --- speech.check --------------------------------------------------------------------


def _check_job(tmp_path, monkeypatch, options, heard):
    bundle, out = tmp_path / "bundle", tmp_path / "out"
    (bundle / "jobs" / "pending").mkdir(parents=True)
    (bundle / "a.wav").write_bytes(b"RIFF")
    manifest = {"job_id": "c1", "options": options, "outputs": ["r/check.json"]}
    (bundle / "jobs" / "pending" / "c1.json").write_text(json.dumps(manifest))
    monkeypatch.setenv("GPUWORKER_JOB_ID", "c1")
    monkeypatch.setenv("GPUWORKER_BUNDLE", str(bundle))
    monkeypatch.setenv("GPUWORKER_OUT", str(out))
    monkeypatch.setattr(check, "transcribe", lambda *_a: (heard, 2.5))
    return out / "r" / "check.json"


def test_check_failed_check_still_exits_zero_with_a_record(tmp_path, monkeypatch):
    rec_path = _check_job(
        tmp_path, monkeypatch, {"audio": "a.wav", "text": "one two three four"}, "one two"
    )
    assert check.main() == 0
    rec = json.loads(rec_path.read_text())
    assert rec["passed"] is False and rec["missing_tail_words"] == 2
    assert rec["transcript"] == "one two" and rec["duration_s"] == 2.5


def test_check_pass(tmp_path, monkeypatch):
    rec_path = _check_job(tmp_path, monkeypatch, {"audio": "a.wav", "text": "One, two."}, "one two")
    assert check.main() == 0
    assert json.loads(rec_path.read_text())["passed"] is True


def test_check_bad_input_exits_non_zero(tmp_path, monkeypatch):
    _check_job(tmp_path, monkeypatch, {"audio": "missing.wav", "text": "x"}, "x")
    assert check.main() == 2


def test_check_cuda_oom_exits_with_the_retry_code(tmp_path, monkeypatch):
    _check_job(tmp_path, monkeypatch, {"audio": "a.wav", "text": "x"}, "x")

    def oom(*_a):
        raise RuntimeError("CUDA failed with error out of memory")

    monkeypatch.setattr(check, "transcribe", oom)
    assert check.main() == jobio.EXIT_CUDA_OOM == 75

    def cuda_error(*_a):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(check, "transcribe", cuda_error)
    assert check.main() == jobio.EXIT_CUDA_ERROR == 76


# --- audiogen.stableaudio ------------------------------------------------------------


class FakeTensor:
    def __init__(self, a):
        self.a = a

    @property
    def T(self):  # noqa: N802 (mirrors torch.Tensor.T)
        return FakeTensor(self.a.T)

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class FakePipe:
    def __init__(self, level=0.5):
        self.level = level
        self.calls = []
        self.vae = type("Vae", (), {"sampling_rate": 1000})()

    def __call__(self, prompt, **kw):
        self.calls.append((prompt, kw))
        frames = int(kw["audio_end_in_s"] * 1000)
        audios = [FakeTensor(np.full((2, frames), self.level, dtype=np.float32))]
        return type("Out", (), {"audios": audios})()


def _no_probe(_pipe):
    pass


def test_audiogen_renders_stereo_and_sidecar(tmp_path):
    pipe = FakePipe()
    manifest = {
        "options": {"prompt": "rain", "duration_s": "2", "steps": 20},
        "outputs": ["o/clip.wav", "meta/clip.json"],
    }
    info = generate.run(manifest, str(tmp_path), "/w/sa", load=lambda _d: pipe, check=_no_probe)
    assert _read_wav(str(tmp_path / "o" / "clip.wav")) == (2, 1000, 2000)
    assert info["channels"] == 2 and info["duration_s"] == 2.0
    side = json.loads((tmp_path / "meta" / "clip.json").read_text())
    assert side["settings"]["steps"] == 20 and side["settings"]["seed"] is None
    prompt, kw = pipe.calls[0]
    assert prompt == "rain" and kw["negative_prompt"] == generate.DEFAULT_NEGATIVE


def test_audiogen_probe_and_silence_checks(tmp_path):
    manifest = {"options": {"prompt": "x"}, "outputs": ["a.wav"]}
    generate.probe(FakePipe())
    with pytest.raises(RuntimeError, match="silent"):
        generate.run(manifest, str(tmp_path), "/w", load=lambda _d: FakePipe(0.0), check=_no_probe)


@pytest.mark.parametrize(
    "options,outputs",
    [
        ({}, ["a.wav"]),
        ({"prompt": "x", "duration_s": 48}, ["a.wav"]),
        ({"prompt": "x", "steps": 0}, ["a.wav"]),
        ({"prompt": "x"}, ["a.json"]),
    ],
)
def test_audiogen_bad_jobs(options, outputs):
    with pytest.raises(generate.BadJob):
        generate.parse_options({"options": options, "outputs": outputs})


def test_audiogen_cuda_oom_exits_with_the_retry_code(tmp_path, monkeypatch):
    (tmp_path / "jobs" / "pending").mkdir(parents=True)
    (tmp_path / "jobs" / "pending" / "g1.json").write_text("{}")
    monkeypatch.setenv("GPUWORKER_JOB_ID", "g1")
    monkeypatch.setenv("GPUWORKER_BUNDLE", str(tmp_path))
    monkeypatch.setenv("GPUWORKER_OUT", str(tmp_path / "out"))

    def oom(*_a, **_k):
        raise OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(generate, "run", oom)
    assert generate.main() == audiogen_cuda.EXIT_CUDA_OOM == 75
    monkeypatch.setattr(
        generate,
        "run",
        lambda *_a: (_ for _ in ()).throw(RuntimeError("CUDA error: launch failure")),
    )
    assert generate.main() == audiogen_cuda.EXIT_CUDA_ERROR == 76
    monkeypatch.setattr(generate, "run", lambda *_a: (_ for _ in ()).throw(ValueError("x")))
    assert generate.main() == 1


@pytest.mark.parametrize("bad", ["../escape.txt", "/abs/escape.txt", "out/../../escape.txt"])
def test_the_echo_workers_never_write_outside_their_output_dir(tmp_path, monkeypatch, bad):
    """The example people copy: the broker refuses such a manifest, but the worker also
    refuses on its own rather than trust it."""
    import importlib.util
    import subprocess

    bundle_dir = tmp_path / "b"
    (bundle_dir / "jobs" / "pending").mkdir(parents=True)
    out = tmp_path / "o" / "out"
    out.mkdir(parents=True)
    manifest = {"job_id": "a", "outputs": [bad], "options": {"text": "x"}}
    (bundle_dir / "jobs" / "pending" / "a.json").write_text(json.dumps(manifest))
    env = dict(
        os.environ, GPUWORKER_JOB_ID="a", GPUWORKER_BUNDLE=str(bundle_dir), GPUWORKER_OUT=str(out)
    )
    proc = subprocess.run(
        [sys.executable, "modules/echo/src/worker.py"], env=env, capture_output=True, text=True
    )
    assert proc.returncode == 1 and "refusing output" in proc.stderr

    spec = importlib.util.spec_from_file_location("echo_serve", "modules/echo/src/serve_worker.py")
    echo_serve = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(echo_serve)
    monkeypatch.setenv("GPUWORKER_OUT", str(out))
    with pytest.raises(ValueError, match="outside"):
        echo_serve.handle("k", manifest)
    assert not list(tmp_path.rglob("escape.txt"))


def test_speech_collects_garbage_before_emptying_the_cuda_cache(monkeypatch):
    """empty_cache alone left the tensors a failed job still held through reference cycles
    (its saved traceback) in use, so the memory never reached the cache to be freed."""
    import gc
    import types

    calls = []
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(empty_cache=lambda: calls.append("empty_cache"))
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(gc, "collect", lambda *a: calls.append("collect") or 0)
    serve._free_cached_memory()
    assert calls == ["collect", "empty_cache"]
