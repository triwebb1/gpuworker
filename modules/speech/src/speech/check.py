"""speech.check: the quality check. Transcribe a rendered WAV with Whisper and score it
against the text it should contain.

A oneshot job, kept apart from speech.voxcpm2 on purpose. A serve container keeps its
process, so a Whisper model loaded by the first render that ran a check would stay
resident for the container's life, and the broker would learn the render type's footprint
as text-to-speech plus Whisper and reserve that for every later render. As its own job
type, the check gets its own footprint and the scheduler places it independently.

Options:

  audio              bundle-relative WAV to check (required)
  text | text_file   the text the audio should contain (exactly one)
  language           Whisper language code, default "en"
  max_wer            pass threshold on the word error rate, default 0.1
  max_missing_tail   pass threshold on expected words missing at the end, default 0

Output: the first declared .json receives the record. A failed check is a result, not a
job failure: the worker exits 0 and the record says passed: false, because a non-zero exit
would make the broker drop the record. Bad input exits 2, CUDA running out of memory exits
75 and any other CUDA error 76, and any other error exits 1. The broker retries 76, and 75
when something else was using the GPU.
"""

from __future__ import annotations

import json
import os
import sys

from .jobio import (
    BadJob,
    cpu_budget,
    exit_code_for,
    job_path,
    opt_num,
    opt_str,
    pick_output,
    src_fingerprint,
)
from .text import score

DEFAULT_MODEL_DIR = "/job/weights/faster-whisper-large-v3"


def transcribe(model_dir: str, audio_path: str, language: str) -> tuple[str, float]:
    """Return the transcript and the audio duration in seconds."""
    from faster_whisper import WhisperModel

    if not os.path.isfile(os.path.join(model_dir, "model.bin")):
        raise RuntimeError(f"no faster-whisper model at {model_dir} (stage the weights first)")
    model = WhisperModel(model_dir, device="cuda", compute_type="float16", cpu_threads=cpu_budget())
    segments, info = model.transcribe(audio_path, language=language, beam_size=5)
    text = " ".join(seg.text.strip() for seg in segments)
    return text, float(info.duration)


def check(manifest: dict, bundle_dir: str, out_dir: str, model_dir: str) -> dict:
    options = manifest.get("options") or {}
    outputs = list(manifest.get("outputs") or [])
    audio_rel = opt_str(options, "audio")
    if audio_rel is None:
        raise BadJob("options.audio is required")
    audio_path = job_path(bundle_dir, audio_rel)
    if not os.path.isfile(audio_path):
        raise BadJob(f"audio {audio_rel!r} is not in the bundle")
    text = opt_str(options, "text")
    text_file = opt_str(options, "text_file")
    if (text is None) == (text_file is None):
        raise BadJob("set exactly one of options.text and options.text_file")
    if text_file is not None:
        with open(job_path(bundle_dir, text_file), encoding="utf-8") as fh:
            text = fh.read()
    assert text is not None
    language = opt_str(options, "language", "en")
    max_wer = opt_num(options, "max_wer", 0.1, 0.0, 1.0, float)
    max_tail = opt_num(options, "max_missing_tail", 0, 0, 100000, int)
    json_rel = pick_output(outputs, ".json")
    assert json_rel is not None and language is not None

    heard, duration = transcribe(model_dir, audio_path, language)
    result = score(text, heard)
    record = {
        "audio": audio_rel,
        "model": os.path.basename(model_dir.rstrip("/")),
        "language": language,
        "duration_s": round(duration, 3),
        **result,
        "max_wer": max_wer,
        "max_missing_tail": max_tail,
        "passed": result["wer"] <= max_wer and result["missing_tail_words"] <= max_tail,
        "transcript": heard,
    }
    dest = os.path.join(out_dir, json_rel)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    return record


def main() -> int:
    job_id = os.environ.get("GPUWORKER_JOB_ID", "")
    bundle_dir = os.environ.get("GPUWORKER_BUNDLE", "/job/bundle")
    out_dir = os.environ.get("GPUWORKER_OUT", "/job/out")
    model_dir = os.environ.get("SPEECH_WHISPER_DIR", DEFAULT_MODEL_DIR)
    try:
        with open(
            os.path.join(bundle_dir, "jobs", "pending", f"{job_id}.json"), encoding="utf-8"
        ) as fh:
            manifest = json.load(fh)
        print(src_fingerprint(), flush=True)
        record = check(manifest, bundle_dir, out_dir, model_dir)
    except BadJob as exc:
        print(f"bad job: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return exit_code_for(exc)
    verdict = "passed" if record["passed"] else "did not pass"
    print(f"{job_id}: wer={record['wer']} missing_tail={record['missing_tail_words']} {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
