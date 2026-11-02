"""speech.voxcpm2: text-to-speech with VoxCPM2, as a warm serve worker.

The model loads once per container and each `run` frame renders one job: the text is split
into chunks of whole sentences, each chunk is generated, and the chunks are joined with a
short pause into one WAV. A reference recording in the bundle (the persona) sets the voice,
so jobs that pass the same recording share one voice; with its transcript as well,
VoxCPM2 also continues from the recording, which holds the voice closer to it.

Options (all optional except the text):

  text | text_file     the text, inline or as a bundle-relative UTF-8 file (exactly one)
  reference           bundle-relative WAV of the persona's voice
  reference_text      transcript of that recording
  cfg_value           guidance scale, default 2.0
  inference_timesteps diffusion steps per chunk, default 10
  seed                seeds torch and numpy before the first chunk
  max_chunk_chars     longest chunk, default 300
  pause_ms            silence between chunks, default 250

Outputs: the first declared .wav receives the audio; a declared .json receives a sidecar
with the settings and per-chunk durations.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from . import serve
from .jobio import (
    BadJob,
    cpu_budget,
    job_path,
    opt_num,
    opt_str,
    pick_output,
    src_fingerprint,
    write_wav,
)
from .text import split_chunks

DEFAULT_MODEL_DIR = "/job/weights/VoxCPM2"
PROBE_TEXT = "This is a short test sentence."


def load_model(model_dir: str) -> Any:
    """Load VoxCPM2 from a staged directory, with the denoiser off.

    The default constructor also loads a separate denoiser model that is not part of the
    staged weights and cannot be downloaded with no network. The constructor compiles the
    model with torch.compile and runs one warm-up generation before it returns.
    """
    from voxcpm import VoxCPM

    if not os.path.isfile(os.path.join(model_dir, "config.json")):
        raise RuntimeError(f"no VoxCPM2 checkpoint at {model_dir} (stage the weights first)")
    return VoxCPM(voxcpm_model_path=model_dir, zipenhancer_model_path=None, enable_denoiser=False)


def seed_all(seed: int) -> None:
    """Seed the global generators. VoxCPM2's generate() takes no seed of its own."""
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def render(
    job_id: str,
    manifest: dict,
    model: Any,
    sample_rate: int,
    bundle_root: str,
    out_root: str,
    seed_fn=seed_all,
) -> tuple[str, list[str], str]:
    """Render one job and return its (status, outputs, message) for the result frame.
    *job_id* is the frame's: it names the job's directories, not the client's job."""
    import numpy as np

    options = manifest.get("options") or {}
    outputs = list(manifest.get("outputs") or [])
    in_dir = os.path.join(bundle_root, job_id)
    out_dir = os.path.join(out_root, job_id)

    text = opt_str(options, "text")
    text_file = opt_str(options, "text_file")
    if (text is None) == (text_file is None):
        raise BadJob("set exactly one of options.text and options.text_file")
    if text_file is not None:
        with open(job_path(in_dir, text_file), encoding="utf-8") as fh:
            text = fh.read()
    assert text is not None
    if not text.strip():
        raise BadJob("the text is empty")

    kwargs: dict[str, Any] = {
        "cfg_value": opt_num(options, "cfg_value", 2.0, 0.1, 10.0, float),
        "inference_timesteps": opt_num(options, "inference_timesteps", 10, 1, 100, int),
    }
    reference = opt_str(options, "reference")
    reference_text = opt_str(options, "reference_text")
    if reference is not None:
        ref_path = job_path(in_dir, reference)
        if not os.path.isfile(ref_path):
            raise BadJob(f"reference {reference!r} is not in the bundle")
        kwargs["reference_wav_path"] = ref_path
        if reference_text:
            kwargs["prompt_wav_path"] = ref_path
            kwargs["prompt_text"] = reference_text
    elif reference_text is not None:
        raise BadJob("options.reference_text needs options.reference")

    seed = opt_num(options, "seed", None, 0, 2**32 - 1, int)
    max_chars = opt_num(options, "max_chunk_chars", 300, 20, 2000, int)
    pause_ms = opt_num(options, "pause_ms", 250, 0, 5000, int)
    wav_rel = pick_output(outputs, ".wav")
    json_rel = pick_output(outputs, ".json", required=False)
    assert wav_rel is not None

    if seed is not None:
        seed_fn(seed)
    gap = np.zeros(int(sample_rate * pause_ms / 1000), dtype=np.float32)
    pieces: list[Any] = []
    chunk_log = []
    for i, chunk in enumerate(split_chunks(text, max_chars)):
        audio = np.asarray(model.generate(text=chunk, **kwargs), dtype=np.float32).reshape(-1)
        if audio.size == 0:
            raise RuntimeError(f"chunk {i} produced no audio")
        if pieces:
            pieces.append(gap)
        pieces.append(audio)
        chunk_log.append({"text": chunk, "seconds": round(audio.size / sample_rate, 3)})

    duration = write_wav(os.path.join(out_dir, wav_rel), np.concatenate(pieces), sample_rate)
    written = [wav_rel]
    if json_rel is not None:
        sidecar = {
            "job_id": manifest.get("job_id", job_id),
            "model": "VoxCPM2",
            "sample_rate": sample_rate,
            "duration_s": round(duration, 3),
            "settings": {
                **{k: v for k, v in kwargs.items() if not k.endswith("_path")},
                "reference": reference,
                "seed": seed,
                "max_chunk_chars": max_chars,
                "pause_ms": pause_ms,
            },
            "chunks": chunk_log,
        }
        dest = os.path.join(out_dir, json_rel)
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(sidecar, fh, indent=2)
        written.append(json_rel)
    return "done", written, f"{len(chunk_log)} chunks, {duration:.1f} s"


def main() -> int:
    model_dir = os.environ.get("VOXCPM_MODEL_DIR", DEFAULT_MODEL_DIR)
    bundle_root = os.environ.get("GPUWORKER_BUNDLE", "/job/bundle")
    out_root = os.environ.get("GPUWORKER_OUT", "/job/out")
    state: dict[str, Any] = {}
    # A serve job's log is the slice written while it ran, so the marker is printed per job.
    marker = src_fingerprint()

    def load() -> serve.Handler:
        import numpy as np
        import torch

        torch.set_num_threads(cpu_budget())
        model = load_model(model_dir)
        sample_rate = int(model.tts_model.sample_rate)
        # Check the whole generate path before reporting ready, so a broken load fails
        # here with a named error instead of failing every job.
        probe = np.asarray(model.generate(text=PROBE_TEXT), dtype=np.float32)
        if probe.size == 0 or not np.all(np.isfinite(probe)):
            raise RuntimeError("VoxCPM2 loaded but its probe generation returned no audio")
        print(f"VoxCPM2 ready at {sample_rate} Hz", file=sys.stderr, flush=True)
        state["model"] = model

        def handle(job_id: str, manifest: dict) -> tuple[str, list[str], str]:
            print(marker, file=sys.stderr, flush=True)
            return render(job_id, manifest, state["model"], sample_rate, bundle_root, out_root)

        return handle

    def release() -> None:
        state.clear()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # releasing is best effort; the container is stopped next
            pass

    return serve.run(load, release)


if __name__ == "__main__":
    sys.exit(main())
