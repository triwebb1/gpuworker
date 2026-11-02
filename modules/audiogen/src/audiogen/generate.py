"""audiogen.stableaudio: music, ambience and sound effects from a text prompt, with Stable
Audio Open 1.0.

A oneshot worker: load the pipeline, check it can generate at all, render one clip, write
it. The check matters because a pipeline loaded from an incomplete offline cache can load
without error and then produce nothing, which would otherwise surface only as a job
timeout or a silent file.

Options:

  prompt           what to generate (required)
  negative_prompt  what to steer away from, default "low quality, distorted, noisy"
  duration_s       clip length in seconds, default 10, at most 47 (the model's limit)
  steps            diffusion steps, default 100
  guidance_scale   default 7.0
  seed             seeds the torch generator for the run

Outputs: the first declared .wav receives the clip (stereo, 16-bit, at the model's
sample rate); a declared .json receives a sidecar with the settings.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import wave
from typing import Any

from .cuda_errors import exit_code_for

DEFAULT_MODEL_DIR = "/job/weights/stable-audio-open-1.0"
MAX_DURATION_S = 47.0
DEFAULT_NEGATIVE = "low quality, distorted, noisy"
SILENCE_PEAK = 1e-4  # about -80 dBFS; a finished clip quieter than this is a broken render


SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def src_fingerprint(src: str = SRC_ROOT) -> str:
    """The marker line for MODULE_GUIDE 1.5: a digest of the source this process runs,
    computed the way the broker hashes the module's src/ (gpuworker.module)."""
    h, count = hashlib.sha256(), 0
    for root, dirs, files in os.walk(src):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for fn in sorted(files):
            if fn.endswith((".pyc", ".pyo")):
                continue
            full = os.path.join(root, fn)
            h.update(os.path.relpath(full, src).replace(os.sep, "/").encode("utf-8") + b"\0")
            try:
                with open(full, "rb") as fh:
                    h.update(fh.read())
            except OSError:
                h.update(b"<unreadable>")
            count += 1
    return f"GPUWORKER_RUNTIME_MODULE src_sha256={h.hexdigest()} files={count}"


class BadJob(ValueError):
    """The job's own input is wrong (exit 2, not retried)."""


def _num(options: dict, key: str, default: Any, lo: float, hi: float, kind: type) -> Any:
    value = options.get(key, default)
    if value is None:
        return None
    if isinstance(value, str):  # `gpuworker run-once --option k=v` passes strings
        try:
            value = kind(value)
        except ValueError:
            raise BadJob(f"options.{key} must be a {kind.__name__}") from None
    ok = isinstance(value, kind) or (kind is float and isinstance(value, int))
    if not ok or isinstance(value, bool) or not lo <= value <= hi:
        raise BadJob(f"options.{key} must be a {kind.__name__} in [{lo}, {hi}]")
    return kind(value)


def _text(options: dict, key: str, default: str | None = None) -> str | None:
    value = options.get(key, default)
    if value is not None and not isinstance(value, str):
        raise BadJob(f"options.{key} must be a string")
    return value


def parse_options(manifest: dict) -> dict:
    options = manifest.get("options") or {}
    prompt = _text(options, "prompt")
    if not prompt or not prompt.strip():
        raise BadJob("options.prompt is required")
    outputs = list(manifest.get("outputs") or [])
    wav = next((o for o in outputs if o.lower().endswith(".wav")), None)
    if wav is None:
        raise BadJob("declare an output ending in .wav")
    return {
        "prompt": prompt,
        "negative_prompt": _text(options, "negative_prompt", DEFAULT_NEGATIVE),
        "duration_s": _num(options, "duration_s", 10.0, 1.0, MAX_DURATION_S, float),
        "steps": _num(options, "steps", 100, 1, 500, int),
        "guidance_scale": _num(options, "guidance_scale", 7.0, 0.0, 20.0, float),
        "seed": _num(options, "seed", None, 0, 2**63 - 1, int),
        "wav": wav,
        "json": next((o for o in outputs if o.lower().endswith(".json")), None),
    }


def load_pipeline(model_dir: str) -> Any:
    import torch
    from diffusers import StableAudioPipeline

    if not os.path.isfile(os.path.join(model_dir, "model_index.json")):
        raise RuntimeError(f"no Stable Audio pipeline at {model_dir} (stage the weights first)")
    pipe = StableAudioPipeline.from_pretrained(model_dir, torch_dtype=torch.float16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def probe(pipe: Any) -> None:
    """One cheap generation. One step gives near silence, so check shape and finiteness."""
    import numpy as np

    out = pipe("probe", num_inference_steps=1, audio_end_in_s=1.0, num_waveforms_per_prompt=1)
    samples = out.audios[0].float().cpu().numpy()
    if samples.size == 0 or not np.all(np.isfinite(samples)):
        raise RuntimeError("Stable Audio loaded but its probe returned empty or non-finite audio")


def render(pipe: Any, opts: dict) -> tuple[Any, int]:
    """Return (frames x channels float32 array, sample rate)."""
    import numpy as np

    generator = None
    if opts["seed"] is not None:
        import torch

        generator = torch.Generator("cuda").manual_seed(opts["seed"])
    result = pipe(
        opts["prompt"],
        negative_prompt=opts["negative_prompt"],
        num_inference_steps=opts["steps"],
        guidance_scale=opts["guidance_scale"],
        audio_end_in_s=opts["duration_s"],
        num_waveforms_per_prompt=1,
        generator=generator,
    )
    audio = result.audios[0].T.float().cpu().numpy()
    if audio.size == 0 or not np.all(np.isfinite(audio)):
        raise RuntimeError("the render is empty or not finite")
    if float(np.abs(audio).max()) < SILENCE_PEAK:
        raise RuntimeError("the render is silent")
    return audio, int(pipe.vae.sampling_rate)


def write_wav(path: str, audio: Any, sample_rate: int) -> float:
    """Write frames x channels float audio as 16-bit PCM; return the duration in seconds."""
    import numpy as np

    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).round().astype("<i2")
    if pcm.ndim == 1:
        pcm = pcm[:, None]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with wave.open(path, "wb") as fh:
        fh.setnchannels(pcm.shape[1])
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(pcm.tobytes())
    return pcm.shape[0] / float(sample_rate)


def run(manifest: dict, out_dir: str, model_dir: str, load=load_pipeline, check=probe) -> dict:
    opts = parse_options(manifest)
    pipe = load(model_dir)
    check(pipe)
    audio, sample_rate = render(pipe, opts)
    duration = write_wav(os.path.join(out_dir, opts["wav"]), audio, sample_rate)
    sidecar = {
        "model": os.path.basename(model_dir.rstrip("/")),
        "sample_rate": sample_rate,
        "channels": int(audio.shape[1]) if audio.ndim == 2 else 1,
        "duration_s": round(duration, 3),
        "settings": {k: v for k, v in opts.items() if k not in ("wav", "json")},
    }
    if opts["json"] is not None:
        dest = os.path.join(out_dir, opts["json"])
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(sidecar, fh, indent=2)
    return sidecar


def main() -> int:
    job_id = os.environ.get("GPUWORKER_JOB_ID", "")
    bundle_dir = os.environ.get("GPUWORKER_BUNDLE", "/job/bundle")
    out_dir = os.environ.get("GPUWORKER_OUT", "/job/out")
    model_dir = os.environ.get("AUDIOGEN_MODEL_DIR", DEFAULT_MODEL_DIR)
    try:
        with open(
            os.path.join(bundle_dir, "jobs", "pending", f"{job_id}.json"), encoding="utf-8"
        ) as fh:
            manifest = json.load(fh)
        print(src_fingerprint(), flush=True)
        info = run(manifest, out_dir, model_dir)
    except BadJob as exc:
        print(f"bad job: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"generation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return exit_code_for(exc)
    print(f"{job_id}: {info['duration_s']} s at {info['sample_rate']} Hz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
