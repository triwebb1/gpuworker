# audiogen

Background music, ambience and sound effects from a text prompt. One job type,
`audiogen.stableaudio`, runs
[Stable Audio Open 1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0) through
diffusers at fp16: oneshot, `standard` sandbox, up to 47 seconds of stereo per clip.

The worker runs a one-step probe generation after loading, and refuses a finished clip
that is empty, not finite, or silent. A pipeline loaded from an incomplete offline cache
can load without error and then produce nothing; these checks make that a named failure
instead of a timeout or a silent file. The options are documented at the top of
[`src/audiogen/generate.py`](src/audiogen/generate.py).

## Staging the weights

The model repository is gated on Hugging Face: accept its terms and log in with
`hf auth login` first. Then, from the repository root:

```bash
hf download stabilityai/stable-audio-open-1.0 --local-dir weights/audiogen.stableaudio/stable-audio-open-1.0
gpuworker stage-weights audiogen.stableaudio
```

## Jobs

A music bed and a sound effect, in one bundle:

```json
{
  "job_id": "bed-01",
  "job_type": "audiogen.stableaudio",
  "outputs": ["out/bed-01.wav", "out/bed-01.json"],
  "options": {"prompt": "calm ambient piano, slow tempo, soft pads, background music",
              "duration_s": 45, "seed": 7}
}
```

```json
{
  "job_id": "sfx-door",
  "job_type": "audiogen.stableaudio",
  "outputs": ["out/sfx-door.wav"],
  "options": {"prompt": "a heavy wooden door creaking open, close up, dry room",
              "duration_s": 4, "seed": 3}
}
```

A clip longer than 47 seconds is several jobs. The sidecar records the settings, seed
included.

## Smoke test

`smoke/audiogen.stableaudio.sh` renders a 10-second music clip and a 3-second sound
effect.

## Footprint

`vram_mb` in `module.toml` is the declared footprint. The broker measures the type's peak
when it loads alone and raises the footprint if the declaration was low.
