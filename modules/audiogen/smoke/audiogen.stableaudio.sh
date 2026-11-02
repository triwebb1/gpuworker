#!/usr/bin/env bash
# Smoke test: audiogen.stableaudio (oneshot, vram lane). Needs the sandbox image with the
# audiogen venv and the sealed weights in weights/audiogen.stableaudio/stable-audio-open-1.0.
# Renders a short music clip, then a sound effect.
set -euo pipefail
python -m gpuworker.cli run-once --keep --timeout 900 audiogen.stableaudio \
  --option "prompt=calm ambient piano, slow tempo, background music" --option duration_s=10 \
  --option seed=7 --output out/music.wav --output out/music.json
python -m gpuworker.cli run-once --keep --timeout 900 audiogen.stableaudio \
  --option "prompt=a heavy wooden door creaking open, close up" --option duration_s=3 \
  --option seed=3 --output out/sfx.wav --output out/sfx.json
