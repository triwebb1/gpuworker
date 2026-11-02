#!/usr/bin/env bash
# Smoke test: speech.voxcpm2 (serve, vram lane). Needs the sandbox image with the speech
# venv and the sealed weights in weights/speech.voxcpm2/VoxCPM2. No reference clip, so the
# model picks a speaker.
exec python -m gpuworker.cli run-once --keep --timeout 900 speech.voxcpm2 \
  --text "The broker admitted this job against its memory budget." --output out/smoke.wav
