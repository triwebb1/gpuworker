#!/usr/bin/env bash
# Smoke test: speech.voxcpm2 with a persona. Pass a reference WAV as $1, in a directory of
# its own (the directory becomes the bundle), and its transcript as $2. The WAV from
# speech.voxcpm2.sh works, with that script's sentence as the transcript.
set -euo pipefail
WAV="$1"
REF_TEXT="$2"
exec python -m gpuworker.cli run-once --keep --timeout 900 speech.voxcpm2 --bundle "$(dirname "$WAV")" \
  --option "reference=$(basename "$WAV")" --option "reference_text=${REF_TEXT}" \
  --option seed=7 \
  --text "A second sentence, in the same voice as the reference recording." \
  --output out/persona.wav --output out/persona.json
