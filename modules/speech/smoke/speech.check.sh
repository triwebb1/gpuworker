#!/usr/bin/env bash
# Smoke test: speech.check (oneshot, vram lane). Pass a rendered WAV as $1, in a directory of
# its own (the directory becomes the bundle), and the text it should contain as $2 (default:
# the sentence speech.voxcpm2.sh renders).
set -euo pipefail
WAV="$1"
TEXT="${2:-The broker admitted this job against its memory budget.}"
exec python -m gpuworker.cli run-once --keep --timeout 900 speech.check --bundle "$(dirname "$WAV")" \
  --option "audio=$(basename "$WAV")" --option "text=${TEXT}" --output out/check.json
