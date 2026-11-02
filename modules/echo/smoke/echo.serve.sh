#!/usr/bin/env bash
# Smoke test: echo.serve (serve-mode warm container, jobs streamed over the socket).
# Needs the built sandbox image (echo venv). run-once tears the warm container down after.
exec python -m gpuworker.cli run-once echo.serve --text "serve-smoke" --output out/smoke.txt
