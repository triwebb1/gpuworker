#!/usr/bin/env bash
# Smoke test: echo.container (container launcher path). Needs the built sandbox image.
exec python -m gpuworker.cli run-once echo.container --text "smoke-echo-container" --output out/smoke.txt
