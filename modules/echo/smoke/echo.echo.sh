#!/usr/bin/env bash
# Smoke test: echo.echo (host process / local launcher path). No GPU, no weights.
exec python -m gpuworker.cli run-once echo.echo --text "smoke-echo-echo" --output out/smoke.txt
