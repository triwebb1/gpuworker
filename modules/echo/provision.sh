#!/usr/bin/env bash
# Echo module provisioning: builds /opt/venvs/echo from this module's in-repo source.
# Self-contained (stdlib only); the worked example of the modules/<name>/ contract.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="/opt/venvs/echo"

python3 -m venv "${VENV}"
"${VENV}/bin/python" -m pip install --upgrade pip >/dev/null
# Install this module's workers (stdlib only; a module with packages puts ./src on the
# venv's path instead, as modules/_TEMPLATE/provision.sh does).
install -D -m 0755 "${HERE}/src/worker.py" /opt/echo/worker.py
install -D -m 0755 "${HERE}/src/serve_worker.py" /opt/echo/serve_worker.py
echo "provisioned echo venv at ${VENV}"
