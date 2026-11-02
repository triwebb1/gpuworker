#!/usr/bin/env bash
# Audiogen module provisioning: builds /opt/venvs/audiogen for audiogen.stableaudio.
# Weights are not installed here; they are staged on the broker host (README.md).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="/opt/venvs/audiogen"

python3 -m venv "${VENV}"
"${VENV}/bin/python" -m pip install --upgrade pip >/dev/null
# CUDA 12.8 torch first, so the requirements below resolve against it.
"${VENV}/bin/pip" install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0
"${VENV}/bin/pip" install -r "${HERE}/src/requirements.txt"
# Put the module's source on the venv's path instead of pip-installing it, so jobs run the
# files in /opt/modules/audiogen/src that the runtime fingerprint hashes (MODULE_GUIDE 1.5).
SITE="$("${VENV}/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "${HERE}/src" > "${SITE}/gpuworker-audiogen.pth"

"${VENV}/bin/python" -c "import audiogen.generate, diffusers; diffusers.StableAudioPipeline"
echo "provisioned audiogen venv at ${VENV}"
