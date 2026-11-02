#!/usr/bin/env bash
# Speech module provisioning: builds /opt/venvs/speech for speech.voxcpm2 and speech.check.
# Weights are not installed here; they are staged on the broker host (README.md).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="/opt/venvs/speech"

python3 -m venv "${VENV}"
"${VENV}/bin/python" -m pip install --upgrade pip >/dev/null
# CUDA 12.8 torch first, so the requirements below resolve against it instead of pulling
# a default build. CUDA 12.8 wheels include kernels for current (sm_120) cards.
"${VENV}/bin/pip" install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.11.0 torchaudio==2.11.0
"${VENV}/bin/pip" install -r "${HERE}/src/requirements.txt"
# Put the module's source on the venv's path instead of pip-installing it, so jobs run the
# files in /opt/modules/speech/src that the runtime fingerprint hashes (MODULE_GUIDE 1.5).
SITE="$("${VENV}/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "${HERE}/src" > "${SITE}/gpuworker-speech.pth"

# The entrypoints and their libraries must import; the GPU is checked at the first run.
"${VENV}/bin/python" -c "import speech.tts, speech.check, voxcpm, faster_whisper"
echo "provisioned speech venv at ${VENV}"
