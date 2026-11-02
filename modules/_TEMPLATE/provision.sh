#!/usr/bin/env bash
# Copy to modules/<your-module>/provision.sh. Builds /opt/venvs/<module> for the source in
# ./src (committed in-repo). No weights here (staged separately, SPEC 6.2).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODULE="<module>"
VENV="/opt/venvs/${MODULE}"
SRC="${HERE}/src"

python3 -m venv "${VENV}"
"${VENV}/bin/python" -m pip install --upgrade pip >/dev/null
[ -d "${SRC}" ] || { echo "ERROR: commit your worker source to modules/${MODULE}/src/" >&2; exit 1; }
# If you need torch: install the cu128 wheel first, then the rest.
#   "${VENV}/bin/pip" install --index-url https://download.pytorch.org/whl/cu128 "torch>=2.2"
if [ -f "${SRC}/requirements.txt" ]; then
    "${VENV}/bin/pip" install -r "${SRC}/requirements.txt"
fi
# Put the source on the venv's path instead of pip-installing it, so jobs run the files in
# /opt/modules/<module>/src that the runtime fingerprint hashes (MODULE_GUIDE 1.5). A pip
# install would copy them into site-packages, and the fingerprint would no longer match.
SITE="$("${VENV}/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "${SRC}" > "${SITE}/gpuworker-${MODULE}.pth"
"${VENV}/bin/python" -c "import <your.worker>"   # sanity: entrypoint must import
echo "provisioned ${MODULE} venv at ${VENV}"
