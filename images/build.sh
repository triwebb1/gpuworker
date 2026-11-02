#!/usr/bin/env bash
# Build the one shared gpuworker sandbox image (SPEC 6.1). Run on the GPU host (Linux or
# WSL2), where podman/docker + the NVIDIA Container Toolkit live. The build is online and
# needs no GPU. After building, pin the digest into the broker config.
#
# Usage:
#   images/build.sh [--runtime podman|docker] [--tag <tag>] [--only "m1 m2"] [--pin] [--repin]
#
#   --runtime  container runtime (default: podman)
#   --tag      image tag (default: gpuworker/sandbox:cuda12.8)
#   --only     space-separated module subset to provision (e.g. "echo" or
#              "echo mymodule"); empty = all. Use "echo" for a fast first build.
#   --pin      after build, print the @sha256 digest to paste into gpuworker.toml
#   --repin    if gpuworker.toml pins a local image id, rewrite it to the new build's id
#
# All module sources are committed in-repo under modules/<name>/src, so the build needs
# no clone and no SSH. The Dockerfile COPYs the whole modules/ tree and runs each
# modules/<name>/provision.sh.
set -euo pipefail

RUNTIME="podman"
TAG="gpuworker/sandbox:cuda12.8"
ONLY=""
PIN=0
REPIN=0
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"   # build context = repo root (modules/ lives there)

while [ $# -gt 0 ]; do
  case "$1" in
    --runtime) RUNTIME="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    --only) ONLY="$2"; shift 2;;
    --pin) PIN=1; shift;;
    --repin) REPIN=1; shift;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done

echo "== building ${TAG} with ${RUNTIME}${ONLY:+ (only: $ONLY)} (online build, no GPU needed) =="
# Recent Docker BuildKit adds provenance+SBOM attestations by default, which makes the
# result a manifest list (image index). `docker image inspect .Id` + `docker run <id>`
# (the re-pin check below) don't work on an index, so the pin would silently never
# update. Disabling attestations gives a single-manifest image with a runnable local id.
EXTRA=""
[ "${RUNTIME}" = "docker" ] && { export DOCKER_BUILDKIT=1; EXTRA="--provenance=false --sbom=false"; }
"${RUNTIME}" build ${EXTRA} -f "${HERE}/Dockerfile.sandbox" -t "${TAG}" \
  --build-arg "PROVISION_ONLY=${ONLY}" "${REPO}"

echo "== built ${TAG} =="

# The Dockerfile keeps building past a failed module so one build reports every failure.
# That makes "the build succeeded" mean nothing on its own: read the per-module status and
# fail loudly, before anything below can pin a partial image.
STATUS="$("${RUNTIME}" run --rm --entrypoint cat "${TAG}" /opt/modules/.provision_status 2>/dev/null || true)"
echo "${STATUS}" | sed 's/^/   /'
if [ -z "${STATUS}" ] || echo "${STATUS}" | grep -q ' FAIL$'; then
  echo "== provisioning FAILED for:$(echo "${STATUS}" | awk '$2=="FAIL"{printf " %s", $1}') ==" >&2
  echo "   ${TAG} was built but is partial; not re-pinning gpuworker.toml." >&2
  exit 1
fi

# A rebuild gives the image a new local id. A config pinned to the old local id keeps
# running the old image (or fails with "No such image" once it is pruned), so say so, and
# rewrite the pin only when asked: changing what the broker runs is the operator's call.
# A registry-digest pin is never touched.
CFG="${REPO}/gpuworker.toml"
if [ -f "${CFG}" ] && grep -qE '^\s*sandbox_image\s*=\s*"sha256:[0-9a-f]{64}"' "${CFG}"; then
  NEWID="$("${RUNTIME}" image inspect --format '{{.Id}}' "${TAG}" 2>/dev/null || true)"
  if [ -n "${NEWID}" ] && ! grep -q "\"${NEWID}\"" "${CFG}"; then
    if [ "${REPIN}" = "1" ] && "${RUNTIME}" run --rm --entrypoint true "${NEWID}" >/dev/null 2>&1; then
      sed -i -E "s|^(\s*sandbox_image\s*=).*|\1 \"${NEWID}\"|" "${CFG}"
      echo "== re-pinned gpuworker.toml sandbox_image -> ${NEWID} =="
    else
      echo "== NOTE: gpuworker.toml still pins the previous image; the broker keeps running it." >&2
      echo "   Re-run with --repin, or set sandbox_image = \"${NEWID}\" yourself. ==" >&2
    fi
  fi
fi

if [ "${PIN}" = "1" ]; then
  # Resolve the local image digest to pin in gpuworker.toml sandbox_image.
  DIGEST="$("${RUNTIME}" inspect --format '{{index .RepoDigests 0}}' "${TAG}" 2>/dev/null || true)"
  if [ -n "${DIGEST}" ]; then
    echo "PIN this in gpuworker.toml:  sandbox_image = \"${DIGEST}\""
  else
    ID="$("${RUNTIME}" image inspect --format '{{.Id}}' "${TAG}")"
    echo "no repo digest yet (image not pushed). Local id: ${ID}"
    echo "Push to your registry, then re-run with --pin to get the @sha256 digest."
  fi
fi

cat <<'EOF'

Next steps:
  1. Verify the host and the GPU at run time (NOT at build):
       gpuworker doctor && gpuworker run-once echo.container
  2. Stage weights (separate from the image):        gpuworker stage-weights <job_type>
  3. Dev: set allow_unpinned_images=true + sandbox_image=<tag> in gpuworker.toml
     Prod: push the image, pin sandbox_image to the @sha256 digest, flag back to false.
EOF
