"""Bring-up preflight (`gpuworker doctor`). Run this on the GPU host (Linux or a WSL2
distro): it checks each prerequisite for running jobs and prints one line per check with
what is missing. Each check is best-effort and never raises; the command exits non-zero
if any hard requirement fails."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

from . import module as _mod
from . import weights as _weights
from .config import Config
from .launcher import LaunchSpec, make_launcher

OK, WARN, FAIL = "ok", "warn", "fail"
_MARK = {OK: "PASS", WARN: "WARN", FAIL: "FAIL"}

# Free space below which the disk check warns, and below which it fails. The broker
# writes unpacked bundles and outputs to state_dir and result bundles to each queue root.
DISK_WARN_MB = 5 * 1024
DISK_FAIL_MB = 1024


def _run(cmd: list[str], timeout: float = 15.0) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, str(exc)


# Run inside the sandbox image by the GPU probe: initialize CUDA through the driver API
# (no torch needed) and count the devices the container was given.
_CUDA_PROBE = (
    "import ctypes, sys\n"
    "cuda = ctypes.CDLL('libcuda.so.1')\n"
    "rc = cuda.cuInit(0)\n"
    "n = ctypes.c_int(0)\n"
    "if rc == 0:\n"
    "    rc = cuda.cuDeviceGetCount(ctypes.byref(n))\n"
    "print(f'cuInit={rc} devices={n.value}')\n"
    "sys.exit(0 if rc == 0 and n.value > 0 else 1)\n"
)


def _cdi_check(gpu_index: int) -> tuple[str, str]:
    """Is the device the broker asks Podman for (``nvidia.com/gpu=<index>``) in the CDI
    spec? A spec generated on WSL2 can list only ``nvidia.com/gpu=all``."""
    rc, o = _run(["nvidia-ctk", "cdi", "list"])
    if rc != 0:
        return WARN, f"nvidia-ctk cdi list failed: {o.strip()[:80]}"
    names = {line.strip() for line in o.splitlines()}
    want = f"nvidia.com/gpu={gpu_index}"
    if want in names:
        uuids = sorted(n for n in names if n.startswith("nvidia.com/gpu=GPU-"))
        return OK, f"{want} present" + (f" ({len(uuids)} by UUID too)" if uuids else "")
    if "nvidia.com/gpu=all" in names:
        return FAIL, f"the CDI spec has nvidia.com/gpu=all but not {want}, which jobs request"
    return FAIL, f"{want} is not in the CDI spec; regenerate it with nvidia-ctk cdi generate"


def gpu_probe(cfg: Config, timeout: float = 180.0) -> tuple[str, str]:
    """Start the sandbox image the way the broker starts a vram-lane job (the same
    hardening, user and GPU flags) and initialize CUDA in it. This is the one check that
    exercises the runtime, the GPU injection and the driver together."""
    jt = _mod.JobType(
        id="doctor.gpu_probe",
        resource_class=_mod.LANE_VRAM,
        vram_mb=1,
        env="container",
        entrypoint=["python3", "-c", _CUDA_PROBE],
    )
    launcher = make_launcher(cfg.container_runtime, cfg.gpu_index)
    with tempfile.TemporaryDirectory(prefix="gpuworker-doctor-") as tmp:
        bundle_dir, out_dir = os.path.join(tmp, "bundle"), os.path.join(tmp, "out")
        os.makedirs(bundle_dir)
        os.makedirs(out_dir)
        spec = LaunchSpec(
            jt,
            bundle_dir=bundle_dir,
            out_dir=out_dir,
            image=cfg.sandbox_image,
            tmpfs_mb=cfg.tmpfs_mb,
            user=cfg.container_user,
            name=f"gpuworker-doctor-{os.getpid()}",
        )
        try:
            rc, o = _run(launcher.build_argv(spec), timeout=timeout)
        finally:
            launcher.cleanup(spec)
    last = o.strip().splitlines()[-1][:120] if o.strip() else ""
    if rc == 0:
        return OK, f"CUDA initialized in the sandbox: {last}"
    return FAIL, f"sandbox GPU probe exited {rc}: {last or 'no output'}"


def _existing(path: str) -> str | None:
    """*path*, or its nearest existing parent (a dir the broker has not created yet)."""
    path = os.path.abspath(path)
    while not os.path.isdir(path):
        parent = os.path.dirname(path)
        if parent == path:
            return None
        path = parent
    return path


def disk_checks(cfg: Config) -> list[tuple[str, str, str]]:
    """Free space where the broker writes (state_dir, each queue root) and where weights
    are staged."""
    out: list[tuple[str, str, str]] = []
    paths = [("state_dir", cfg.state_dir), ("weights_dir", cfg.weights_dir)]
    paths += [(f"queue:{root}", root) for root in cfg.queue_roots]
    for name, path in paths:
        where = _existing(path)
        if where is None:
            continue
        try:
            free_mb = shutil.disk_usage(where).free // (1024 * 1024)
        except OSError as exc:
            out.append((f"disk:{name}", WARN, f"could not read free space: {exc}"))
            continue
        level = FAIL if free_mb < DISK_FAIL_MB else WARN if free_mb < DISK_WARN_MB else OK
        out.append((f"disk:{name}", level, f"{free_mb} MB free at {where}"))
    return out


def _checks(cfg: Config) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []

    # Platform: the broker is a Linux process (natively or inside a WSL2 distro).
    rel = ""
    try:
        with open("/proc/version", encoding="utf-8") as fh:
            rel = fh.read().lower()
    except OSError:
        pass
    out.append(
        (
            "platform",
            OK if rel else WARN,
            ("Linux (WSL2)" if "microsoft" in rel else "Linux")
            if rel
            else "not Linux; run doctor on the GPU host (Linux or a WSL2 distro)",
        )
    )

    # Container runtime.
    rt = cfg.container_runtime
    if rt == "local":
        out.append(("runtime", WARN, "container_runtime=local (no isolation; dev only)"))
    elif shutil.which(rt):
        rc, o = _run([rt, "--version"])
        out.append(("runtime", OK if rc == 0 else FAIL, f"{rt}: {o.strip()[:60]}"))
    else:
        out.append(("runtime", FAIL, f"{rt} not found on PATH"))

    # GPU visible.
    rc, o = _run(["nvidia-smi", "-L"])
    out.append(
        (
            "gpu",
            OK if rc == 0 else FAIL,
            o.strip().splitlines()[0][:70] if rc == 0 else "nvidia-smi failed",
        )
    )

    # CDI (Podman GPU injection). Docker uses --gpus and doesn't need CDI.
    if rt == "podman":
        has_ctk = shutil.which("nvidia-ctk") is not None
        has_spec = os.path.isfile("/etc/cdi/nvidia.yaml") or os.path.isfile(
            "/var/run/cdi/nvidia.yaml"
        )
        if has_ctk and has_spec:
            level, detail = _cdi_check(cfg.gpu_index)
            out.append(("cdi", level, detail))
        elif has_ctk:
            out.append(
                (
                    "cdi",
                    WARN,
                    "nvidia-ctk present but no CDI spec; run: "
                    "sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml",
                )
            )
        else:
            out.append(("cdi", FAIL, "nvidia-ctk not found (NVIDIA Container Toolkit)"))

    # The sandbox itself, with the GPU: only if the pieces above are in place.
    if rt in ("podman", "docker"):
        if all(lvl != FAIL for _n, lvl, _d in out):
            level, detail = gpu_probe(cfg)
            out.append(("sandbox_gpu", level, detail))
        else:
            out.append(("sandbox_gpu", WARN, "skipped: fix the failures above first"))

    # Registry + image resolution.
    try:
        reg = _mod.scan_registry(cfg.registry_dir)
        n_err = 0
        for jt_id, jt in sorted(reg.job_types.items()):
            level, msg = _mod.image_status(jt, cfg.sandbox_image, cfg.allow_unpinned_images)
            if level == "error":
                n_err += 1
                out.append((f"image:{jt_id}", FAIL, msg))
        out.append(
            (
                "registry",
                FAIL if n_err else OK,
                f"{len(reg.job_types)} job types, {len(reg.modules)} modules"
                + (f", {n_err} image error(s)" if n_err else ""),
            )
        )
        weighted_jts = [
            jt
            for jt in reg.job_types.values()
            if jt.resource_class == _mod.LANE_VRAM or jt.weights_required
        ]
    except Exception as exc:
        out.append(("registry", FAIL, f"scan failed: {exc}"))
        weighted_jts = []

    # Weights staged + verified for each vram job and every job type that explicitly
    # requires a cache (CPU models need weights too).
    for jt in sorted(weighted_jts, key=lambda j: j.id):
        cache = os.path.join(cfg.weights_dir, jt.id)
        manifest = os.path.join(cache, _weights.MANIFEST_NAME)
        if not os.path.isdir(cache):
            level = FAIL if jt.weights_required else WARN
            out.append(
                (
                    f"weights:{jt.id}",
                    level,
                    f"required cache missing at {cache}"
                    if jt.weights_required
                    else f"no cache at {cache} (stage before run)",
                )
            )
        elif not os.path.isfile(manifest):
            level = FAIL if jt.weights_required else WARN
            out.append(
                (
                    f"weights:{jt.id}",
                    level,
                    f"required cache unsealed - run: gpuworker stage-weights {jt.id}"
                    if jt.weights_required
                    else f"cache present but unhashed - run: gpuworker stage-weights {jt.id}",
                )
            )
        else:
            ok, problems = _weights.verify(cache)
            out.append(
                (
                    f"weights:{jt.id}",
                    OK if ok else FAIL,
                    "verified" if ok else f"verify failed: {problems[:2]}",
                )
            )

    # Queue roots reachable + writable (read-only check; broker creates the tree).
    if not cfg.queue_roots:
        out.append(("queues", WARN, "no queue_roots configured"))
    for root in cfg.queue_roots:
        # A Windows UNC (\\host\share, //host/share) or drive-letter (Z:\...) path can't be
        # used by the Linux broker: os.makedirs fails (PermissionError). Catch it here
        # instead of crashing at serve start.
        if root.startswith(("//", "\\\\")) or (len(root) >= 2 and root[1] == ":"):
            out.append(
                (
                    f"queue:{root}",
                    FAIL,
                    "Windows path; mount the share inside Linux and set "
                    "queue_roots to its Linux mountpoint (for example "
                    "/srv/gpuworker/queues/<client>)",
                )
            )
            continue
        parent = root if os.path.isdir(root) else os.path.dirname(root.rstrip("/\\"))
        if not os.path.isdir(parent):
            out.append((f"queue:{root}", WARN, "not present yet (broker creates it on start)"))
        elif os.access(parent, os.W_OK):
            out.append((f"queue:{root}", OK, "reachable + writable"))
        else:
            out.append((f"queue:{root}", FAIL, "not writable"))

    out += disk_checks(cfg)

    # Sandbox image pinned (prod) vs dev tag.
    if _mod.is_pinned(cfg.sandbox_image):
        out.append(("sandbox_image", OK, "pinned by digest"))
    else:
        out.append(
            (
                "sandbox_image",
                WARN,
                f"unpinned tag {cfg.sandbox_image} (dev only; pin for prod, "
                f"strict jobs won't run until pinned)",
            )
        )

    return out


def run(cfg: Config) -> int:
    rows = _checks(cfg)
    width = max(len(name) for name, _, _ in rows)
    fails = warns = 0
    for name, level, detail in rows:
        if level == FAIL:
            fails += 1
        elif level == WARN:
            warns += 1
        print(f"  [{_MARK.get(level, level)}] {name.ljust(width)}  {detail}")
    print()
    if fails:
        print(
            f"NOT READY: {fails} failure(s), {warns} warning(s). "
            f"Resolve the FAILs above, then re-run `gpuworker doctor`."
        )
        return 1
    print(
        f"READY (with {warns} warning(s)). "
        f"Warnings are usually staging/pinning steps - see README.md."
    )
    return 0
