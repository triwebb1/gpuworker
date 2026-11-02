"""Modules + job types (SPEC 4.1, docs/MODULE_GUIDE.md).

A module is a declarative ``module.toml`` declaring one or more job types, each with
a resource footprint and how to execute it. The broker reads these by metadata only;
it never imports torch or module code. The registry configures the sandbox, so
only the host admin writes it (SPEC 6.3); loading it rejects floating image tags and
enforces a sandbox floor per lane.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import time
import tomllib
from dataclasses import dataclass, field

# Resource lanes (SPEC 5).
LANE_VRAM = "vram"
LANE_CPU = "cpu"
LANES = frozenset({LANE_VRAM, LANE_CPU})

# Sandbox levels (SPEC 6), ordered weakest -> strongest.
SANDBOX_LEVELS = ("none", "standard", "strict")

# VRAM allocation profile (SPEC 5): does the model allocate its full footprint at
# load and stay flat, or grow as it works through a job? Only a `preallocated` model
# has a footprint that is fully known at admit time, so only `preallocated` models are
# eligible to co-reside; a `growing` model's peak isn't realized until mid-job, which
# makes packing it unsafe; it gets the card to itself.
VRAM_PREALLOCATED = "preallocated"
VRAM_GROWING = "growing"
VRAM_PROFILES = frozenset({VRAM_PREALLOCATED, VRAM_GROWING})

# Execution environments (SPEC 4.1).
ENV_CONTAINER = "container"  # a sandboxed container via the configured runtime
ENV_HOST = "host"  # a plain host process via the local launcher: no isolation
ENVS = frozenset({ENV_CONTAINER, ENV_HOST})
CONTAINER_ENVS = frozenset({ENV_CONTAINER})

MODES = frozenset({"serve", "oneshot"})

# A module registry entry may inject fixed, non-sensitive build identity into its worker
# process (for example the exact source revision used to vendor an adapter). The
# registry is trusted, but broker-owned runtime variables must remain broker-owned.
_ENV_NAME_RE = re.compile(r"\A[A-Z_][A-Z0-9_]*\Z")
_RESERVED_FIXED_ENV = frozenset(
    {
        "GPUWORKER_JOB_ID",
        "GPUWORKER_BUNDLE",
        "GPUWORKER_OUT",
        "GPUWORKER_SERVE_SOCKET",
        "GPUWORKER_CPU_BUDGET",
        "GPUWORKER_MEM_BUDGET_MB",
        "GPUWORKER_SCRATCH",
        "HF_HOME",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "NVIDIA_DRIVER_CAPABILITIES",
        "NVIDIA_VISIBLE_DEVICES",
    }
)

# A pinned image reference is immutable content-addressing, not a mutable tag. Two forms
# count as pinned (SPEC 6.1, 6.3):
#   * a registry digest:  repo/name@sha256:<64hex>   (portable across machines)
#   * a local image id:   sha256:<64hex>             (this Docker/Podman daemon only)
# The local id is what `docker build` produces and is just as immutable as a registry
# digest for the "run exactly these bytes" guarantee strict needs, so a single-box
# worker can pin without standing up a registry (images/build.sh --pin uses it).
# \Z, not $, so a trailing newline cannot pass.
_DIGEST_RE = re.compile(r"@sha256:[0-9a-f]{64}\Z")
_LOCAL_ID_RE = re.compile(r"\Asha256:[0-9a-f]{64}\Z")


def is_pinned(image: str) -> bool:
    """True if *image* is pinned (immutable), not a mutable tag: either a registry
    digest (repo@sha256:...) or a bare local image id (sha256:...)."""
    return bool(_DIGEST_RE.search(image) or _LOCAL_ID_RE.match(image))


class ModuleError(ValueError):
    """A module.toml is malformed or violates a registry policy."""


@dataclass
class JobType:
    id: str
    resource_class: str
    vram_mb: int
    env: str
    entrypoint: list[str]
    image: str = ""
    modes: tuple[str, ...] = ("oneshot",)
    load_cost_s: float = 0.0
    keep_warm_s: float = 0.0
    co_resident: bool = False
    vram_profile: str = VRAM_GROWING
    sandbox: str = "standard"
    serve_socket: str = ""
    serve_job_timeout: float = 300.0  # serve: per-job socket round-trip bound (seconds)
    oneshot_job_timeout_s: float = 0.0  # 0 => broker default; manifest may still override
    weights_required: bool = False  # fail before launch unless a sealed cache is present
    fixed_env: dict[str, str] = field(default_factory=dict)
    env_passthrough: tuple[str, ...] = ()  # broker-env names forwarded into the job
    serve_max_jobs: int = 0  # serve: recycle the container after N jobs (0 = never)
    module_name: str = ""

    @property
    def is_container(self) -> bool:
        return self.env in CONTAINER_ENVS

    def __post_init__(self) -> None:
        # A strict loader executes whatever weights it is given, so it never runs without a
        # sealed, verified cache (SPEC 6.2).
        if self.sandbox == "strict":
            self.weights_required = True

    def supports(self, mode: str) -> bool:
        return mode in self.modes

    @property
    def packable(self) -> bool:
        """Eligible to co-reside with other models. Requires opt-in (`co_resident`)
        and a `preallocated` profile (a known, stable footprint at admit). A growing
        model is never packed: its peak is unknown until it runs (SPEC 5)."""
        return (
            self.resource_class == LANE_VRAM
            and self.co_resident
            and self.vram_profile == VRAM_PREALLOCATED
        )


def _sandbox_floor(resource_class: str, sandbox: str) -> None:
    """Enforce the per-lane sandbox floor (SPEC 6.3). The broker refuses to run a
    job type below the isolation its lane/risk implies."""
    if sandbox not in SANDBOX_LEVELS:
        raise ModuleError(f"unknown sandbox level: {sandbox!r}")
    # A cpu-lane job that runs no model code may declare `none`. vram jobs load models;
    # they must be at least `standard`.
    if resource_class == LANE_VRAM and sandbox == "none":
        raise ModuleError(
            "vram-lane job types must declare sandbox >= 'standard' (they load model code)"
        )


def _seconds(raw: dict, jt_id: str, key: str, default: float, positive: bool = False) -> float:
    """A duration field: a finite number, at least 0 (above 0 with *positive*). NaN or inf
    would make a deadline that never passes, so they are refused like a negative value."""
    v = raw.get(key, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ModuleError(f"{jt_id}: {key} must be a finite number of seconds, got {v!r}")
    if v < 0 or (positive and v == 0):
        bound = "greater than 0" if positive else "at least 0"
        raise ModuleError(f"{jt_id}: {key} must be {bound}, got {v!r}")
    return float(v)


# The keys a [[job_type]] table may have (MODULE_GUIDE 2.2) and those of [module].
JOB_TYPE_KEYS = frozenset(
    {
        "id",
        "resource_class",
        "vram_mb",
        "vram_profile",
        "co_resident",
        "env",
        "entrypoint",
        "image",
        "modes",
        "load_cost_s",
        "keep_warm_s",
        "sandbox",
        "serve_socket",
        "serve_job_timeout",
        "serve_max_jobs",
        "oneshot_job_timeout_s",
        "weights_required",
        "fixed_env",
        "env_passthrough",
    }
)
MODULE_KEYS = frozenset({"name", "version"})


def _unknown_keys(table: dict, known: frozenset[str], where: str) -> None:
    """Refuse a key the broker does not read: a misspelling would otherwise keep the
    default without a word."""
    unknown = sorted(set(table) - known)
    if unknown:
        raise ModuleError(f"{where}: unknown key(s) {', '.join(map(repr, unknown))}")


def _parse_job_type(raw: dict, module_name: str) -> JobType:
    def req(key: str):
        if key not in raw:
            raise ModuleError(f"job_type missing required key: {key!r}")
        return raw[key]

    jt_id = req("id")
    if not isinstance(jt_id, str) or "." not in jt_id:
        raise ModuleError(f"job_type id must be 'module.name', got {jt_id!r}")
    _unknown_keys(raw, JOB_TYPE_KEYS, jt_id)

    resource_class = raw.get("resource_class", LANE_VRAM)
    if resource_class not in LANES:
        raise ModuleError(f"{jt_id}: resource_class must be one of {sorted(LANES)}")

    env = req("env")
    if env not in ENVS:
        raise ModuleError(f"{jt_id}: env must be one of {sorted(ENVS)}")

    entrypoint = req("entrypoint")
    if (
        not isinstance(entrypoint, list)
        or not entrypoint
        or not all(isinstance(x, str) for x in entrypoint)
    ):
        raise ModuleError(f"{jt_id}: entrypoint must be a non-empty list of strings")

    image = raw.get("image", "")
    if not isinstance(image, str):
        raise ModuleError(f"{jt_id}: image must be a string")
    if env in CONTAINER_ENVS and image and not is_pinned(image):
        # An image is optional (empty => inherit the broker's shared sandbox_image), but
        # a module's own image must be immutable: an @sha256: registry digest or a bare
        # local sha256:<hex> id, never a mutable tag. Consistent with is_pinned.
        raise ModuleError(
            f"{jt_id}: a module-declared image must be pinned (an @sha256: digest or a "
            f"local sha256: id), got {image!r} (omit image to inherit the shared sandbox)"
        )

    modes_raw = raw.get("modes", ["oneshot"])
    if not isinstance(modes_raw, list) or not modes_raw or any(m not in MODES for m in modes_raw):
        raise ModuleError(f"{jt_id}: modes must be a non-empty subset of {sorted(MODES)}")

    vram_mb = raw.get("vram_mb", 0)
    if not isinstance(vram_mb, int) or isinstance(vram_mb, bool) or vram_mb < 0:
        raise ModuleError(f"{jt_id}: vram_mb must be a non-negative int")
    if resource_class == LANE_VRAM and vram_mb <= 0:
        raise ModuleError(f"{jt_id}: vram-lane job types need a measured vram_mb > 0")

    vram_profile = raw.get("vram_profile", VRAM_GROWING)
    if vram_profile not in VRAM_PROFILES:
        raise ModuleError(f"{jt_id}: vram_profile must be one of {sorted(VRAM_PROFILES)}")

    sandbox = raw.get("sandbox", "standard")
    _sandbox_floor(resource_class, sandbox)

    if env == ENV_HOST and resource_class == LANE_VRAM:
        # A vram type loads model code, which belongs in a container (SPEC 12.1). The
        # host env stays available off the GPU for trusted code such as echo.echo.
        raise ModuleError(f'{jt_id}: vram-lane job types need env = "container"')
    if env == ENV_HOST and sandbox == "strict":
        raise ModuleError(f"{jt_id}: a host-process job type cannot be strict (no sandbox)")

    if "serve" in modes_raw and resource_class != LANE_VRAM:
        # Warm serve containers are a VRAM-residency mechanism; the scheduler only runs
        # serve on the vram lane, so accepting it elsewhere would silently mean oneshot.
        raise ModuleError(f"{jt_id}: serve mode is only supported on the vram lane")
    if "serve" in modes_raw and env == ENV_HOST:
        # A warm serve worker is reached over a socket bind-mounted into its container;
        # the host launcher has no such contract.
        raise ModuleError(f'{jt_id}: serve mode requires env = "container"')

    serve_socket = raw.get("serve_socket", "")
    if not isinstance(serve_socket, str):
        raise ModuleError(f"{jt_id}: serve_socket must be a string")
    if "serve" in modes_raw and env in CONTAINER_ENVS and not serve_socket:
        raise ModuleError(f"{jt_id}: serve mode in a container requires serve_socket")

    oneshot_job_timeout_s = _seconds(raw, jt_id, "oneshot_job_timeout_s", 0.0)
    if oneshot_job_timeout_s and "oneshot" not in modes_raw:
        raise ModuleError(f"{jt_id}: oneshot_job_timeout_s requires oneshot mode")

    weights_required = raw.get("weights_required", False)
    if not isinstance(weights_required, bool):
        raise ModuleError(f"{jt_id}: weights_required must be a bool")
    co_resident = raw.get("co_resident", False)
    if not isinstance(co_resident, bool):
        raise ModuleError(f"{jt_id}: co_resident must be a bool")

    fixed_env = raw.get("fixed_env", {})
    if not isinstance(fixed_env, dict):
        raise ModuleError(f"{jt_id}: fixed_env must be a table of string values")
    for key, value in fixed_env.items():
        if not isinstance(key, str) or not _ENV_NAME_RE.fullmatch(key):
            raise ModuleError(f"{jt_id}: invalid fixed_env name {key!r}")
        if key in _RESERVED_FIXED_ENV:
            raise ModuleError(f"{jt_id}: fixed_env cannot override broker variable {key}")
        if not isinstance(value, str):
            raise ModuleError(f"{jt_id}: fixed_env {key} must be a string")

    env_passthrough = raw.get("env_passthrough", [])
    if not isinstance(env_passthrough, list):
        raise ModuleError(f"{jt_id}: env_passthrough must be a list of variable names")
    for key in env_passthrough:
        if not isinstance(key, str) or not _ENV_NAME_RE.fullmatch(key):
            raise ModuleError(f"{jt_id}: invalid env_passthrough name {key!r}")
        if key in _RESERVED_FIXED_ENV:
            raise ModuleError(f"{jt_id}: env_passthrough cannot carry broker variable {key}")

    serve_max_jobs = raw.get("serve_max_jobs", 0)
    if (
        not isinstance(serve_max_jobs, int)
        or isinstance(serve_max_jobs, bool)
        or serve_max_jobs < 0
    ):
        raise ModuleError(f"{jt_id}: serve_max_jobs must be a non-negative int")
    if serve_max_jobs and "serve" not in modes_raw:
        raise ModuleError(f"{jt_id}: serve_max_jobs requires serve mode")

    return JobType(
        id=jt_id,
        resource_class=resource_class,
        vram_mb=vram_mb,
        env=env,
        entrypoint=list(entrypoint),
        image=image,
        modes=tuple(modes_raw),
        load_cost_s=_seconds(raw, jt_id, "load_cost_s", 0.0),
        keep_warm_s=_seconds(raw, jt_id, "keep_warm_s", 0.0),
        co_resident=co_resident,
        vram_profile=vram_profile,
        sandbox=sandbox,
        serve_socket=serve_socket,
        serve_job_timeout=_seconds(raw, jt_id, "serve_job_timeout", 300.0, positive=True),
        oneshot_job_timeout_s=oneshot_job_timeout_s,
        weights_required=weights_required,
        fixed_env=dict(fixed_env),
        env_passthrough=tuple(env_passthrough),
        serve_max_jobs=serve_max_jobs,
        module_name=module_name,
    )


@dataclass
class Module:
    name: str
    version: str
    job_types: list[JobType] = field(default_factory=list)
    path: str = ""
    src_fingerprint: str = ""  # sha256 over src/, or "" if the module ships no src/
    src_files: int = 0
    src_newest_mtime: str = ""


def source_fingerprint(module_dir: str) -> tuple[str, int, str]:
    """Hash a module's ``src/`` tree: (sha256, file_count, newest mtime ISO-Z).

    A fix made in a client's repo that never reaches the deployed ``modules/<name>/src/``
    leaves every job silently running the old code while producing plausible output.
    Recording this hash in the result lets the client compare what the broker actually
    ran against what it thinks it shipped.

    Hashes path and content of every file, sorted, so a rename or a deletion moves the
    digest and not just an edit. Newest mtime is carried beside it because it is the field
    a human reads first when asking "is this stale?", but the hash is the authority, since
    mtime survives a copy that changes nothing and lies after one that changes everything.
    """
    src = os.path.join(module_dir, "src")
    if not os.path.isdir(src):
        return "", 0, ""
    h = hashlib.sha256()
    count, newest = 0, 0.0
    for root, dirs, files in os.walk(src):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for fn in sorted(files):
            if fn.endswith((".pyc", ".pyo")):
                continue
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, src).replace(os.sep, "/")
            h.update(rel.encode("utf-8"))
            h.update(b"\0")
            try:
                with open(full, "rb") as fh:
                    while chunk := fh.read(1 << 20):
                        h.update(chunk)
                newest = max(newest, os.path.getmtime(full))
                count += 1
            except OSError:
                # An unreadable file must change the digest, not be skipped into
                # invisibility; otherwise a permissions problem reads as "unchanged".
                h.update(b"<unreadable>")
                count += 1
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(newest)) if newest else ""
    return h.hexdigest(), count, stamp


def load_module(toml_path: str) -> Module:
    with open(toml_path, "rb") as fh:
        data = tomllib.load(fh)
    meta = data.get("module")
    if not isinstance(meta, dict) or "name" not in meta:
        raise ModuleError(f"{toml_path}: missing [module] name")
    name = meta["name"]
    _unknown_keys(meta, MODULE_KEYS, f"{toml_path}: [module]")
    _unknown_keys(data, frozenset({"module", "job_type"}), toml_path)
    jts_raw = data.get("job_type", [])
    if not isinstance(jts_raw, list) or not jts_raw:
        raise ModuleError(f"{toml_path}: module declares no [[job_type]]")
    if not all(isinstance(jt, dict) for jt in jts_raw):
        raise ModuleError(f"{toml_path}: job_type must be an array of tables ([[job_type]])")
    jts = [_parse_job_type(jt, name) for jt in jts_raw]
    return Module(name=name, version=str(meta.get("version", "0")), job_types=jts, path=toml_path)


@dataclass
class Registry:
    """All job types the broker knows, keyed by job_type id."""

    job_types: dict[str, JobType] = field(default_factory=dict)
    modules: list[Module] = field(default_factory=list)

    def get(self, job_type_id: str) -> JobType:
        try:
            return self.job_types[job_type_id]
        except KeyError:
            raise ModuleError(f"unknown job_type: {job_type_id!r}") from None


def resolve_image(jt: JobType, sandbox_image: str, allow_unpinned: bool) -> str:
    """Effective image for a container job type: its own pinned image if it declares
    one, else the shared ``sandbox_image``. Enforces the guardrails (SPEC 6.1, 6.3):

      * a container job type must resolve to *some* image;
      * an unpinned (tag) image is allowed only when ``allow_unpinned`` is set
        (dev bring-up), never in normal use;
      * a ``strict`` job type (RCE-class loader such as trust_remote_code) never runs on an
        unpinned image, regardless of the flag: a tag can be repointed at different
        bytes, and strict exists for code that must be the bytes that were reviewed.
    """
    if not jt.is_container:
        return ""
    image = jt.image or sandbox_image
    if not image:
        raise ModuleError(f"{jt.id}: no image - declare one or set the broker's sandbox_image")
    if not is_pinned(image):
        if jt.sandbox == "strict":
            raise ModuleError(
                f"{jt.id}: strict job type refuses an unpinned image ({image!r}); "
                f"pin it by @sha256: digest"
            )
        if not allow_unpinned:
            raise ModuleError(
                f"{jt.id}: image {image!r} is unpinned; pin a digest or set "
                f"allow_unpinned_images=true for dev bring-up"
            )
    return image


def image_status(jt: JobType, sandbox_image: str, allow_unpinned: bool) -> tuple[str, str]:
    """Classify a job type's image for `registry validate` (a lint), without raising.
    Returns (level, message) where level is "ok" | "warn" | "error".

    Mirrors resolve_image's rules but distinguishes a not-ready-yet bring-up state from
    a hard error: a `strict` job inheriting an unpinned shared tag is an error in prod
    (allow_unpinned off) but a warning in dev (it simply can't run until the image is
    pinned; launch still refuses it via resolve_image). This keeps `validate` green
    during dev bring-up while a planned strict job type is registered."""
    if not jt.is_container:
        return ("ok", "")
    image = jt.image or sandbox_image
    if not image:
        return ("error", f"{jt.id}: no image (declare one or set sandbox_image)")
    if is_pinned(image):
        return ("ok", image)
    # unpinned tag:
    if jt.sandbox == "strict":
        if allow_unpinned:
            return (
                "warn",
                f"{jt.id}: strict - won't run until the shared image is pinned ({image})",
            )
        return ("error", f"{jt.id}: strict job needs a pinned image ({image})")
    if allow_unpinned:
        return ("warn", f"{jt.id}: unpinned tag {image} (dev only)")
    return (
        "error",
        f"{jt.id}: unpinned image {image} needs allow_unpinned_images or a pinned digest",
    )


def scan_registry(registry_dir: str) -> Registry:
    """Scan a directory for ``*/module.toml`` (or ``*.toml``) and build the
    registry. Duplicate job-type ids across modules are a hard error."""
    reg = Registry()
    if not os.path.isdir(registry_dir):
        return reg
    found: list[str] = []
    for entry in sorted(os.listdir(registry_dir)):
        p = os.path.join(registry_dir, entry)
        if os.path.isdir(p):
            cand = os.path.join(p, "module.toml")
            if os.path.isfile(cand):
                found.append(cand)
        elif entry.endswith(".toml"):
            found.append(p)
    for toml_path in found:
        mod = load_module(toml_path)
        # Fingerprint once per scan, not per job, to keep disk I/O off the launch path.
        # The broker scans the registry only at startup, so a changed module (and its
        # fingerprint) takes effect after a broker restart.
        fp, n, stamp = source_fingerprint(os.path.dirname(os.path.abspath(toml_path)))
        mod.src_fingerprint, mod.src_files, mod.src_newest_mtime = fp, n, stamp
        reg.modules.append(mod)
        for jt in mod.job_types:
            if jt.id in reg.job_types:
                raise ModuleError(f"duplicate job_type id across modules: {jt.id!r}")
            reg.job_types[jt.id] = jt
    return reg
