# Writing a gpuworker module

This is the module author's contract. [SPEC.md](../SPEC.md) explains the design; this
guide is the concrete interface and wins on interface details.

Three modules and a template ship with the repo:

- `modules/echo/`: three working job types (host process, container, serve). It loads no
  model, and its host-process type needs no GPU.
- `modules/speech/` and `modules/audiogen/`: real models, a serve type and two oneshot
  types: text-to-speech with a quality check, and music and sound-effect generation
  (section 8.2).
- `modules/_TEMPLATE/`: a skeleton to copy.

Terms: a **module** is a directory under `modules/<name>/` declaring one or more **job
types** (such as `echo.echo`); a **client** submits jobs and collects results; the
**broker** is the `gpuworker serve` process that owns the GPU.

---

## 1. The mental model

### 1.1 The broker owns the GPU

A client submits jobs, a module declares how to run them, and the broker decides what is
resident, when to load and when to evict. So **your worker must be willing to let go of
the GPU.** Do not build "load once and stay warm forever" into it. The broker keeps a
model warm while its queue has work, and for `keep_warm_s` after, then evicts it so
another model can run.

### 1.2 Two layers: the module and the work package

- **Module**: registered once, declarative. It tells the broker which job types exist and
  what each costs. The broker reads it without importing your code or any model library.
- **Work package**: a zip bundle holding one or more JSON job manifests plus input files.
  Each manifest names a job type. The resource footprint comes from the module, not the
  job.

### 1.3 Resource lanes

| Lane | Contends for | Admission rule |
|---|---|---|
| `vram` | CUDA compute and VRAM | Resident effective `vram_mb` plus yours must fit the admission budget, and the card must show enough free memory (the reclaim gate, SPEC 9). |
| `cpu` | Host CPU and RAM | `cpu_slots` (default 2). Runs beside a resident `vram` model. |

The admission budget is `gpu_budget_mb - headroom_mb`. A `vram` job whose effective
footprint can never fit is failed with a message rather than queued forever. A `cpu`
container gets no GPU.

### 1.4 Job lifecycle

A manifest moves through `queued`, `running`, then `done` or `failed`. Only the last two
are terminal, and every job reaches one with a message, including every job in a bundle
that cannot be unpacked. The broker's JSON-lines event log records the steps:

```
claimed          bundle renamed inbox/ -> processing/, manifests read, jobs queued
reclaim_pending  (vram only) admitted, parked until NVML shows enough free VRAM
running          worker launched (oneshot) or job handed to the warm container (serve)
retry            the broker's infrastructure failed; the job is re-queued (section 3.6)
quarantined      the job failed on infrastructure on every allowed attempt
done | failed    terminal, with a message
batch_done       every job in the bundle is terminal; result bundle written to outbox/
```

**Claim is not start.** The broker claims bundles greedily, then holds each job until its
lane and the VRAM budget allow it. To time a job, print timestamps from the worker and
read them from its worker log (section 6.1), not from claim-to-done.

### 1.5 Is the broker running your code? Two fingerprints

A container job type's source is copied into the shared image at build time, so
**updating the repository does not change what a job executes** until the image is
rebuilt and the broker resolves the new image.

Every worker log (`_gpuworker/<job_id>.log`, section 6.1) starts with:

```
# gpuworker worker log job_id=<job_id> job_type=<job_type>
# runtime module: MATCHES staged (<first 16 hex>...)
# module: echo v0.1.0 src_sha256=<64 hex> files=2 newest=<UTC time>
```

- **`module:`** is the staged source: the module's `src/` directory on the broker host,
  hashed once when the registry is scanned at broker start. It says what the broker's
  checkout contains, not what the container ran. A module with no `src/` reports
  `(no src/)`.
- **`runtime module:`** compares that with the digest your worker reports, if it reports
  one: `MATCHES staged`, `*** MISMATCH ***` (with both digests),
  `src_sha256=<hex> (broker staged no src/)`, or `not reported by this worker`.

The two can disagree: with the image pinned by digest, a rebuild and retag is invisible to
the broker, and jobs keep running old code while `module:` reports the new source.

**To opt in**, print your digest from inside the worker:

```
GPUWORKER_RUNTIME_MODULE src_sha256=<64 hex> files=<n>
```

The broker searches the job's stdout and stderr for `src_sha256` followed by 64 hex
characters, so `src_sha256=<hex>` or a JSON `"src_sha256": "<hex>"` also match. Oneshot
workers print it once per run. Serve workers print it inside each job's handler, because a
serve job's log is the slice of the container log written while that job ran, and a line
printed during warmup lands in no job's log.

**Compute it the way the broker does** (`source_fingerprint` in `gpuworker/module.py`):
sha256 over the relative path, a NUL byte and the content of every file under `src/`, in a
sorted walk, skipping `__pycache__` directories and `*.pyc` / `*.pyo` files. An unreadable
file changes the digest instead of being skipped. In the shared image your source is at
`/opt/modules/<name>/src`:

```python
import hashlib, os


def src_fingerprint(src="/opt/modules/<name>/src"):
    h, n = hashlib.sha256(), 0
    for root, dirs, files in os.walk(src):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for fn in sorted(files):
            if fn.endswith((".pyc", ".pyo")):
                continue
            full = os.path.join(root, fn)
            h.update(os.path.relpath(full, src).replace(os.sep, "/").encode() + b"\0")
            try:
                with open(full, "rb") as fh:
                    h.update(fh.read())
            except OSError:
                h.update(b"<unreadable>")
            n += 1
    return h.hexdigest(), n


digest, n = src_fingerprint()
print(f"GPUWORKER_RUNTIME_MODULE src_sha256={digest} files={n}", flush=True)
```

Build the image from a clean checkout: untracked files under `src/` change the digest. The
`files=` count tells the cases apart: same count with a different digest is a content
difference; different counts is a file-set difference.

### 1.6 Broker limits: read them, do not hard-code them

At start the broker writes `broker_limits.json` into the root of every queue tree:

```json
{
  "default_timeout_s": 1800,
  "generated_at": "<ISO-8601 UTC time>",
  "job_type_timeouts_s": {"echo.serve": 30.0},
  "reclaim_after_s": 7200.0,
  "schema": "gpuworker-broker-limits/v1"
}
```

- **`job_type_timeouts_s`** gives every type's bound when a manifest sets no `timeout_s`,
  in the mode the broker actually uses.
- **Why it matters:** a bundle whose summed timeouts exceed `reclaim_after_s` can outlive
  the stale-reclaim window and, after a broker restart mid-batch, be reclaimed while still
  valid work. Sum the timeouts before you submit, and split large batches.
- `schema` is a string. Warn on an unrecognized one rather than failing; if the file is
  absent, fall back to your own values and say so.

---

## 2. The module manifest (`module.toml`)

The broker scans its `registry_dir` (default `modules`) for `<registry_dir>/*/module.toml`
and for `*.toml` directly inside it. `modules/_TEMPLATE/` ships `module.toml.example`, so
it is never registered. The registry is read once when the broker starts; **restart the
broker after changing a `module.toml`**.

### 2.1 Full reference

```toml
[module]
name    = "mymodel"
version = "0.1.0"                   # optional; shown in the worker log

[[job_type]]
id                    = "mymodel.generate"
resource_class        = "vram"               # vram | cpu
vram_mb               = 9000
vram_profile          = "growing"            # growing | preallocated
co_resident           = false
env                   = "container"          # required: container | host
# image               = "sha256:<64 hex>"    # omit to inherit the shared sandbox image
entrypoint            = ["/opt/venvs/mymodel/bin/python", "-m", "mymodel.worker"]
modes                 = ["serve"]            # serve and/or oneshot
serve_socket          = "/run/gpuworker/job.sock"
serve_job_timeout     = 900
serve_max_jobs        = 0
oneshot_job_timeout_s = 0
load_cost_s           = 60
keep_warm_s           = 120
weights_required      = true
fixed_env             = { MYMODEL_SOURCE_REV = "abc1234" }
env_passthrough       = ["MYMODEL_THREADS"]
sandbox               = "standard"           # none | standard | strict
```

An unknown key fails the registry load with its name, so a misspelled `serve_max_job`
is caught instead of silently keeping the default. The table below lists every key.

### 2.2 Field by field

| Field | Type | Default | Enforced at registry load |
|---|---|---|---|
| `id` | string | required | Contains a dot. Unique across modules. Convention `<module>.<job>`. |
| `resource_class` | string | `"vram"` | `vram` or `cpu`. |
| `vram_mb` | int | `0` | Non-negative; `> 0` on the `vram` lane. Use `0` elsewhere. |
| `vram_profile` | string | `"growing"` | `growing` or `preallocated`. |
| `co_resident` | bool | `false` | |
| `env` | string | required | `container` or `host`; `vram` needs `container`. |
| `image` | string | `""` | On a container env, must be pinned: `repo@sha256:<64 hex>` or `sha256:<64 hex>`. |
| `entrypoint` | list of strings | required | Non-empty. |
| `modes` | list of strings | `["oneshot"]` | Non-empty subset of `serve`, `oneshot`. `serve` requires the `vram` lane and `env = "container"`. |
| `serve_socket` | string | `""` | Required for `serve` on a container env. |
| `serve_job_timeout` | number (s) | `300` | Finite and greater than 0. |
| `serve_max_jobs` | int | `0` | Non-negative; non-zero requires `serve`. |
| `oneshot_job_timeout_s` | number (s) | `0` | Finite, non-negative; non-zero requires `oneshot`. |
| `load_cost_s` | number (s) | `0` | Finite, non-negative. |
| `keep_warm_s` | number (s) | `0` | Finite, non-negative. |
| `weights_required` | bool | `false` | Always true for `strict` types. |
| `fixed_env` | table of strings | `{}` | Names match `[A-Z_][A-Z0-9_]*`; no reserved names. |
| `env_passthrough` | list of strings | `[]` | Same name rules; no reserved names. |
| `sandbox` | string | `"standard"` | `none`, `standard` or `strict`. `vram` needs at least `standard`; `host` cannot be `strict`. |

What the fields mean:

- **`vram_profile` and `co_resident`.** `preallocated` means the model allocates about its
  full footprint at load and stays flat; `growing` (the default) means VRAM climbs during
  the job. Co-residence needs both `co_resident = true` and `preallocated`. Any other
  `vram` type gets the card to itself. Set `co_resident` only for a small helper that
  should stay loaded while larger models swap, and give it a long `keep_warm_s`.
- **`env`.** `container` runs in a Linux container from the shared image; use it for
  anything that loads a model, which the `vram` lane requires. `host` runs a plain host
  subprocess with **no isolation**, for trusted code only (`echo.echo` uses it so the
  quick start needs no runtime). A broker with a container runtime refuses `host` types
  unless its config sets `allow_host_jobs = true`.
- **`image`**: almost always omit it to inherit `sandbox_image` (section 2.3).
- **`modes`**: `oneshot` for a fresh process per job, `serve` for one warm process
  handling many jobs (section 3.2). If both are declared, serve is used. Serve is rejected
  at load unless the type is on the `vram` lane with `env = "container"` (the worker is
  reached over a socket mounted into its container).
- **`serve_socket`**: the container-side path your serve worker binds, passed as
  `GPUWORKER_SERVE_SOCKET`.
- **Timeouts.** `serve_job_timeout` and `oneshot_job_timeout_s` are the longest a job of
  the type may run (`0` for the oneshot key inherits `default_timeout_s`). A manifest's
  `timeout_s` can lower that bound for one job, never raise it. Set them from your real
  worst case plus margin; the watchdog arms from the same value.
- **`serve_max_jobs`**: recycle the serve container after this many jobs (section 3.2).
- **`load_cost_s` / `keep_warm_s`.** Among cold `vram` types of equal priority the broker
  prefers the largest (queued jobs x `load_cost_s`). `keep_warm_s` is how long a serve
  container stays after its queue empties; oneshot types release their slot on exit and
  usually set it to 0.
- **`fixed_env`**: constant, non-sensitive values for the worker, such as the upstream
  revision an adapter was built against.
- **`env_passthrough`**: names copied from the **broker's** environment into the job
  container, if set and non-empty there, so a setting can vary per broker run
  (`MYMODEL_THREADS=4 gpuworker serve`). Container job types only. Not for credentials:
  the containers run offline and need none.

Broker-reserved names (`_RESERVED_FIXED_ENV` in `gpuworker/module.py`), rejected in both
`fixed_env` and `env_passthrough`:

```
GPUWORKER_JOB_ID  GPUWORKER_BUNDLE  GPUWORKER_OUT  GPUWORKER_SERVE_SOCKET
GPUWORKER_CPU_BUDGET  GPUWORKER_MEM_BUDGET_MB  GPUWORKER_SCRATCH
HF_HOME  HF_HUB_OFFLINE  TRANSFORMERS_OFFLINE
NVIDIA_DRIVER_CAPABILITIES  NVIDIA_VISIBLE_DEVICES
```

In a container, `fixed_env` is applied first, the broker's other defaults (such as
`TMPDIR` and `PYTHONUNBUFFERED`) fill the remaining names, and `env_passthrough` fills only
what is still unset. When enabled, the CUDA cache path is applied after that.

### 2.3 Your module lives in `modules/<name>/`

There is one shared sandbox image. Every container job type runs from it, isolated from
other modules by its own venv. You contribute a venv, not an image.

```
modules/<name>/
  module.toml      # your job type declarations
  provision.sh     # builds /opt/venvs/<name> for ./src at image build time
  src/             # worker-side source
  smoke/           # optional smoke scripts per job type (see modules/echo/smoke/)
```

1. Copy `modules/_TEMPLATE/` to `modules/<name>/` and rename `module.toml.example`.
2. Put your worker source in `src/`.
3. Fill in `provision.sh`: create `/opt/venvs/<name>`, install your dependencies (a CUDA
   build of torch first if you need it), and put `./src` on the venv's path with a `.pth`
   file rather than pip-installing it, so jobs run the files the runtime fingerprint
   hashes (section 1.5). End by importing your entrypoint module, so a broken install
   fails at build time.
4. In `module.toml`, omit `image` and point `entrypoint` at `/opt/venvs/<name>/bin/python`.

The build copies `modules/` to `/opt/modules/` and runs each `provision.sh` as root
(skipping names that start with `_` or `.`); it is online and needs no GPU. A failing
provision script does not stop the build, so one build reports every failure, but
`images/build.sh` then exits non-zero and does not re-pin the image.

```bash
images/build.sh --runtime docker --tag gpuworker/sandbox:cuda12.8 --only "echo <name>"
images/build.sh --runtime docker --tag gpuworker/sandbox:cuda12.8 --pin   # print a digest
```

`sandbox_image` defaults to a tag, which is refused unless `allow_unpinned_images = true`
(development only; `strict` types refuse it regardless). For real use, pin it to
`repo@sha256:<digest>` or to the local image id `sha256:<64 hex>`, valid only on the
daemon that built it.

**Never commit weights into `src/`.** Weights are staged separately on the broker host
(section 3.3). The broker never imports your code; everything above is metadata.

---

## 3. The execution contract

### 3.1 `oneshot`: one process per job

The broker unpacks the bundle and mounts it read-only with a read-write output directory,
runs your entrypoint once with `GPUWORKER_JOB_ID` set, waits for exit within the job's
timeout, collects declared outputs and tears the container down. The model load is paid
on every job.

Your entrypoint:

1. Reads `GPUWORKER_JOB_ID`.
2. Loads its manifest from `$GPUWORKER_BUNDLE/jobs/pending/<job_id>.json`. The bundle
   directory holds the whole unpacked batch; inputs are at `$GPUWORKER_BUNDLE/<relpath>`.
   The broker wrote that manifest file itself from the one it validated, so it holds only
   the fields of section 6, with the broker's values (a capped `priority` or `timeout_s`,
   `status` reset to `queued`).
3. Does the work using `manifest["options"]`, which the broker passes through untouched.
4. Writes each path in `manifest["outputs"]` to `$GPUWORKER_OUT/<relpath>`, as a regular
   file. `$GPUWORKER_OUT` is a fresh, empty directory for every attempt.
5. Exits `0` on success. To fail, exit non-zero with a message on stderr; the job's
   message carries the last 500 characters of stderr. Exit `75` when CUDA ran out of
   memory and `76` after any other CUDA error. The broker may retry either in a fresh
   process (section 3.6).
6. Writes only under `$GPUWORKER_OUT`, even though the broker has already refused any
   output path that is not a plain relative path: `modules/echo/src/worker.py` resolves
   each output and refuses one that would land outside.

`modules/echo/src/worker.py` is exactly this contract.

Output rules: after a zero exit the broker copies the declared outputs that exist into the
result. A declared output that is a symlink, or has a symlink anywhere on its path, or is
not a regular file, or is larger than the broker's `max_output_mb` (default 8192), fails
the job with `refused output: ...`. The broker keeps only the start and the end of your
stdout and stderr ([SPEC 11](../SPEC.md#11-observability-config-and-cli) gives the
limits). If any declared output is missing, the job fails with
`worker exited 0 but did not produce ...` naming the missing ones, even if others exist.
Declare only the outputs your worker always writes.

#### Environment the broker provides

| Variable | Container jobs | `host` jobs |
|---|---|---|
| `GPUWORKER_JOB_ID` | Oneshot only (serve gets the id in each `run` frame) | Oneshot only |
| `GPUWORKER_BUNDLE` | `/job/bundle` (read-only) | Host path of the unpacked bundle |
| `GPUWORKER_OUT` | `/job/out` | Host path of this attempt's output directory |
| `GPUWORKER_SERVE_SOCKET` | `serve_socket`, for serve types | Not set |
| `GPUWORKER_CPU_BUDGET` | Host CPUs minus a spare (broker env `GPUWORKER_CPU_SPARE`, default 2), minimum 1; also the `--cpus` limit | Not set |
| `GPUWORKER_MEM_BUDGET_MB` | System RAM minus a reserve (broker env `GPUWORKER_MEM_SPARE_MB`, default 3072), minimum 512; advisory | Not set |
| `GPUWORKER_SCRATCH` | `/job/scratch` when scratch is enabled | Not set |
| `TMPDIR` | `/job/scratch/tmp` when scratch is enabled | Not set (`TEMP` and `TMP` pass through) |
| `HF_HOME`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` | When a weight cache exists (`HF_HOME=/job/weights`) | Not set |
| `PYTHONUNBUFFERED=1` | Set | Not set |
| `NVIDIA_DRIVER_CAPABILITIES=compute,utility`, `NVIDIA_VISIBLE_DEVICES=<gpu_index>` | `vram` lane; a `cpu` container gets `NVIDIA_VISIBLE_DEVICES=void` and no GPU | Not set |
| `fixed_env` entries | Set | Set |
| `env_passthrough` entries | Set if present in the broker's environment | Not set |

Size thread pools to `GPUWORKER_CPU_BUDGET`, not `nproc` or `os.cpu_count()`, which still
report the full host count under the `--cpus` limit. If you fan out memory-heavy
subprocesses, bound the count by RAM too:
`workers = max(1, min(cpu_budget, mem_budget_mb // per_task_peak_mb))`.

A `host` job does not inherit the broker's environment, which may hold credentials; it
gets a short allowlist (`PATH`, `HOME`, `LANG`, `LC_ALL`, `TEMP`, `TMP`, Windows system
variables) plus the variables marked "Set". Read these variables rather than hard-coding
`/job/...` so one worker runs under both launchers.

### 3.2 `serve`: one warm process, many jobs

For models that are expensive to load. Model your worker on
`modules/echo/src/serve_worker.py`, which has no dependencies; the wire format is defined
in `gpuworker/serve.py` and summarized in SPEC 8.

**Transport.** A unix socket at `serve_socket`. Each frame is a 4-byte big-endian length
and a UTF-8 JSON object, at most 8 MiB. Frames carry control data only; files travel
through the mounted directories.

```
worker -> broker   {"type": "ready", "protocol": 1}
                   {"type": "result", "job_id": "...", "status": "done" | "failed",
                    "outputs": [...], "message": "...",
                    "error": "cuda_oom" | "cuda_error"}      # error: optional, failed only
broker -> worker   {"type": "run", "job_id": "...", "manifest": {...}}
                   {"type": "unload"}
```

**Connection.** The worker listens. The broker starts the container, connects (retrying
for up to `serve_connect_timeout_s`, default 60 s), then waits for one `ready` frame (up
to `serve_ready_timeout_s`, default 600 s, for the whole frame, however slowly it
arrives). `protocol` must equal the broker's (currently 1). The scheduler also bounds the
load from outside: a container that is not ready within those two timeouts plus 30 s is
removed, and the jobs waiting on it each spend a retry.

**What your serve worker must do:**

- **Bind early, then load, then report ready.** Bind and `listen()` on
  `$GPUWORKER_SERVE_SOCKET` as soon as the process starts (remove a stale socket file
  first), so the short connect timeout is met. Then load the model, check it works
  (section 3.4), `accept()` the connection and send exactly one `ready` frame. If loading
  fails, exit non-zero: the broker fails the queued jobs with `serve load failed` plus the
  tail of the container log. That counts as an infrastructure failure, so the jobs are
  retried once on a fresh container (section 3.6); a load that always fails is paid twice,
  then the jobs are quarantined.
- **Handle `run` frames one at a time**, replying with one `result` frame carrying the
  same `job_id`.
- **Namespace per-job I/O by the frame's `job_id`.** One container serves many batches
  under fixed mounts. The frame's `job_id` is a key the broker chooses for this attempt
  (its own id for the job plus the attempt number), not the client's id, which is
  `manifest["job_id"]`: two bundles can reuse a client id, and neither may see the other's
  files. Read inputs from `$GPUWORKER_BUNDLE/<frame job_id>/<relpath>` and write declared
  outputs to `$GPUWORKER_OUT/<frame job_id>/<relpath>`. The result is built from the files
  found there; the `outputs` list in your frame is informational. A `done` result missing
  any declared output on disk becomes `failed`. Use `manifest["job_id"]` when you need
  the client's id, for example in a sidecar.
- **Stay up across a bad job.** Catch the exception and reply `status: "failed"` with a
  message. A crash is treated as the container dying: it takes every queued job with it.
- **Name a CUDA failure.** If the job failed because CUDA ran out of memory, free what it
  allocated (drop its tensors, `torch.cuda.empty_cache()`), add `"error": "cuda_oom"` to
  the result and stay up: the broker keeps the container warm and may retry the job on it
  (section 3.6). For any other CUDA error, after which the process's CUDA context cannot
  be trusted, add `"error": "cuda_error"` and then exit non-zero. The broker replaces the
  container instead of sending the next job to a process that would fail it too, and may
  retry the job on the new one. Check the exceptions a failure was raised from too, since
  libraries wrap CUDA errors in their own; `modules/speech/src/speech/cuda_errors.py` is
  one way to classify them. `outputs` must be a list of strings and `message` a string; a
  result that breaks the format is a protocol error.
- **Exit cleanly on `unload` or EOF.** Release the model (`del` references,
  `torch.cuda.empty_cache()`) and exit 0. The broker then stops the container and does
  not admit the next `vram` job until the card shows the memory free (SPEC 9), so a prompt
  exit keeps that wait short.

Each serve job is bounded by `serve_job_timeout`, or the manifest's `timeout_s` if that
is shorter. A
timeout fails the job without retry. A timeout, a protocol error (including a `result` for
the wrong `job_id`) or a dead socket tears the container down, and every job queued behind
it is retried on a new container. So does a job that outlives the broker's watchdog for
any other reason. A staging or collect error fails only that job.

The container's stdout and stderr go to one log file; each job's worker log is the slice
written while it ran, and the first job after the container starts also gets what it
printed while it loaded the model. Between jobs the broker empties that file once it passes 64 MB;
the earlier jobs' slices are already in their worker logs.

#### Batching

Serve mode pays the model load once per warm period. The container stays for
`keep_warm_s` after its last job, and can be evicted sooner when another job needs the
card. Best is one bundle with many manifests of the same serve type; otherwise submit
back to back within `keep_warm_s`. Interleaving an unrelated heavy job can evict the model
and force a reload.

#### Recycling a warm container

- **`serve_max_jobs = N`** recycles the container after every N jobs.
- **`options.reset_serve = true`** in a manifest recycles it before that job. It takes
  effect only if a job has run since the last load, so a fresh container is never
  recycled needlessly. Use it from a client that detects degraded output and wants a clean
  process.

Both send `unload`, stop the container, wait until the card shows the type's footprint
plus `headroom_mb` free (bounded by `reclaim_timeout_s`), relaunch, and wait for `ready`.
The type keeps its ledger slot throughout, so no other model is admitted into the gap; the
log shows `serve_respawn`. If the recycle fails, that job and the queued ones fail as
infrastructure failures. A recycle costs one full model load, so set `serve_max_jobs` from
a measured need.

### 3.3 Weights: staged, sealed, offline

Containers run with `--network none`, so a model can only load what is on disk.

1. Put the job type's files in `<weights_dir>/<job_type id>/` on the broker host. It is
   mounted read-only at `/job/weights` with `HF_HOME` pointing there, so a Hugging Face
   cache layout works as-is.
2. Seal it: `gpuworker stage-weights mymodel.generate` hashes every file into
   `weights.sha256`, skipping downloader logs, locks, `.locks/` and `.no_exist/`.
3. Check it with `gpuworker verify-weights mymodel.generate`.

If a seal exists, the broker re-hashes the whole cache before every oneshot launch and
each serve container start, and fails the job on any missing, extra or changed file. For
a large cache this adds time to every launch; serve mode pays it once per container. With
`weights_required = true`, a missing directory or seal also fails the job. Without it, a
type with no cache runs without the weights mount and offline variables.

**Stage every file the code path touches.** Anything your library downloads lazily
(`hf_hub_download`, `requests.get` and the like) fails with no network; stage it or
install it in `provision.sh`. If a library
writes next to its cache (transformers copies remote-code modules into `HF_MODULES_CACHE`,
under `HF_HOME` by default), point that at `/tmp` with `fixed_env`.

### 3.4 Fail fast at warmup

- **Probe each critical component at warmup** on a small input, check the result, and
  raise with a named message before sending `ready`. A worker that exits before `ready`
  fails quickly with your message and the container log tail.
- **Check post-conditions yourself.** Many libraries catch their own initialization error
  and return `None` or an empty object. Never swallow a load error that later code
  tolerates.
- **A bad job is not a broken component.** Malformed input gets a `failed` result and the
  process stays up; a broken shared component means raise at warmup and exit.

### 3.5 Writable paths inside a container

The root filesystem is read-only, including `/opt` and the home directory; writes there
fail with `Errno 30 Read-only file system`.

| Path | Backing | Use |
|---|---|---|
| `/tmp` | RAM tmpfs, mounted `exec`, `tmpfs_mb` (default 2048) | Small caches, JIT-compiled kernels loaded with `dlopen`, sockets |
| `$GPUWORKER_SCRATCH` (`/job/scratch`) | Host directory on local disk, when enabled | Many or large intermediate files |
| `$GPUWORKER_OUT` | A fresh directory per attempt (oneshot) or a per-job subdirectory (serve) | Declared outputs only |
| `/cuda-cache` | Host directory, only when the broker sets `GPUWORKER_CUDA_CACHE_DIR` | CUDA and TorchInductor JIT caches across containers |

- `/tmp` competes with model loading for system RAM; do not use it for bulk data.
- Scratch is a directory of the container's own, under the broker's scratch root
  (default `~/.cache/gpuworker-scratch`; `GPUWORKER_SCRATCH_DIR` moves it, or `""` turns
  it off, and then `GPUWORKER_SCRATCH` and the `TMPDIR` redirect are absent). It is empty
  when the container starts and deleted after it exits, so no other job sees it. A serve
  container keeps one scratch directory across all its jobs: namespace per-job files
  under `<job_id>/` and delete them when the job ends.
- A job's output directory is copied out file by file after the job. Keep high-churn
  intermediate files in scratch.
- Point library caches that default to `$HOME` at `/tmp` or scratch with `fixed_env` (for
  example `XDG_CACHE_HOME`).

### 3.6 Failures and retries

The broker retries a job only when the failure looks like **its own side**, up to the
broker's `max_retries` (default 1; `0` disables; there is no per-type setting). The
classification goes by exit code and event, and some of those codes can also come from
the workload: a job that allocates host memory until it is killed exits 137 like any other
OOM-kill. After the last allowed attempt the job is quarantined: it fails with a
`[quarantined: ...]` note and is never re-queued.

Classified as infrastructure, retried:

- The launcher raised before a worker existed (for example the runtime binary is missing).
- A oneshot worker exited 137 or -9: killed from outside the job, usually an OOM-kill. For
  a container the result says whether the runtime recorded one.
- A oneshot worker exited 75, or a serve result carried `"error": "cuda_oom"`: CUDA ran
  out of memory. Retried only if the job did not have the GPU to itself, that is another
  job used the GPU during the attempt or memory outside the ledger was in use when it
  started. With the card to itself the job would fail the same way again, so it fails
  with a note to ask for less in its options or raise `vram_mb`. A serve job is retried
  on the same warm container. The broker logs a `cuda_oom` event with the peak it
  measured, but does not record that peak as the type's footprint: at an OOM it is what
  the card had, not what the model needs.
- A oneshot worker exited 76, or a serve result carried `"error": "cuda_error"`: another
  CUDA error. The broker first asks the GPU monitor whether the card still answers and
  reports no hardware fault (with NVML: new uncorrected ECC errors, a failed or pending
  row remap, or a critical Xid that NVIDIA lists as a hardware fault). If the monitor
  reports a fault, or reports the GPU lost, the job fails without a retry and the GPU is
  marked degraded until the broker restarts. A monitor that does not answer twice in a row
  marks the GPU degraded only until it answers. Otherwise the retry runs in a fresh
  process. The broker runs the same check when a
  `vram` job times out, is stopped by the watchdog, is killed or dies on a signal.
- A container job's runtime exited 125: it could not create the container.
- A serve container failed to start (connect or `ready` timeout, the scheduler's load
  deadline, exit before `ready`, protocol mismatch), failed to recycle, or died mid-job
  (broken socket, malformed frame, a `result` for the wrong job).
- A job was queued behind a serve container that died, timed out, hung, or was stopped.
- The reclaim gate timed out.

Job failures, never retried:

- Any other non-zero exit, including a model that fails to load in a oneshot run.
- A `failed` result, missing declared outputs, a timeout in either mode, a watchdog
  failure.
- A weight verification or image resolution failure (oneshot or serve), and a serve
  staging or collect error.
- Anything that fails while the broker is shutting down.

What this means for your module:

- **A non-zero exit is final.** Do not count on a retry for flaky code.
- **Do not crash a serve process on bad input.** A crash is an infrastructure failure: it
  takes the queued jobs with it and spends their retries.
- **A retry starts clean.** A retried oneshot job gets a new, empty `$GPUWORKER_OUT`, so
  it can only succeed on files it writes itself. Files outside it (scratch, caches) are
  not reset.
- A retry re-runs the same manifest and options and is logged as a `retry` event. Jobs
  queued in a serve container that died on another job are re-queued without spending
  their retry.

---

## 4. Sandboxing: declare the real risk

Every container job gets the same controls whatever its `sandbox` level: non-root, no
network, read-only root filesystem, no capabilities, limited mounts (SPEC 6.1). The
declared level is your statement of what the job's code does, enforced as registry and
launch policy:

- **`none`**: no model code and no deserialization, for example a job that only packages
  files. Not allowed on the `vram` lane.
- **`standard`**: safe loaders only: `safetensors`, `from_pretrained`, no `pickle`, no
  `trust_remote_code`, no `torch.load(weights_only=False)`. The minimum for `vram`.
- **`strict`**: the job executes third-party code or unpickles checkpoints. Treat it as
  arbitrary code execution. It must run in a container and refuses an unpinned image even
  with `allow_unpinned_images = true`, and `weights_required` is forced on. Pair it with
  a lockfile install in `provision.sh`.

A `host` job type has none of the container controls. The trust assumptions are in
SPEC 6.3: the registry is trusted, so whoever can write to `modules/` controls what runs.

---

## 5. Certifying your footprint (`vram_mb`)

A guessed `vram_mb` makes the scheduler over-commit or under-use the card. Measure it:

1. Run the heaviest realistic job alone on the card.
2. Sample peak VRAM (`nvidia-smi --query-gpu=memory.used --format=csv -l 1`, or
   `torch.cuda.max_memory_allocated()` plus CUDA context and allocator slack).
3. Declare the true peak. Do not pad it; the broker applies `headroom_mb` globally.
4. Measure again when the model, dtype, batch size or input limits change.

**The broker corrects upward.** When a type runs as the only resident, the broker
measures its peak, after every job for a serve type. If that exceeds `vram_mb`, the
effective footprint ratchets up and is
persisted in `footprints.json`, keyed by job type and image so a rebuilt image starts
fresh, and an overshoot beyond `footprint_variance_tol` (default 10%) logs
`footprint_variance`, a signal to measure again. The broker never uses a number lower
than yours.

---

## 6. The work package (per-job manifest)

A bundle is a zip of paths relative to the bundle root. Manifests sit at
`jobs/pending/<job_id>.json`, named for the `job_id` they hold; inputs at their own paths.
Every member name must be a plain relative path (the rule below), and no two may differ
only in case. `jobs/pending/` may hold nothing but manifests. The broker validates the
manifests from the unpacked files, the same ones a worker reads.

```json
{
  "job_id": "batch42_item3",
  "job_type": "mymodel.generate",
  "priority": 0,
  "inputs": ["inputs/batch42/item3.json"],
  "outputs": ["out/batch42_item3.bin", "out/batch42_item3.meta.json"],
  "options": {"steps": 40, "seed": 7341},
  "deliver_to": null,
  "timeout_s": 0
}
```

| Field | Rules |
|---|---|
| `job_id` | Matches `[A-Za-z0-9][A-Za-z0-9._-]{0,127}`; unique within the bundle. It becomes a host path component (output subdirectories, log files). |
| `job_type` | A registered job type id. An unknown id fails the batch. |
| `priority` | Integer in [-1000, 1000], default 0, higher first (section 6.2). |
| `inputs` | Bundle-relative paths. Informational; the files must be packed into the bundle. |
| `outputs` | Relative paths the worker writes. Only these are collected, and all of them must exist for the job to be `done`. They may not overlap another output in the same bundle, or sit under `jobs/` or `_gpuworker/`. At most 10,000 per job, and no more across a bundle than the broker's `max_bundle_members`. |
| `options` | Object passed to the worker untouched. The broker reads only `reset_serve` (section 3.2). |
| `deliver_to` | String or null; client-side delivery path (section 6.1). |
| `timeout_s` | Integer 0 to 10,000,000. 0 uses the type's bound; a larger value is lowered to it. |
| `schema` | Optional integer, default 1. A newer schema is rejected. |

Every path in `inputs` and `outputs`, and every bundle member name, must be a plain
relative path: forward slashes only, no empty, `.` or `..` part, no part ending in a dot
or a space, and none of the characters `\ : * ? " < > |` or a control character. On a
Windows broker, device names such as `NUL` are refused too. Anything else fails the
batch. These are spellings that some reader would take for a different path, or for the
same file as another path; refusing them means each path names one file. A worker that
takes file paths in `options` should apply the same rule (`speech.jobio.job_path` does).

Put model knobs in `options`, and leave `timeout_s` at 0 unless one job needs a different
bound. Status fields are reset by the broker on intake.

Submitting from Python:

```python
from gpuworker.client import collect, submit
from gpuworker.manifest import JobManifest

q = "/srv/gpuworker/queues/default"
job = JobManifest.from_dict(
    {
        "job_id": "item3",
        "job_type": "mymodel.generate",
        "inputs": ["in/item3.json"],
        "outputs": ["out/item3.bin"],
    }
)
fn = submit(q, [job], inputs_root=".", extra_members=["in/item3.json"])
manifests = collect(q, fn, dest_root="results", timeout_s=3600)
```

`submit` packs the manifests plus `extra_members` (relative to `inputs_root`) into
`<first job_id>-<UTC stamp>-<8 hex>.zip`, writes it as `.tmp` and renames it into
`inbox/`, and returns that filename. The unique name means resubmitting a `job_id` never
collides with an older bundle or its result.

### 6.1 Getting your files back (collect)

When every job in a bundle is terminal, the broker writes `outbox/<bundle name>`: the
terminal manifests, every declared output that exists, and each job's worker log. The
input moves to `archive/`. If the outputs cannot be packed, the result carries the
manifests only and those jobs are `failed` with the reason; if nothing can be written, the
broker keeps the batch and retries (SPEC 10).

- **Worker side:** write every file you want back under `GPUWORKER_OUT` (or
  `GPUWORKER_OUT/<frame job_id>/` in serve mode) and declare each in `outputs`, sidecars
  included. Undeclared files are discarded.
- **Client side:** `collect(queue_root, filename, dest_root)` waits for the terminal
  result, extracts outputs and worker logs into `dest_root`, and returns the manifests.
  `wait_result(queue_root, filename)` is the status-only variant. Both raise
  `TimeoutError` if no result arrives.
- **`deliver_to`**: `collect()` copies the job's **first** output there, atomically and
  idempotently, and records `result["delivered"]`. Use an absolute path. `collect()`
  writes wherever the manifest says, so only collect bundles whose manifests you or a
  trusted producer wrote.

**The worker log** is `_gpuworker/<job_id>.log`, present on every outcome including a
timeout kill. It starts with the fingerprint header (section 1.5). Each stream is clipped
at **both ends** ([SPEC 11](../SPEC.md#11-observability-config-and-cli) gives the
limits): on a chained traceback the head holds
the original cause, while the manifest's `message` holds only the end of stderr. It never
appears in `result["outputs"]` and cannot satisfy the declared-outputs check; do not write
your own files under `_gpuworker/`.

### 6.2 Priority

`priority` (higher first, default 0) is the primary ordering key across every queue. A
higher-priority job for another model can get the loaded model evicted once it is idle;
a running job is never killed. Within a priority the broker prefers the loaded model,
then the most queued work times `load_cost_s`, then the order the broker took the jobs
in (SPEC 5). Leave routine work at 0. A broker can cap priority per queue
(`max_priority`).

---

## 7. Registration and testing

1. Create `modules/<name>/` with `module.toml`, `provision.sh` and `src/` (section 2.3).
2. For container types, rebuild the shared image and point `sandbox_image` at it.
3. Validate the registry. Any rule violation in section 2.2 is reported as
   `registry error`; each container type's image is checked too:

   ```bash
   gpuworker registry validate
   gpuworker registry list
   ```

4. Run one job end to end. `run-once` runs one job (id `runonce`) through a temporary
   queue, prints the result and tears down any serve container. It uses its own
   container prefix and private state files, so it is safe beside a live broker:

   ```bash
   gpuworker --mock-gpu run-once echo.echo --text hello --output out/hello.txt
   ```

   Flags: `--option key=value`, `--bundle <dir>` (its files become inputs), `--output
   <relpath>`, `--timeout <s>` (default 300). Global flags (`--config`, `--mock-gpu`) go
   before the subcommand.
5. Add your clients' queue roots to `queue_roots` in the broker config, then restart the
   broker (`gpuworker serve`) so it reads the new module.

`gpuworker doctor` checks the host: runtime, GPU and CDI, registry, weights, queue roots
and image pinning.

---

## 8. Examples

### 8.1 The echo module

| Job type | Lane | Env | Mode | What it exercises |
|---|---|---|---|---|
| `echo.echo` | `cpu` | `host` | oneshot | Queue, scheduler and result path with the local launcher. No GPU or runtime. |
| `echo.container` | `cpu` | `container` | oneshot | Container launch: mounts, environment, output collection. |
| `echo.serve` | `vram` (10 MB, `preallocated`) | `container` | serve | Warm-container lifecycle: socket, `ready`, streamed jobs, keep-warm, eviction. |

`src/worker.py` (oneshot) and `src/serve_worker.py` (serve) are the reference workers.
Both print to stdout and stderr, so the worker log is never empty.

### 8.2 The model modules

| Job type | Lane | Mode | Sandbox | What it shows |
|---|---|---|---|---|
| `speech.voxcpm2` | `vram`, `growing` | serve | `strict` | Text-to-speech in the voice of a reference recording. A model that is slow to load, kept warm; bind before load, a probe generation before `ready`, per-job I/O under the frame's `job_id` with input paths confined to it, CUDA failures named in the result, and the runtime fingerprint printed per job. |
| `speech.check` | `vram`, `growing` | oneshot | `standard` | The quality check: a second model kept out of the serve container so it does not inflate that type's footprint, and exit codes that separate a failed check (exit 0, record written) from bad input (2) and CUDA running out of memory (75). |
| `audiogen.stableaudio` | `vram`, `growing` | oneshot | `standard` | Music and sound effects from a prompt. A post-load probe and output checks that turn a silently broken pipeline into a named failure. |

Their workers' logic is tested without a GPU in `tests/test_example_modules.py`, against
fake models. Each module's README covers staging its weights.

### 8.3 Illustrative job types

Each passes `registry validate` under a `[module]` header. `echo.serve` already shows a
small `preallocated` serve type.

**A large model whose memory grows with input, recycled periodically:**

```toml
[[job_type]]
id                = "bigmodel.generate"
resource_class    = "vram"
vram_mb           = 14000             # longest supported input: measured peak
vram_profile      = "growing"
env               = "container"
entrypoint        = ["/opt/venvs/bigmodel/bin/python", "-m", "bigmodel.serve_worker"]
modes             = ["serve"]
serve_socket      = "/run/gpuworker/job.sock"
serve_job_timeout = 1800
serve_max_jobs    = 20
load_cost_s       = 120
keep_warm_s       = 180
weights_required  = true
env_passthrough   = ["BIGMODEL_SAMPLER_STEPS"]
sandbox           = "standard"
```

**A loader that runs third-party code** (`strict`: container, pinned image, sealed
weights):

```toml
[[job_type]]
id                    = "remotecode.infer"
resource_class        = "vram"
vram_mb               = 6000
env                   = "container"   # inherits sandbox_image, which must be pinned
entrypoint            = ["/opt/venvs/remotecode/bin/python", "-m", "remotecode.worker"]
modes                 = ["oneshot"]
oneshot_job_timeout_s = 600
load_cost_s           = 40
weights_required      = true
fixed_env             = { HF_MODULES_CACHE = "/tmp/hf_modules" }
sandbox               = "strict"
```

---

## 9. Author checklist

- [ ] The right lane, a `vram_mb` declaration (measured if you can, section 5; the broker
      raises it when a solo load measures higher) and an honest `vram_profile`.
- [ ] A `container` env and the matching `sandbox` level for anything that loads a model.
- [ ] The execution contract for every declared mode; serve I/O under the frame's `job_id`, and a
      serve process that survives a failed job.
- [ ] Warmup that raises on a broken component (section 3.4).
- [ ] Every needed file staged and sealed, with `weights_required = true` (section 3.3).
- [ ] Outputs safe to overwrite on a retry (section 3.6); scratch cleaned up.
- [ ] Fan-out sized to `GPUWORKER_CPU_BUDGET` and `GPUWORKER_MEM_BUDGET_MB`.
- [ ] Measured timeouts, with each bundle's sum under `reclaim_after_s` (section 1.6).
- [ ] The runtime fingerprint printed (section 1.5).
- [ ] `registry validate` and a `run-once` smoke test pass for each job type.
