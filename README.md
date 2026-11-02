# gpuworker

[![ci](https://github.com/triwebb1/gpuworker/actions/workflows/tests.yml/badge.svg)](https://github.com/triwebb1/gpuworker/actions/workflows/tests.yml)

A single-node GPU job broker. Several GPU-bound model services want the same card, and if
each one loads its model whenever it likes, the card runs out of memory and jobs crash
halfway through. gpuworker owns the GPU and decides what runs on it: clients drop job
bundles into a file-share queue, and the broker admits each job against a VRAM budget,
runs it, and writes back a result. Container job types run in a locked-down sandbox,
because loading a third-party model can run arbitrary code. Host-process job types,
including the quick start below, run with no isolation; beside a container runtime they
run only if the config allows them.

gpuworker is a personal project for one GPU host and clients I run myself. It is not built
or reviewed for public, shared or multi-tenant hosting, and should not be exposed to
clients you don't trust.

Two model modules ship with it: text-to-speech in the voice of a reference recording, with
a quality-check job type, and generation of background music and sound effects from a text
prompt.

## How it works

Each point links to where it lives in the code.

- **Single-writer loop.** [`Scheduler.tick`](gpuworker/scheduler.py) owns all admission
  and ledger state. Worker threads ([`run_oneshot`](gpuworker/oneshot.py),
  [`run_serve_manager`](gpuworker/serve_manager.py)) never touch it; they post a
  [`Completion`](gpuworker/jobs.py) that the loop drains.
- **VRAM-budgeted admission.** Each job type declares its peak VRAM in a `module.toml`,
  and a job is admitted only if it fits the remaining budget in the
  [`Ledger`](gpuworker/ledger.py). A model gets the card to itself unless its type opts
  in to co-residence with a flat (`preallocated`) profile. When a model loads with nothing
  else on the GPU, the broker measures its peak and raises the footprint if the
  declaration was too low. Higher `priority` goes first, then a job whose model is
  already loaded.
- **Reclaim gate.** Before loading a model, the broker waits until NVML shows the previous
  model's memory as free and the ledger agrees, rather than trusting a process exit
  ([`Scheduler._service_reclaim`](gpuworker/scheduler.py)). Memory that the ledger says is
  free but NVML never shows free marks the GPU degraded, which stops VRAM admission until
  it recovers.
- **Warm serve containers.** A `vram` job type can run in `serve` mode: one container
  loads the model once and receives jobs over a unix socket, one at a time. Messages carry
  the container's epoch and job reports their attempt number, so reports from an evicted
  container or a superseded attempt are dropped.
- **Sandbox.** No network, non-root, read-only root filesystem, all capabilities dropped,
  `no-new-privileges`, CPU and process limits, read-only inputs and weights, a scratch
  directory of its own, and a GPU only on the `vram` lane
  ([`launcher.py`](gpuworker/launcher.py)). It keeps downloaded model code off the rest of
  the host; it is not a boundary between users. What it does not cover is listed in
  [SPEC 6.3](SPEC.md#63-trust-assumptions).
- **Retries.** A launch error, OOM-kill, serve container failure or reclaim-gate timeout
  re-queues the job up to `max_retries` times, then quarantines it. The classification is
  by exit code, and a workload can cause some of those codes too; for a container killed
  with exit 137 the result also says whether the runtime recorded an OOM-kill. A job that
  fails on its own is never re-run, and a CUDA out-of-memory failure is retried only if
  something else was using the GPU.
- **GPU faults.** After a CUDA error, or a `vram` job that timed out, was killed or
  crashed, and also on a timer, the broker asks the GPU monitor whether the card still
  answers and reports no hardware fault. A reported fault pauses VRAM admission until the
  broker restarts. A job that outlives its timeout has its worker stopped, and the process
  and container confirmed gone, before its memory counts as free
  ([`oneshot.worker_gone`](gpuworker/oneshot.py)).
- **Terminal results.** While the broker runs, every job it claims ends `done` or
  `failed` with a reason, including every job of a bundle it cannot read
  ([`QueueTree.fail_batch`](gpuworker/share.py)). A bundle interrupted by a broker
  restart runs again, and a result that can never be written moves its input to
  `failed/` with the reason. Each result carries the worker's stdout and stderr, clipped
  at both ends ([SPEC 11](SPEC.md#11-observability-config-and-cli)), and a fingerprint of
  the module `src/` the broker staged.

Input and path checks, the queue directories, sealed weights, the full ordering rules and
the serve protocol are in [SPEC.md](SPEC.md), with the main design decisions.

## Quick start (no GPU needed)

This runs the `echo` module, which loads no model and runs as a host process, through the
queue, scheduler, launcher and result path. Run from the repository root with Python
3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
gpuworker --mock-gpu run-once echo.echo --text "hello" --output out/hello.txt
```

`run-once` submits one job, runs the broker until it finishes and collects the result,
in a temporary directory it removes afterwards (`--keep` keeps it). It ends with:

```
status : done
message: done
output : .../collected/out/hello.txt
   ---
   hello
```

To run the broker and a client as separate processes, start the broker in one terminal:

```bash
gpuworker --config examples/local.toml --mock-gpu serve
```

and submit from another:

```bash
python examples/submit_echo.py ./queues/demo "hello"
```

```
submitted echo-1a2b3c4d-20261031T120000Z-5e6f7a8b.zip to ./queues/demo/inbox
status : done
message: done
output : outputs/echo-1a2b3c4d/out/hello.txt -> 'hello'
```

The bundle is named for its first job id plus a UTC timestamp and a random suffix. The
client only writes a zip into `queues/demo/inbox/` and reads the result from
`queues/demo/outbox/`. `python -m gpuworker` is equivalent to `gpuworker`.

## Development

```bash
pip install -e ".[dev]"          # pytest, ruff, mypy
ruff check . && ruff format --check . && mypy
python -m pytest -q
```

The tests need no GPU or container runtime. Tests that create symlinks skip where the user
cannot create them, as on a default Windows account. CI runs the lint steps plus `bash -n`
on the shell scripts, then the tests and the quick-start command on Ubuntu and Windows for
Python 3.11 to 3.13. A third job, [`ci/sandbox/run.py`](ci/sandbox/run.py), runs the
container path with Docker on a runner with no GPU: it builds a small image with the echo
module and a probe, runs `echo.container`, two `echo.serve` jobs and the probe through the
broker, and checks from inside the container that it has no network, no capabilities, a
read-only root and a non-root user. The `nvml` extra installs NVML bindings; without them
the broker polls `nvidia-smi`.

## On a GPU host

Real container jobs need Linux (or WSL2) with an NVIDIA GPU, the NVIDIA Container
Toolkit, and rootful Podman with CDI (`sudo nvidia-ctk cdi generate
--output=/etc/cdi/nvidia.yaml`) or Docker with `--gpus` support. Rootless Podman needs
user-namespace options the launcher does not pass yet
([SPEC 14](SPEC.md#14-status-and-limits)).

```bash
cp gpuworker.example.toml gpuworker.toml   # set queue_roots and container_runtime
images/build.sh --runtime podman --only echo --pin
gpuworker doctor
gpuworker run-once echo.container --text "hello" --output out/hello.txt
gpuworker run-once echo.serve --text "hello" --output out/hello.txt
gpuworker serve
```

The example config refuses unpinned images. Set `sandbox_image` to the digest or local
id that `images/build.sh --pin` prints, or set `allow_unpinned_images = true` for a first
bring-up on a tag. `gpuworker doctor` checks the runtime, GPU, CDI (including that the
spec names the device jobs ask for), registry, weights, queue roots and image pinning,
and starts the sandbox image with the broker's GPU flags to initialize CUDA in it.
`images/build.sh` exits non-zero without pinning if any module failed to provision. To
build the model modules as well, list them, as in `--only "echo speech audiogen"`, and
stage their weights as each module's README shows.

`gpuworker serve` stops cleanly on SIGTERM, so it can run as a systemd service; give it
time to tear warm containers down and stop running jobs. Relative paths in the config
(`registry_dir = "modules"`, `state_dir = "state"`) resolve against the working
directory, so set it to the checkout. `RestartSec` keeps a config error or a refused lock
from restarting in a tight loop:

```ini
[Service]
WorkingDirectory=/opt/gpuworker
ExecStart=/opt/gpuworker/.venv/bin/gpuworker --config /etc/gpuworker/gpuworker.toml serve --log /var/log/gpuworker/events.jsonl
TimeoutStopSec=120
Restart=on-failure
RestartSec=30
```

## Operating it

- **Start:** `gpuworker serve`, or the unit above. One broker per `state_dir`: a second
  one started by mistake exits with an error naming the PID of the first. At start the
  broker removes the containers a crashed run left, including its own oneshot ones that
  are still running.
- **Stop:** SIGTERM or Ctrl-C. Intake stops, warm containers are torn down, and running
  oneshot jobs get up to 20 s to finish, then are stopped and reported `failed`. A bundle
  still without a result stays in `processing/` and runs again after a restart, once it
  is older than `reclaim_after_s` (two hours by default); at start the broker logs each
  such bundle and when it will run again (`left_in_processing`).
- **Upgrade:** stop the broker, update the code, rebuild and re-pin the image if a module
  changed (`images/build.sh --pin`, then `sandbox_image`), run `gpuworker doctor`, start
  it. The broker reads modules only at start. There is no drain command yet, so stop it
  when the queues are quiet if running jobs should finish.
- **Logs:** `serve --log PATH` appends one JSON event per line (stdout without it). The
  file stays open in append mode, so logrotate's `copytruncate` works on it. A
  `tick_error` event carries its traceback. Each result bundle holds its jobs' worker
  logs under `_gpuworker/`.
- **Status:** the broker rewrites `status_path` every loop, and `gpuworker status` prints
  it: VRAM, the ledger, `degraded` and `degraded_reason`, `gpu_fault`, footprints, inbox
  depths and the last loop error, which the next clean iteration clears. It also says how
  old the file is. It exits 1 while the GPU is degraded, or once the file is more than two
  minutes old: the broker has stopped, or one loop iteration (a large bundle unpacking) is
  taking that long.
- **Degraded:** `degraded: true` means VRAM admission is paused. Queued `vram` jobs wait;
  `cpu` jobs still run. It is set when the reclaim gate times out, when a stopped worker
  did not exit, or when the GPU did not answer a health check twice in a row, and it
  clears by itself once the GPU answers and its memory use is back to the idle baseline
  plus what is loaded. If
  `gpu_fault` is also set, a hardware fault was reported and it stays until the broker
  restarts: check the card (`nvidia-smi -q`, the kernel log), reset or reboot it, then
  restart the broker.

## Modules

A module is a directory `modules/<name>/` with a `module.toml` declaring its job types, a
`provision.sh` that builds its virtualenv in the shared image, and its source in `src/`.
The broker reads only the metadata and never imports module code; restart it after
changing a module. [docs/MODULE_GUIDE.md](docs/MODULE_GUIDE.md) is the contract.

| Module | Job types | |
|---|---|---|
| [echo](modules/echo) | `echo.echo`, `echo.container`, `echo.serve` | Loads no model; one type per launch path, used by the quick start and for bring-up. |
| [speech](modules/speech) | `speech.voxcpm2` (serve), `speech.check` (oneshot) | Text-to-speech in the voice of a reference recording (jobs that pass the same recording get the same voice), and a quality-check job type that transcribes a result and compares it with its text. |
| [audiogen](modules/audiogen) | `audiogen.stableaudio` (oneshot) | Background music, ambience and sound effects from a text prompt, with Stable Audio Open 1.0. |
| [_TEMPLATE](modules/_TEMPLATE) | | A skeleton to copy. |

`speech` and `audiogen` are rewrites of modules I ran on my own GPU host. Each module's
README shows how to stage its weights.

## Scope and limits

- One node and one NVIDIA GPU.
- The queue is a shared directory tree (local disk or a mounted share), not a network
  service. A client needs write access to its `inbox/` and read access to its `outbox/`.
- With `container_runtime = "local"` (no container runtime), only host-process job types
  run, with no isolation. Container job types are refused unless
  `allow_unsandboxed_containers = true`, which runs them as host processes; `vram` job
  types must be container types. With Podman or Docker it is the other way round:
  host-process types are refused unless `allow_host_jobs = true`.
- Known gaps are listed in [SPEC.md section 14](SPEC.md#14-status-and-limits).

## How it was built

Built with AI coding agents, from an architecture and specification I set. I accepted
changes against the test suite and independent reviews. [SPEC 12](SPEC.md#12-key-decisions)
lists the design decisions, and [SPEC 13](SPEC.md#13-notable-failures-and-what-changed)
the failures found running the private broker this code was rewritten from, and what
changed because of them. I built it as the GPU side of a larger private system; this
repository is the broker and two of its model modules. The public history is a single
squashed commit of a private working repository.

## License

MIT. See [LICENSE](LICENSE).
