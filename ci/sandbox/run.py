"""Run the broker against Docker on a host with no GPU, and check from inside the
container that the sandbox flags took effect.

    python ci/sandbox/run.py

Needs Linux, Docker and an installed gpuworker. It builds ci/sandbox/Dockerfile (the echo
module and the sandbox probe on a plain Python base), starts `gpuworker serve` with
`--mock-gpu` on a registry of those two modules, and submits:

- `probe.sandbox`, which reports its user, capabilities, writable paths and network;
- `echo.container`, the oneshot container path;
- two `echo.serve` jobs in one bundle, the serve path: one warm container runs both.

Before the broker starts, it leaves a running container named as the broker's own oneshot
containers are, which the broker should remove at start.

The broker adds `--gpus` for a `vram` container such as `echo.serve`, which Docker
refuses on a host with no GPU, so the broker reaches Docker through a wrapper that drops
that one flag and passes every other argument through unchanged. Exits non-zero if any
check fails.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from gpuworker.client import collect, submit  # noqa: E402
from gpuworker.launcher import DockerLauncher, job_name_prefix  # noqa: E402
from gpuworker.manifest import JobManifest  # noqa: E402

TAG = "gpuworker/ci-sandbox:latest"
# The --pids-limit the launcher passes (gpuworker/launcher.py).
PIDS_LIMIT = "4096"
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: object = "") -> None:
    results.append((name, bool(ok), str(detail)))
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail != "" else ""))


def skip(name: str, why: str) -> None:
    print(f"SKIP {name}  ({why})")


def job(job_id: str, job_type: str, output: str, text: str = "") -> JobManifest:
    options = {"text": text} if text else {}
    return JobManifest.from_dict(
        {"job_id": job_id, "job_type": job_type, "outputs": [output], "options": options}
    )


def read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read().strip()


def write_wrapper(path: str, docker: str, argv_log: str) -> None:
    """A `docker` that logs each argv as a JSON line, drops `--gpus <value>`, and runs the
    real binary with the rest."""
    script = f"""#!/usr/bin/env bash
set -euo pipefail
python3 -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "$@" >> {shlex.quote(argv_log)}
args=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--gpus" ]; then shift 2; continue; fi
  args+=("$1"); shift
done
exec {shlex.quote(docker)} "${{args[@]}}"
"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(script)
    os.chmod(path, 0o755)


def main() -> int:
    docker = shutil.which("docker")
    if not docker:
        print("docker not found", file=sys.stderr)
        return 2
    subprocess.run(
        [docker, "build", "--provenance=false", "-f", "ci/sandbox/Dockerfile", "-t", TAG, "."],
        cwd=REPO,
        check=True,
    )
    image = subprocess.run(
        [docker, "image", "inspect", "-f", "{{.Id}}", TAG],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    # Short, because a unix socket path under state_dir has a length limit.
    work = tempfile.mkdtemp(prefix="gw-")
    registry = os.path.join(work, "registry")
    ignore = shutil.ignore_patterns("__pycache__")
    shutil.copytree(
        os.path.join(REPO, "modules", "echo"), os.path.join(registry, "echo"), ignore=ignore
    )
    shutil.copytree(
        os.path.join(REPO, "ci", "sandbox", "probe"), os.path.join(registry, "probe"), ignore=ignore
    )
    bindir = os.path.join(work, "bin")
    os.mkdir(bindir)
    argv_log = os.path.join(work, "docker-argv.jsonl")
    write_wrapper(os.path.join(bindir, "docker"), docker, argv_log)
    queue = os.path.join(work, "queues", "ci")
    config = os.path.join(work, "gpuworker.toml")
    with open(config, "w", encoding="utf-8") as fh:
        fh.write(
            f"""container_runtime = "docker"
container_user = "{os.getuid()}:{os.getgid()}"
sandbox_image = "{image}"
registry_dir = "{registry}"
weights_dir = "{os.path.join(work, "weights")}"
state_dir = "{os.path.join(work, "state")}"
queue_roots = ["{queue}"]
status_path = "{os.path.join(work, "status.json")}"
footprints_path = "{os.path.join(work, "footprints.json")}"
"""
        )
    env = dict(
        os.environ,
        PATH=bindir + os.pathsep + os.environ.get("PATH", ""),
        GPUWORKER_SCRATCH_DIR=os.path.join(work, "scratch"),
    )
    # A oneshot container that a crashed run of this broker left running, named with the
    # id the broker will find in its state_dir. The broker should remove it at start.
    state = os.path.join(work, "state")
    os.makedirs(state)
    with open(os.path.join(state, "broker_id"), "w", encoding="utf-8") as fh:
        fh.write("c1c1c1c1\n")
    leftover = job_name_prefix("c1c1c1c1") + "left-0"
    subprocess.run(
        [docker, "run", "-d", "--name", leftover, image, "sleep", "600"],
        check=True,
        capture_output=True,
    )
    runtime = DockerLauncher()
    check("the runtime reports a running container as there", runtime.container_exists(leftover))
    never = "gpuworker-ci-never-created"
    check("and one that never existed as gone", not runtime.container_exists(never))

    events = os.path.join(work, "events.jsonl")
    broker = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "gpuworker",
            "--config",
            config,
            "--mock-gpu",
            "serve",
            "--log",
            events,
        ],
        cwd=REPO,
        env=env,
    )
    dest = os.path.join(work, "collected")
    try:
        fn_probe = submit(queue, [job("probe", "probe.sandbox", "out/probe.json")])
        fn_echo = submit(queue, [job("echo", "echo.container", "out/echo.txt", "from a container")])
        fn_serve = submit(
            queue,
            [
                job("s1", "echo.serve", "out/s1.txt", "first served"),
                job("s2", "echo.serve", "out/s2.txt", "second served"),
            ],
        )
        res = {}
        for fn in (fn_probe, fn_echo, fn_serve):
            res.update({m.job_id: m for m in collect(queue, fn, dest, timeout_s=300)})
    finally:
        broker.send_signal(signal.SIGTERM)
        try:
            broker.wait(90)
        except subprocess.TimeoutExpired:
            broker.kill()

    for jid in ("probe", "echo", "s1", "s2"):
        check(f"{jid} is done", res[jid].status == "done", res[jid].message[:300])

    # ---- the sandbox, seen from inside the probe's container ------------------------------
    facts = (
        json.loads(read(os.path.join(dest, "out", "probe.json")))
        if res["probe"].status == "done"
        else {}
    )
    print(json.dumps(facts, indent=2, sort_keys=True))
    check(
        "runs as the configured non-root user",
        facts.get("uid") == os.getuid() != 0,
        facts.get("uid"),
    )
    check(
        "has no capabilities",
        facts.get("cap_eff") == 0 and facts.get("cap_prm") == 0,
        (facts.get("cap_eff"), facts.get("cap_prm")),
    )
    check("cannot gain privileges", facts.get("no_new_privs") is True)
    check(
        "root filesystem is read-only", facts.get("root_write") == "EROFS", facts.get("root_write")
    )
    check(
        "bundle mount is read-only", facts.get("bundle_write") == "EROFS", facts.get("bundle_write")
    )
    check("/tmp is writable", facts.get("tmp_write") == "", facts.get("tmp_write"))
    check(
        "no network interface but loopback",
        facts.get("net_interfaces") == ["lo"],
        facts.get("net_interfaces"),
    )
    check("an outbound connection fails", facts.get("connect", "") != "", facts.get("connect"))
    # The limit is read from the probe's own cgroup, wherever this runner mounts it; a
    # cgroup layout the probe cannot find is reported, not failed. The argv check below
    # covers the flag itself.
    if facts.get("pids_max"):
        check(
            "the process limit applies inside the container",
            facts["pids_max"] == PIDS_LIMIT,
            f"{facts['pids_max']} at {facts.get('pids_max_path')}",
        )
    else:
        skip("the process limit applies inside the container", "no pids.max visible")
    check("a cpu-lane container sees no GPU", facts.get("nvidia_visible_devices") == "void")

    # ---- the oneshot and serve paths ------------------------------------------------------
    for jid, rel, text in (
        ("echo", "out/echo.txt", "from a container"),
        ("s1", "out/s1.txt", "first served"),
        ("s2", "out/s2.txt", "second served"),
    ):
        path = os.path.join(dest, rel)
        check(f"{jid} wrote its output", os.path.isfile(path) and read(path) == text)
    with open(argv_log, encoding="utf-8") as fh:
        calls = [json.loads(line) for line in fh]
    runs = [a for a in calls if a[:1] == ["run"]]
    serve_runs = [a for a in runs if any(x.startswith("gpuworker-serve-echo.serve") for x in a)]
    check("one warm container ran both serve jobs", len(serve_runs) == 1, len(serve_runs))
    check("the vram-lane container asked for a GPU", all("--gpus" in a for a in serve_runs))
    check(
        "cpu-lane containers did not", not any("--gpus" in a for a in runs if a not in serve_runs)
    )
    check(
        "oneshot containers run without --rm, for their exit state; serve ones with it",
        all("--rm" not in a for a in runs if a not in serve_runs)
        and all("--rm" in a for a in serve_runs),
    )
    check(
        "every container got the process limit",
        all("--pids-limit" in a and a[a.index("--pids-limit") + 1] == PIDS_LIMIT for a in runs),
    )
    check(
        "the broker removed the oneshot container its last run left running",
        not runtime.container_exists(leftover),
    )
    left = subprocess.run(
        [docker, "ps", "-aq", "--filter", "name=gpuworker-"], capture_output=True, text=True
    ).stdout.split()
    check("no container is left after the broker stops", left == [], left)
    with open(events, encoding="utf-8") as fh:
        kinds = {json.loads(line)["event"] for line in fh}
    check("no loop errors in the broker log", "tick_error" not in kinds)

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print(f"(work directory kept: {work})")
        return 1
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
