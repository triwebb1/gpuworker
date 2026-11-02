"""Broker CLI / service entrypoint (SPEC 11).

gpuworker serve            run the broker loop
gpuworker doctor           bring-up preflight for the GPU host
gpuworker run-once ID      run one synthetic job of a job type end-to-end
gpuworker status           print the running broker's status.json
gpuworker registry list    list known job types
gpuworker registry validate validate all modules (exit 1 on error)
gpuworker stage-weights ID  hash (seal) a job type's staged weight cache
gpuworker verify-weights ID verify a staged weight cache
gpuworker footprints reset ID  forget a job type's learned VRAM footprint
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import os
import shutil
import signal
import sys

from . import gpu, weights
from .config import Config
from .footprints import Footprints
from .launcher import make_launcher
from .lock import BrokerRunning
from .module import scan_registry
from .obs import JobLogger, write_status
from .scheduler import Scheduler

# Serve containers started by `run-once` get their own name prefix (see cmd_run_once).
RUN_ONCE_SERVE_PREFIX = "gpuworker-runonce-serve-"
# The broker rewrites status_path every loop iteration, and waits at most
# TICK_ERROR_BACKOFF_MAX_S between iterations; an older file means it is not running, or
# one iteration (a large bundle unpacking) is taking that long.
STATUS_STALE_S = 120.0


def _load_config(path: str | None) -> Config:
    if path:
        # An explicit --config that does not exist is an error, never a silent fallback
        # to defaults (which would point the broker at the wrong queues and registry).
        if not os.path.isfile(path):
            raise SystemExit(f"gpuworker: config file not found: {path}")
        return Config.load(path)
    if os.path.isfile("gpuworker.toml"):
        return Config.load("gpuworker.toml")
    return Config()


def install_stop_handlers(sched: Scheduler) -> None:
    """Stop the loop cleanly on SIGTERM (systemd stop, `docker stop`) and SIGINT. Without
    this, SIGTERM kills the process before Scheduler.run reaches shutdown(), and warm serve
    containers keep holding VRAM after the broker is gone."""

    def handler(signum, frame):
        sched.stop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, handler)


def cmd_serve(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    reg = scan_registry(cfg.registry_dir)
    launcher = make_launcher(cfg.container_runtime, cfg.gpu_index)
    monitor = gpu.MockGpu() if args.mock_gpu else gpu.make_monitor(cfg.gpu_index)
    logger = JobLogger.to_path(args.log) if args.log else JobLogger.to_stdout()
    sched = Scheduler(cfg, reg, launcher, monitor, logger=logger)
    print(
        f"gpuworker: {len(reg.job_types)} job types, "
        f"budget {cfg.admit_budget_mb()} MB, runtime {cfg.container_runtime}",
        file=sys.stderr,
    )
    install_stop_handlers(sched)
    try:
        sched.run(interval_s=args.interval)  # returns after stop(), once shutdown() is done
    except BrokerRunning as exc:
        print(f"gpuworker: {exc}", file=sys.stderr)
        return 1
    finally:
        logger.close()
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    from . import doctor

    cfg = _load_config(args.config)
    print(
        f"gpuworker doctor - runtime={cfg.container_runtime}, sandbox_image={cfg.sandbox_image}\n"
    )
    return doctor.run(cfg)


def cmd_run_once(args: argparse.Namespace) -> int:
    """Submit one synthetic job of a job type, run the broker until it finishes, and
    print the result. The end-to-end smoke test / per-module bring-up driver: proves
    the real launcher + container + collect path in one command. Its temporary directory
    is removed at the end unless ``--keep`` is given."""
    import tempfile

    workdir = tempfile.mkdtemp(prefix="gpuw-runonce-")
    try:
        return _run_once(args, workdir)
    finally:
        if args.keep:
            print(f"(kept {workdir})", file=sys.stderr)
        else:
            shutil.rmtree(workdir, ignore_errors=True)


def _run_once(args: argparse.Namespace, workdir: str) -> int:
    import time as _time

    from .client import collect, submit
    from .manifest import JobManifest
    from .scheduler import Scheduler
    from .share import QueueTree

    cfg = _load_config(args.config)
    reg = scan_registry(cfg.registry_dir)
    reg.get(args.job_type)  # validate it exists
    monitor = gpu.MockGpu() if args.mock_gpu else gpu.make_monitor(cfg.gpu_index)
    launcher = make_launcher(cfg.container_runtime, cfg.gpu_index)

    queue_root = os.path.join(workdir, "queue")
    # A smoke test may run on a host where the real broker is serving. Keep its state
    # files private: read the learned footprints, but write to a copy, never the broker's.
    fp_copy = os.path.join(workdir, "footprints.json")
    if os.path.isfile(cfg.footprints_path):
        shutil.copy2(cfg.footprints_path, fp_copy)
    # allow_host_jobs guards a broker from its clients' manifests; this one the operator
    # wrote on the command line, so a host job type such as echo.echo may run.
    cfg = dataclasses.replace(
        cfg,
        footprints_path=fp_copy,
        status_path=os.path.join(workdir, "status.json"),
        state_dir=os.path.join(workdir, "state"),
        allow_host_jobs=True,
    )
    out_rels = args.output or [f"out/{args.job_type.replace('.', '_')}.out"]
    opts = {}
    for kv in args.option or []:
        k, _, v = kv.partition("=")
        opts[k] = v
    if args.text:
        opts["text"] = args.text

    members = []
    inputs_root = args.bundle or workdir
    if args.bundle:
        for dp, _dirs, files in os.walk(args.bundle):
            for f in files:
                members.append(os.path.relpath(os.path.join(dp, f), args.bundle).replace("\\", "/"))

    manifest = JobManifest.from_dict(
        {
            "job_id": "runonce",
            "job_type": args.job_type,
            "outputs": out_rels,
            "options": opts,
            "inputs": members,
            "timeout_s": args.timeout,
        }
    )
    fn = submit(queue_root, [manifest], inputs_root=inputs_root, extra_members=members)
    print(
        f"run-once: {args.job_type}  runtime={cfg.container_runtime}  queue={queue_root}",
        file=sys.stderr,
    )

    sched = Scheduler(
        cfg,
        reg,
        launcher,
        monitor,
        logger=JobLogger.to_stdout(),
        queues=[QueueTree(queue_root).ensure()],
        serve_name_prefix=RUN_ONCE_SERVE_PREFIX,
    )
    # run-once bypasses Scheduler.run()'s startup reap, so clear serve containers a prior
    # crashed run-once left holding VRAM (its own prefix only: a live broker's warm
    # containers are named gpuworker-serve-* and are left alone).
    launcher.reap_serve_orphans(RUN_ONCE_SERVE_PREFIX)
    out_zip = os.path.join(queue_root, "outbox", fn)
    # A cold serve job's first run also pays the one-time model load (up to
    # serve_ready_timeout_s) before the job itself runs, so allow that window too.
    serve_grace = cfg.serve_ready_timeout_s if reg.get(args.job_type).supports("serve") else 0
    deadline = _time.monotonic() + args.timeout + serve_grace + 30
    while _time.monotonic() < deadline:
        sched.tick()
        if os.path.exists(out_zip):
            sched.tick()
            break
        _time.sleep(0.1)

    dest = os.path.join(workdir, "collected")
    try:
        mans = collect(queue_root, fn, dest, timeout_s=10)
    except Exception as exc:
        print(f"FAILED: no terminal result ({exc})", file=sys.stderr)
        return 1
    finally:
        # Serve mode leaves a warm container; tear it down so run-once never orphans one.
        # shutdown() joins the manager thread + drains completions (no full ticks).
        sched.shutdown()
    m = mans[0]
    print(f"\nstatus : {m.status}\nmessage: {m.message}")
    outs = (m.result or {}).get("outputs") if m.result else None
    if outs:
        for o in outs:
            p = os.path.join(dest, o)
            print(f"output : {p}")
            if os.path.isfile(p) and os.path.getsize(p) < 2000:
                try:
                    with open(p, encoding="utf-8") as fh:
                        body = fh.read().strip().replace("\n", "\n   ")
                    print("   ---\n   " + body)
                except Exception:
                    pass
    if args.keep:
        print(f"\n(collected under {dest})", file=sys.stderr)
    return 0 if m.status == "done" else 1


def cmd_status(args: argparse.Namespace) -> int:
    """Print the status file the running broker refreshes every loop. With --snapshot,
    build one from the config instead (GPU, registry and queues, but no live ledger) and
    write it to that path, which must not be the broker's own status_path."""
    cfg = _load_config(args.config)
    if not args.snapshot:
        if not cfg.status_path or not os.path.isfile(cfg.status_path):
            print(
                f"gpuworker: no status file at {cfg.status_path or '(status_path unset)'}; "
                "is the broker running? (--snapshot PATH writes one without it)",
                file=sys.stderr,
            )
            return 1
        with open(cfg.status_path, encoding="utf-8") as fh:
            text = fh.read()
        print(text)
        rc = 0
        age = _status_age(text)
        if age is not None:
            print(f"gpuworker: status written {age:.0f}s ago", file=sys.stderr)
            if age > STATUS_STALE_S:
                print(
                    f"gpuworker: the status file is older than {STATUS_STALE_S:.0f}s; "
                    "the broker may have stopped",
                    file=sys.stderr,
                )
                rc = 1
        degraded = _status_degraded(text)
        if degraded is not None:
            print(f"gpuworker: the GPU is degraded: {degraded}", file=sys.stderr)
            rc = 1
        return rc
    if cfg.status_path and os.path.abspath(args.snapshot) == os.path.abspath(cfg.status_path):
        print(
            f"gpuworker: {args.snapshot} is the broker's status_path; pick another path",
            file=sys.stderr,
        )
        return 1
    reg = scan_registry(cfg.registry_dir)
    monitor = gpu.MockGpu() if args.mock_gpu else gpu.make_monitor(cfg.gpu_index)
    sched = Scheduler(cfg, reg, make_launcher("local"), monitor)
    write_status(args.snapshot, sched)
    with open(args.snapshot, encoding="utf-8") as fh:
        print(fh.read())
    return 0


def _status_age(text: str) -> float | None:
    """Seconds since the status file's ``ts``, or None if it has none."""
    try:
        ts = datetime.datetime.fromisoformat(json.loads(text)["ts"])
    except (ValueError, KeyError, TypeError):
        return None
    if ts.tzinfo is None:
        return None
    return (datetime.datetime.now(datetime.UTC) - ts).total_seconds()


def _status_degraded(text: str) -> str | None:
    """Why the status file says the GPU is degraded, or None if it does not."""
    try:
        ledger = json.loads(text)["ledger"]
        if not ledger.get("degraded"):
            return None
        return str(ledger.get("degraded_reason") or ledger.get("gpu_fault") or "no reason given")
    except (ValueError, KeyError, TypeError, AttributeError):
        return None


def cmd_registry(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    try:
        reg = scan_registry(cfg.registry_dir)
    except Exception as exc:
        print(f"registry error: {exc}", file=sys.stderr)
        return 1
    if args.action == "validate":
        from . import module as _mod

        problems = []
        for _jt_id, jt in sorted(reg.job_types.items()):
            level, msg = _mod.image_status(jt, cfg.sandbox_image, cfg.allow_unpinned_images)
            if level == "warn":
                print(f"  warn: {msg}")
            elif level == "error":
                problems.append(msg)
        for p in problems:
            print(f"  ERROR: {p}", file=sys.stderr)
        if problems:
            print(f"FAILED: {len(problems)} image issue(s)", file=sys.stderr)
            return 1
        print(
            f"ok: {len(reg.job_types)} job types in {len(reg.modules)} modules; "
            f"sandbox_image={cfg.sandbox_image}"
        )
        return 0
    for jt_id, jt in sorted(reg.job_types.items()):
        print(
            json.dumps(
                {
                    "id": jt_id,
                    "lane": jt.resource_class,
                    "vram_mb": jt.vram_mb,
                    "env": jt.env,
                    "modes": list(jt.modes),
                    "sandbox": jt.sandbox,
                    "co_resident": jt.co_resident,
                }
            )
        )
    return 0


def cmd_stage_weights(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    cache = os.path.join(cfg.weights_dir, args.job_type)
    if not os.path.isdir(cache):
        print(f"no cache dir at {cache}; stage weights there first", file=sys.stderr)
        return 1
    path = weights.write_manifest(cache)
    print(f"wrote {path}")
    return 0


def cmd_verify_weights(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    cache = os.path.join(cfg.weights_dir, args.job_type)
    ok, problems = weights.verify(cache)
    if ok:
        print(f"ok: {cache} verified")
        return 0
    for p in problems:
        print(f"  {p}", file=sys.stderr)
    print(f"FAILED: {cache}", file=sys.stderr)
    return 1


def cmd_footprints(args: argparse.Namespace) -> int:
    """Forget a learned footprint, so admission falls back to the declared vram_mb. For an
    observation that was wrong (another process grew during the measured run) and now
    keeps the type waiting for VRAM it does not need. Stop the broker first: a running
    broker keeps its own copy and would write the old value back."""
    cfg = _load_config(args.config)
    fp = Footprints(cfg.footprints_path)
    if fp.reset(args.job_type):
        print(f"reset {args.job_type} in {cfg.footprints_path}")
    else:
        print(f"no learned footprint for {args.job_type} in {cfg.footprints_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gpuworker")
    p.add_argument("--config", help="path to gpuworker.toml")
    p.add_argument("--mock-gpu", action="store_true", help="use an in-memory GPU")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the broker loop")
    s.add_argument("--interval", type=float, default=1.0)
    s.add_argument("--log", help="path to the JSON-lines job log (default: stdout)")
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("doctor", help="bring-up preflight: check the GPU host is ready")
    s.set_defaults(func=cmd_doctor)

    s = sub.add_parser("run-once", help="run ONE job of a job type end-to-end (smoke test)")
    s.add_argument("job_type")
    s.add_argument("--text", help="echo text (for echo job types)")
    s.add_argument(
        "--output",
        action="append",
        help="declared output relpath (repeatable; default out/<job_type>.out)",
    )
    s.add_argument("--option", action="append", help="job option k=v (repeatable)")
    s.add_argument("--bundle", help="dir whose files become the job's input bundle")
    s.add_argument("--timeout", type=int, default=300)
    s.add_argument(
        "--keep", action="store_true", help="keep the temporary directory and the outputs"
    )
    s.set_defaults(func=cmd_run_once)

    s = sub.add_parser("status", help="print the running broker's status.json")
    s.add_argument(
        "--snapshot",
        metavar="PATH",
        help="write a snapshot from the config to PATH instead (no broker needed)",
    )
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("registry", help="list/validate modules")
    s.add_argument("action", choices=["list", "validate"])
    s.set_defaults(func=cmd_registry)

    s = sub.add_parser("stage-weights", help="hash a staged weight cache")
    s.add_argument("job_type")
    s.set_defaults(func=cmd_stage_weights)

    s = sub.add_parser("verify-weights", help="verify a staged weight cache")
    s.add_argument("job_type")
    s.set_defaults(func=cmd_verify_weights)

    s = sub.add_parser("footprints", help="manage learned VRAM footprints")
    s.add_argument("action", choices=["reset"])
    s.add_argument("job_type")
    s.set_defaults(func=cmd_footprints)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
