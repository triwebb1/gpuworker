"""Sandbox probe: a oneshot worker that records what its container can and cannot do,
as JSON in its declared output. Standard library only. ci/sandbox/run.py asserts on it.
"""

import errno
import json
import os
import socket
import sys


def _status() -> dict[str, str]:
    fields = {}
    with open("/proc/self/status", encoding="utf-8") as fh:
        for line in fh:
            key, _, value = line.partition(":")
            fields[key] = value.strip()
    return fields


def _write_error(path: str) -> str:
    """The errno name a write to *path* fails with, or "" if it succeeded."""
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("probe\n")
    except OSError as exc:
        return errno.errorcode.get(exc.errno or 0, str(exc))
    os.unlink(path)
    return ""


def _connect_error() -> str:
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=3).close()
    except OSError as exc:
        return errno.errorcode.get(exc.errno or 0, type(exc).__name__)
    return ""


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def _pids_max() -> tuple[str, str]:
    """(value, path) of this process's cgroup pids limit, or ("", "") if none is found.
    The file is under the cgroup named in /proc/self/cgroup: cgroup v2 has one hierarchy
    ("0::<path>"), v1 a "pids" controller line; with a private cgroup namespace the path
    is "/", and the mount root is tried as well."""
    candidates = []
    for line in _read("/proc/self/cgroup").splitlines():
        _, controllers, path = line.split(":", 2)
        rel = path.lstrip("/")
        if controllers == "":
            candidates.append(os.path.join("/sys/fs/cgroup", rel, "pids.max"))
        elif "pids" in controllers.split(","):
            candidates.append(os.path.join("/sys/fs/cgroup/pids", rel, "pids.max"))
    candidates += ["/sys/fs/cgroup/pids.max", "/sys/fs/cgroup/pids/pids.max"]
    for path in candidates:
        value = _read(path)
        if value:
            return value, path
    return "", ""


def main() -> int:
    job_id = os.environ.get("GPUWORKER_JOB_ID", "")
    bundle_dir = os.environ.get("GPUWORKER_BUNDLE", "/job/bundle")
    out_dir = os.environ.get("GPUWORKER_OUT", "/job/out")
    with open(
        os.path.join(bundle_dir, "jobs", "pending", f"{job_id}.json"), encoding="utf-8"
    ) as fh:
        outputs = json.load(fh).get("outputs") or []
    status = _status()
    pids_max, pids_max_path = _pids_max()
    facts = {
        "uid": os.getuid(),
        "gid": os.getgid(),
        "cap_eff": int(status.get("CapEff", "0"), 16),
        "cap_prm": int(status.get("CapPrm", "0"), 16),
        "no_new_privs": status.get("NoNewPrivs") == "1",
        "root_write": _write_error("/gpuworker-probe"),
        "bundle_write": _write_error(os.path.join(bundle_dir, "gpuworker-probe")),
        "tmp_write": _write_error("/tmp/gpuworker-probe"),
        "net_interfaces": sorted(os.listdir("/sys/class/net")),
        "connect": _connect_error(),
        "pids_max": pids_max,
        "pids_max_path": pids_max_path,
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES", ""),
    }
    print(json.dumps(facts, sort_keys=True))
    for rel in outputs:
        dest = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(facts, fh, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
