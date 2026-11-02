"""The CI sandbox probe (ci/sandbox/probe) finds the container's pids limit on either
cgroup layout, and the CI runner keeps its reporting helpers callable. The CI job itself
needs Docker; this checks only the lookup and the runner's source."""

import ast
import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def probe():
    path = os.path.join(ROOT, "ci", "sandbox", "probe", "src", "probe.py")
    spec = importlib.util.spec_from_file_location("ci_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fs(probe, monkeypatch, files):
    files = {k.replace("\\", "/"): v for k, v in files.items()}
    monkeypatch.setattr(probe, "_read", lambda p: files.get(p.replace("\\", "/"), ""))


@pytest.mark.parametrize(
    "files,want",
    [
        # cgroup v2 with a private namespace: the container's cgroup is the mount root.
        ({"/proc/self/cgroup": "0::/", "/sys/fs/cgroup/pids.max": "4096"}, "4096"),
        # cgroup v2 with the host's namespace: the limit is on the named cgroup.
        (
            {
                "/proc/self/cgroup": "0::/system.slice/docker-abc.scope",
                "/sys/fs/cgroup/system.slice/docker-abc.scope/pids.max": "4096",
                "/sys/fs/cgroup/pids.max": "max",
            },
            "4096",
        ),
        # cgroup v1: a pids controller line.
        (
            {
                "/proc/self/cgroup": "12:pids:/docker/abc\n11:memory:/docker/abc",
                "/sys/fs/cgroup/pids/docker/abc/pids.max": "4096",
            },
            "4096",
        ),
        ({"/proc/self/cgroup": "0::/"}, ""),
    ],
)
def test_the_probe_finds_the_pids_limit_on_either_cgroup_layout(probe, monkeypatch, files, want):
    """It read only /sys/fs/cgroup/pids.max, which a runner with another layout lacks."""
    _fs(probe, monkeypatch, files)
    assert probe._pids_max()[0] == want


def test_the_ci_runner_does_not_shadow_its_own_helpers():
    """main() assigned the copytree ignore filter to `skip`, so the "no pids.max visible"
    branch called the filter and the skip was never reported."""
    path = os.path.join(ROOT, "ci", "sandbox", "run.py")
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    helpers = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    shadowed = [
        (fn.name, node.id)
        for fn in tree.body
        if isinstance(fn, ast.FunctionDef)
        for node in ast.walk(fn)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id in helpers
    ]
    assert not shadowed


def test_every_shell_script_is_executable_in_git():
    """images/build.sh and some module scripts were committed without the executable bit,
    so the README's `images/build.sh ...` failed with "Permission denied" on a fresh clone."""
    import shutil
    import subprocess

    if not shutil.which("git") or not os.path.isdir(os.path.join(ROOT, ".git")):
        pytest.skip("needs a git checkout")
    out = subprocess.run(
        ["git", "ls-files", "-s", "--", "*.sh"], cwd=ROOT, capture_output=True, text=True
    ).stdout.splitlines()
    assert out
    assert [line.split()[-1] for line in out if not line.startswith("100755")] == []
