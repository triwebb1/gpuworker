"""The environment checks (gpu/runtime/cdi) are host-dependent; here we test the pure
checks doctor controls: registry, staged weights, sandbox-image pinning."""

from gpuworker import doctor, weights
from gpuworker.config import Config

REGISTRY = """
[module]
name = "example"
version = "0.1.0"

[[job_type]]
id = "example.generate"
resource_class = "vram"
vram_mb = 6000
env = "container"
entrypoint = ["python", "-m", "example.generate"]

[[job_type]]
id = "example.cpu_model"
resource_class = "cpu"
vram_mb = 0
env = "container"
entrypoint = ["python", "-m", "example.cpu_worker"]
weights_required = true

[[job_type]]
id = "example.package"
resource_class = "cpu"
vram_mb = 0
env = "container"
entrypoint = ["python", "-m", "example.package_worker"]
sandbox = "none"
"""


def _levels(cfg):
    return {name: level for name, level, _ in doctor._checks(cfg)}


def _registry(tmp_path):
    d = tmp_path / "modules" / "example"
    d.mkdir(parents=True)
    (d / "module.toml").write_text(REGISTRY)
    return str(tmp_path / "modules")


def test_weights_missing_warns_then_ok(tmp_path):
    cfg = Config(
        registry_dir=_registry(tmp_path),
        weights_dir=str(tmp_path / "w"),
        queue_roots=[],
        container_runtime="local",
        sandbox_image="gpuworker/sandbox:cuda12.8",
        allow_unpinned_images=True,
    )
    lv = _levels(cfg)
    # vram job types with no staged cache -> warn
    assert lv["weights:example.generate"] == doctor.WARN
    # A required CPU-model cache is a hard prelaunch requirement.
    assert lv["weights:example.cpu_model"] == doctor.FAIL
    # example.package is a cpu type that needs no weights (no weights row)
    assert "weights:example.package" not in lv

    # stage a cache + seal it -> ok
    cache = tmp_path / "w" / "example.generate"
    cache.mkdir(parents=True)
    (cache / "model.bin").write_text("x")
    weights.write_manifest(str(cache))
    assert _levels(cfg)["weights:example.generate"] == doctor.OK

    cpu_cache = tmp_path / "w" / "example.cpu_model"
    cpu_cache.mkdir(parents=True)
    (cpu_cache / "model.safetensors").write_text("sealed")
    # Directory alone is not enough for a required cache.
    assert _levels(cfg)["weights:example.cpu_model"] == doctor.FAIL
    weights.write_manifest(str(cpu_cache))
    assert _levels(cfg)["weights:example.cpu_model"] == doctor.OK


def test_sandbox_image_pin_status(tmp_path):
    dev = Config(
        registry_dir="modules",
        weights_dir=str(tmp_path / "w"),
        queue_roots=[],
        container_runtime="local",
        sandbox_image="gpuworker/sandbox:cuda12.8",
        allow_unpinned_images=True,
    )
    assert _levels(dev)["sandbox_image"] == doctor.WARN

    pinned = Config(
        registry_dir="modules",
        weights_dir=str(tmp_path / "w"),
        queue_roots=[],
        container_runtime="local",
        sandbox_image="gpuworker/sandbox:cuda12.8@sha256:" + "a" * 64,
        allow_unpinned_images=False,
    )
    assert _levels(pinned)["sandbox_image"] == doctor.OK


def test_registry_check_passes_in_dev(tmp_path):
    # dev config: the repo registry (echo) resolves cleanly on the dev tag
    cfg = Config(
        registry_dir="modules",
        weights_dir=str(tmp_path / "w"),
        queue_roots=[],
        container_runtime="local",
        sandbox_image="gpuworker/sandbox:cuda12.8",
        allow_unpinned_images=True,
    )
    assert _levels(cfg)["registry"] == doctor.OK


def test_doctor_exit_code_follows_the_worst_check(monkeypatch, capsys):
    # The host probes (GPU, runtime, CDI) depend on the machine, so pin the check results
    # and assert the exit code: any FAIL -> 1; warnings alone -> 0.
    rows = [("gpu", doctor.OK, "ok"), ("runtime", doctor.WARN, "dev tag")]
    monkeypatch.setattr(doctor, "_checks", lambda cfg: rows)
    assert doctor.run(Config()) == 0
    assert "READY" in capsys.readouterr().out
    rows.append(("cdi", doctor.FAIL, "missing"))
    assert doctor.run(Config()) == 1
    assert "NOT READY: 1 failure(s), 1 warning(s)" in capsys.readouterr().out


def test_the_cdi_check_needs_the_device_jobs_request(monkeypatch):
    """Podman asks for nvidia.com/gpu=<index>. A spec with only nvidia.com/gpu=all (as
    WSL2 can generate) passed the old file-exists check and failed every GPU job."""
    listing = {"out": "nvidia.com/gpu=0\nnvidia.com/gpu=GPU-1234\nnvidia.com/gpu=all\n"}
    monkeypatch.setattr(doctor, "_run", lambda cmd, timeout=15.0: (0, listing["out"]))
    assert doctor._cdi_check(0)[0] == doctor.OK
    assert doctor._cdi_check(1)[0] == doctor.FAIL
    listing["out"] = "nvidia.com/gpu=all\n"
    level, detail = doctor._cdi_check(0)
    assert level == doctor.FAIL and "nvidia.com/gpu=0" in detail


def test_the_gpu_probe_starts_the_sandbox_as_the_broker_would(monkeypatch):
    seen = {}

    def run(cmd, timeout=15.0):
        seen["cmd"] = cmd
        return 0, "cuInit=0 devices=1\n"

    monkeypatch.setattr(doctor, "_run", run)
    cfg = Config(container_runtime="podman", gpu_index=1, container_user="1234:1234")
    level, detail = doctor.gpu_probe(cfg)
    assert level == doctor.OK and "devices=1" in detail
    cmd = seen["cmd"]
    assert cmd[:2] == ["podman", "run"]
    assert "nvidia.com/gpu=1" in cmd and "1234:1234" in cmd
    assert "--read-only" in cmd and cmd[cmd.index("--network") + 1] == "none"
    assert cfg.sandbox_image in cmd and "cuInit" in cmd[-1]
    monkeypatch.setattr(doctor, "_run", lambda cmd, timeout=15.0: (1, "cuInit=100 devices=0\n"))
    level, detail = doctor.gpu_probe(cfg)
    assert level == doctor.FAIL and "cuInit=100" in detail


def test_doctor_checks_free_disk_where_the_broker_writes(tmp_path, monkeypatch):
    from collections import namedtuple

    from gpuworker.config import Config

    usage = namedtuple("usage", "total used free")
    free = {"mb": 50 * 1024}
    monkeypatch.setattr(
        doctor.shutil, "disk_usage", lambda p: usage(0, 0, free["mb"] * 1024 * 1024)
    )
    cfg = Config(
        state_dir=str(tmp_path / "state" / "not-yet"),  # checked at its existing parent
        weights_dir=str(tmp_path / "w"),
        queue_roots=[str(tmp_path / "Q")],
    )
    rows = {name: level for name, level, _ in doctor.disk_checks(cfg)}
    assert rows == {
        "disk:state_dir": doctor.OK,
        "disk:weights_dir": doctor.OK,
        f"disk:queue:{tmp_path / 'Q'}": doctor.OK,
    }
    free["mb"] = 2 * 1024
    assert {lvl for _n, lvl, _d in doctor.disk_checks(cfg)} == {doctor.WARN}
    free["mb"] = 100
    assert {lvl for _n, lvl, _d in doctor.disk_checks(cfg)} == {doctor.FAIL}


def test_doctor_says_why_it_skipped_the_gpu_probe_and_names_image_errors(tmp_path, monkeypatch):
    """The sandbox probe vanished without a line when an earlier check failed, and image
    errors were only counted."""
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
    monkeypatch.setattr(doctor, "_run", lambda cmd, timeout=15.0: (1, ""))
    cfg = Config(
        registry_dir=_registry(tmp_path),
        weights_dir=str(tmp_path / "w"),
        queue_roots=[],
        container_runtime="docker",
        sandbox_image="gpuworker/sandbox:cuda12.8",
        allow_unpinned_images=False,
    )
    rows = {name: (level, detail) for name, level, detail in doctor._checks(cfg)}
    assert rows["sandbox_gpu"] == (doctor.WARN, "skipped: fix the failures above first")
    level, detail = rows["image:example.generate"]
    assert level == doctor.FAIL and detail
    assert "image error(s)" in rows["registry"][1]
