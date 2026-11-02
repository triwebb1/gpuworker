import textwrap

import pytest

from gpuworker import module


def _write(tmp_path, body):
    p = tmp_path / "module.toml"
    p.write_text(textwrap.dedent(body))
    return str(p)


GOOD = """
[module]
name = "example"
version = "0.1.0"

[[job_type]]
id = "example.model"
resource_class = "vram"
vram_mb = 5000
env = "container"
image = "example/model:cuda12.8@sha256:%s"
entrypoint = ["python","-m","example.worker"]
modes = ["serve","oneshot"]
co_resident = true
sandbox = "standard"
serve_socket = "/run/job.sock"
""" % ("a" * 64)


def test_load_good_module(tmp_path):
    mod = module.load_module(_write(tmp_path, GOOD))
    assert mod.name == "example"
    jt = mod.job_types[0]
    assert jt.resource_class == "vram" and jt.vram_mb == 5000
    assert jt.is_container and jt.supports("serve")


def test_rejects_floating_image_tag(tmp_path):
    # A module that pins its own image must use a digest, not a tag.
    body = GOOD.replace("@sha256:" + "a" * 64, ":latest")
    with pytest.raises(module.ModuleError, match="digest"):
        module.load_module(_write(tmp_path, body))


def test_image_optional_inherits_shared(tmp_path):
    # Omitting image is allowed (inherit the shared sandbox).
    body = GOOD.replace('image = "example/model:cuda12.8@sha256:%s"' % ("a" * 64), 'image = ""')
    mod = module.load_module(_write(tmp_path, body))
    assert mod.job_types[0].image == ""


def test_resolve_image_inherits_and_gates(tmp_path):
    body = GOOD.replace('image = "example/model:cuda12.8@sha256:%s"' % ("a" * 64), 'image = ""')
    jt = module.load_module(_write(tmp_path, body)).job_types[0]  # standard sandbox
    pinned = "gpuworker/sandbox:cuda12.8@sha256:" + "c" * 64
    tag = "gpuworker/sandbox:cuda12.8"
    # pinned shared image always resolves
    assert module.resolve_image(jt, pinned, allow_unpinned=False) == pinned
    # unpinned tag needs the dev flag
    with pytest.raises(module.ModuleError, match="unpinned"):
        module.resolve_image(jt, tag, allow_unpinned=False)
    assert module.resolve_image(jt, tag, allow_unpinned=True) == tag


def test_resolve_image_strict_refuses_unpinned(tmp_path):
    body = GOOD.replace('sandbox = "standard"', 'sandbox = "strict"').replace(
        'image = "example/model:cuda12.8@sha256:%s"' % ("a" * 64), 'image = ""'
    )
    jt = module.load_module(_write(tmp_path, body)).job_types[0]
    tag = "gpuworker/sandbox:cuda12.8"
    # even with the dev flag, a strict job type refuses a mutable tag at launch
    with pytest.raises(module.ModuleError, match="strict"):
        module.resolve_image(jt, tag, allow_unpinned=True)


def test_local_image_id_counts_as_pinned(tmp_path):
    # A bare local image id (sha256:<64hex>) is immutable content-addressing, so it
    # counts as pinned: a single-host worker can run strict without a registry digest.
    local_id = "sha256:" + "e" * 64
    assert module.is_pinned(local_id)
    assert module.is_pinned("repo/name@sha256:" + "f" * 64)
    assert not module.is_pinned("gpuworker/sandbox:cuda12.8")
    body = GOOD.replace('sandbox = "standard"', 'sandbox = "strict"').replace(
        'image = "example/model:cuda12.8@sha256:%s"' % ("a" * 64), 'image = ""'
    )
    jt = module.load_module(_write(tmp_path, body)).job_types[0]  # strict
    # strict resolves against a local-id shared image (no registry needed)
    assert module.resolve_image(jt, local_id, allow_unpinned=False) == local_id
    assert module.image_status(jt, local_id, False)[0] == "ok"


def test_image_status_classifies_for_validate(tmp_path):
    tag = "gpuworker/sandbox:cuda12.8"
    pinned = "gpuworker/sandbox:cuda12.8@sha256:" + "d" * 64

    def jt_with(sandbox):
        body = GOOD.replace('sandbox = "standard"', f'sandbox = "{sandbox}"').replace(
            'image = "example/model:cuda12.8@sha256:%s"' % ("a" * 64), 'image = ""'
        )
        return module.load_module(_write(tmp_path, body)).job_types[0]

    std = jt_with("standard")
    strict = jt_with("strict")
    # pinned shared image -> ok for everyone
    assert module.image_status(std, pinned, False)[0] == "ok"
    assert module.image_status(strict, pinned, False)[0] == "ok"
    # dev (allow_unpinned): standard warns, strict warns (not-ready, won't run yet)
    assert module.image_status(std, tag, True)[0] == "warn"
    assert module.image_status(strict, tag, True)[0] == "warn"
    # prod (no flag): both are errors
    assert module.image_status(std, tag, False)[0] == "error"
    assert module.image_status(strict, tag, False)[0] == "error"


def test_vram_lane_cannot_be_sandbox_none(tmp_path):
    body = GOOD.replace('sandbox = "standard"', 'sandbox = "none"')
    with pytest.raises(module.ModuleError, match="standard"):
        module.load_module(_write(tmp_path, body))


def test_vram_lane_requires_positive_vram(tmp_path):
    body = GOOD.replace("vram_mb = 5000", "vram_mb = 0")
    with pytest.raises(module.ModuleError, match="vram_mb"):
        module.load_module(_write(tmp_path, body))


def test_serve_container_requires_socket(tmp_path):
    body = GOOD.replace('serve_socket = "/run/job.sock"', 'serve_socket = ""')
    with pytest.raises(module.ModuleError, match="serve_socket"):
        module.load_module(_write(tmp_path, body))


def test_cpu_none_is_allowed_and_unknown_lanes_are_not(tmp_path):
    body = """
    [module]
    name = "example"
    [[job_type]]
    id = "example.package"
    resource_class = "cpu"
    vram_mb = 0
    env = "container"
    image = "example/package@sha256:%s"
    entrypoint = ["python","-m","example.package_worker"]
    modes = ["oneshot"]
    sandbox = "none"
    """ % ("b" * 64)
    mod = module.load_module(_write(tmp_path, body))
    assert mod.job_types[0].resource_class == "cpu"
    with pytest.raises(module.ModuleError, match="resource_class"):
        module.load_module(_write(tmp_path, body.replace('"cpu"', '"gpu"')))


def test_vram_lane_requires_a_container(tmp_path):
    # Model code belongs in a container (SPEC 12.1); host stays available off the GPU.
    body = GOOD.replace('modes = ["serve","oneshot"]', 'modes = ["oneshot"]').replace(
        'env = "container"', 'env = "host"'
    )
    body = body.replace('image = "example/model:cuda12.8@sha256:' + "a" * 64 + '"', "")
    with pytest.raises(module.ModuleError, match='vram-lane job types need env = "container"'):
        module.load_module(_write(tmp_path, body))


def test_required_weights_fixed_env_and_oneshot_timeout(tmp_path):
    body = """
    [module]
    name = "example"
    [[job_type]]
    id = "example.cpu_model"
    resource_class = "cpu"
    vram_mb = 0
    env = "container"
    entrypoint = ["python", "-m", "example.cpu_worker"]
    modes = ["oneshot"]
    sandbox = "standard"
    weights_required = true
    oneshot_job_timeout_s = 900
    fixed_env = { EXAMPLE_SOURCE_REVISION = "%s" }
    """ % ("f" * 40)
    jt = module.load_module(_write(tmp_path, body)).job_types[0]
    assert jt.weights_required
    assert jt.oneshot_job_timeout_s == 900
    assert jt.fixed_env == {"EXAMPLE_SOURCE_REVISION": "f" * 40}


@pytest.mark.parametrize(
    "line,match",
    [
        ('fixed_env = { GPUWORKER_JOB_ID = "x" }', "broker variable"),
        ('fixed_env = { lower = "x" }', "invalid fixed_env"),
        ("fixed_env = { SOURCE = 1 }", "must be a string"),
        ('weights_required = "yes"', "must be a bool"),
        ("oneshot_job_timeout_s = -1", "at least 0"),
        ("oneshot_job_timeout_s = nan", "finite"),
        ("load_cost_s = -5", "at least 0"),
        ("load_cost_s = inf", "finite"),
        ('load_cost_s = "slow"', "finite"),
        ("keep_warm_s = nan", "finite"),
        ("keep_warm_s = true", "finite"),
        ('co_resident = "yes"', "must be a bool"),
        ("image = 3", "must be a string"),
        ("serve_socket = 1", "must be a string"),
    ],
)
def test_rejects_invalid_prelaunch_contract_fields(tmp_path, line, match):
    body = f"""
    [module]
    name = "example"
    [[job_type]]
    id = "example.cpu_model"
    resource_class = "cpu"
    vram_mb = 0
    env = "container"
    entrypoint = ["python", "-m", "example.cpu_worker"]
    modes = ["oneshot"]
    sandbox = "standard"
    {line}
    """
    with pytest.raises(module.ModuleError, match=match):
        module.load_module(_write(tmp_path, body))


def test_scan_registry_dedup(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "module.toml").write_text(GOOD)
    (tmp_path / "b" / "module.toml").write_text(GOOD)
    with pytest.raises(module.ModuleError, match="duplicate"):
        module.scan_registry(str(tmp_path))


def test_echo_registry_loads():
    # The real echo module in the repo registry must parse.
    reg = module.scan_registry("modules")
    assert {"echo.echo", "echo.container", "echo.serve"} <= set(reg.job_types)
    assert reg.get("echo.echo").env == "host"
    serve_jt = reg.get("echo.serve")
    assert serve_jt.supports("serve") and serve_jt.serve_socket
    assert serve_jt.serve_max_jobs == 0  # not declared -> never recycled


def test_env_passthrough_and_serve_max_jobs(tmp_path):
    body = GOOD + 'env_passthrough = ["MODEL_THREADS", "MODEL_DTYPE"]\nserve_max_jobs = 25\n'
    jt = module.load_module(_write(tmp_path, body)).job_types[0]
    assert jt.env_passthrough == ("MODEL_THREADS", "MODEL_DTYPE")
    assert jt.serve_max_jobs == 25


@pytest.mark.parametrize(
    "line,match",
    [
        ('env_passthrough = "MODEL_THREADS"', "list of variable names"),
        ('env_passthrough = ["lower"]', "invalid env_passthrough"),
        ('env_passthrough = ["GPUWORKER_OUT"]', "broker variable"),
        ("serve_max_jobs = -1", "non-negative int"),
        ("serve_max_jobs = true", "non-negative int"),
    ],
)
def test_rejects_invalid_passthrough_and_recycle(tmp_path, line, match):
    body = GOOD + line + "\n"
    with pytest.raises(module.ModuleError, match=match):
        module.load_module(_write(tmp_path, body))


def test_serve_max_jobs_requires_serve_mode(tmp_path):
    body = """
    [module]
    name = "example"
    [[job_type]]
    id = "example.cpu_model"
    resource_class = "cpu"
    vram_mb = 0
    env = "container"
    entrypoint = ["python", "-m", "example.cpu_worker"]
    modes = ["oneshot"]
    serve_max_jobs = 3
    """
    with pytest.raises(module.ModuleError, match="requires serve mode"):
        module.load_module(_write(tmp_path, body))


def test_serve_mode_is_rejected_off_the_vram_lane(tmp_path):
    body = """
    [module]
    name = "example"
    [[job_type]]
    id = "example.cpu_serve"
    resource_class = "cpu"
    vram_mb = 0
    env = "container"
    entrypoint = ["python", "-m", "example.worker"]
    modes = ["serve"]
    serve_socket = "/run/job.sock"
    """
    with pytest.raises(module.ModuleError, match="only supported on the vram lane"):
        module.load_module(_write(tmp_path, body))


@pytest.mark.parametrize("env", ["wsl2-container", "windows-venv", "docker"])
def test_env_must_be_container_or_host(tmp_path, env):
    body = GOOD.replace('env = "container"', f'env = "{env}"')
    with pytest.raises(module.ModuleError, match="env must be one of"):
        module.load_module(_write(tmp_path, body))


def test_strict_implies_weights_required():
    jt = module.JobType(
        id="a.b",
        resource_class="vram",
        vram_mb=1,
        env="container",
        entrypoint=["x"],
        sandbox="strict",
    )
    assert jt.weights_required


def test_serve_mode_requires_a_container(tmp_path):
    body = GOOD.replace('env = "container"', 'env = "host"').replace(
        'image = "example/model:cuda12.8@sha256:' + "a" * 64 + '"', ""
    )
    with pytest.raises(module.ModuleError, match='env = "container"'):
        module.load_module(_write(tmp_path, body))


@pytest.mark.parametrize(
    "line,match",
    [
        ("serve_job_timeout = nan", "finite"),
        ("serve_job_timeout = inf", "finite"),
        ("serve_job_timeout = 0", "greater than 0"),
        ("serve_job_timeout = -1", "greater than 0"),
    ],
)
def test_rejects_serve_fields_that_would_disable_the_watchdog(tmp_path, line, match):
    """A NaN serve_job_timeout made a NaN watchdog deadline, which no clock reading
    passes, so a hung serve job was never failed."""
    body = f"""
    [module]
    name = "example"
    [[job_type]]
    id = "example.serve"
    vram_mb = 100
    env = "container"
    entrypoint = ["python", "w.py"]
    modes = ["serve"]
    serve_socket = "/run/gpuworker/job.sock"
    {line}
    """
    with pytest.raises(module.ModuleError, match=match):
        module.load_module(_write(tmp_path, body))


@pytest.mark.parametrize(
    "body,named",
    [
        (GOOD + "serve_max_job = 5\n", "serve_max_job"),  # in [[job_type]]
        (GOOD.replace('version = "0.1.0"', 'version = "0.1.0"\nverison = "2"'), "verison"),
        (GOOD + "\n[jobtype]\nid = 'x.y'\n", "jobtype"),
    ],
)
def test_an_unknown_key_fails_the_load(tmp_path, body, named):
    """A misspelled key silently kept its default."""
    with pytest.raises(module.ModuleError, match=f"unknown key.*'{named}'"):
        module.load_module(_write(tmp_path, body))


def test_every_key_the_parser_reads_is_known():
    import inspect
    import re

    src = inspect.getsource(module._parse_job_type)
    read = set(re.findall(r'(?:raw\.get\(|req\(|_seconds\(raw, jt_id, )"([a-z_]+)"', src))
    assert read and read <= module.JOB_TYPE_KEYS
    assert read == module.JOB_TYPE_KEYS


def test_the_guide_lists_each_job_type_key_once():
    """MODULE_GUIDE 2.2 had `co_resident` twice."""
    import os
    import re

    path = os.path.join(os.path.dirname(__file__), "..", "docs", "MODULE_GUIDE.md")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    table = text.split("### 2.2 Field by field", 1)[1].split("\n### ", 1)[0]
    rows = re.findall(r"^\| `([a-z_]+)` \|", table, re.M)
    assert len(rows) == len(set(rows))
    assert set(rows) == module.JOB_TYPE_KEYS
