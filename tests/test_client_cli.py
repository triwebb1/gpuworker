"""Client submit naming and CLI behaviour that a user relies on directly."""

import json
import os

import pytest

from gpuworker import cli
from gpuworker.client import submit
from gpuworker.manifest import JobManifest


def test_submit_names_are_unique_per_submission(tmp_path):
    # Re-submitting a job_id must not overwrite a bundle in inbox/ or be answered by an
    # older result with the same name in outbox/.
    m = JobManifest.from_dict({"job_id": "same", "job_type": "echo.echo"})
    a, b = submit(str(tmp_path), [m]), submit(str(tmp_path), [m])
    assert a != b and a.startswith("same-") and a.endswith(".zip")
    assert sorted(os.listdir(tmp_path / "inbox")) == sorted([a, b])


def test_missing_config_file_is_an_error(tmp_path):
    with pytest.raises(SystemExit, match="config file not found"):
        cli.main(["--config", str(tmp_path / "nope.toml"), "registry", "list"])


def test_run_once_does_not_write_the_brokers_state_files(tmp_path, capsys):
    # run-once may run next to a live broker: it may read the learned footprints but
    # must write its own copy, never the broker's file.
    fp = tmp_path / "footprints.json"
    fp.write_text(json.dumps({"ex.m": 1234}))
    status = tmp_path / "status.json"
    cfg = tmp_path / "gpuworker.toml"
    cfg.write_text(
        'container_runtime = "local"\n'
        'registry_dir = "modules"\n'
        f"footprints_path = {json.dumps(str(fp))}\n"
        f"status_path = {json.dumps(str(status))}\n"
    )
    rc = cli.main(
        [
            "--config",
            str(cfg),
            "--mock-gpu",
            "run-once",
            "echo.echo",
            "--text",
            "hi",
            "--output",
            "out/hi.txt",
        ]
    )
    assert rc == 0
    assert "status : done" in capsys.readouterr().out
    assert json.loads(fp.read_text()) == {"ex.m": 1234}
    assert not status.exists()


def test_run_once_removes_its_temporary_directory(tmp_path, monkeypatch, capsys):
    """Every run left a gpuw-runonce-* directory in the system temp dir."""
    import tempfile

    made = []
    real = tempfile.mkdtemp

    def mkdtemp(**kw):
        if kw.get("prefix") != "gpuw-runonce-":
            return real(**kw)  # the scheduler's own batch dirs
        made.append(real(dir=str(tmp_path), **kw))
        return made[-1]

    monkeypatch.setattr(tempfile, "mkdtemp", mkdtemp)
    cfg = tmp_path / "gpuworker.toml"
    cfg.write_text('container_runtime = "local"\nregistry_dir = "modules"\n')
    argv = ["--config", str(cfg), "--mock-gpu", "run-once", "echo.echo", "--text", "hi"]
    assert cli.main(argv) == 0
    assert made and not os.path.exists(made[0])
    assert cli.main(argv + ["--keep"]) == 0
    assert os.path.isdir(made[1])
    assert f"(kept {made[1]})" in capsys.readouterr().err


def test_the_quick_start_runs_with_the_default_config(tmp_path, monkeypatch, capsys):
    """The README one-liner has no --config, so the runtime defaults to podman, under which
    the broker refuses host job types unless allow_host_jobs is set. run-once runs only the
    operator's own job, so it allows them; the quick start failed without that."""
    import shutil

    shutil.copytree(
        os.path.join(os.path.dirname(__file__), "..", "modules", "echo"),
        tmp_path / "modules" / "echo",
    )
    monkeypatch.chdir(tmp_path)  # no gpuworker.toml here: every setting is a default
    rc = cli.main(
        ["--mock-gpu", "run-once", "echo.echo", "--text", "hello", "--output", "out/hello.txt"]
    )
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "status : done" in out


def _status_config(tmp_path, status):
    cfg = tmp_path / "gpuworker.toml"
    cfg.write_text(
        'container_runtime = "local"\nregistry_dir = "modules"\n'
        f"status_path = {json.dumps(str(status))}\n"
        f"footprints_path = {json.dumps(str(tmp_path / 'fp.json'))}\n"
        f"queue_roots = [{json.dumps(str(tmp_path / 'Q'))}]\n"
    )
    return str(cfg)


def test_status_prints_the_live_file_and_never_overwrites_it(tmp_path, capsys):
    """`gpuworker status` wrote an idle snapshot over the running broker's status.json."""
    status = tmp_path / "status.json"
    live = '{"pending": 7}'
    status.write_text(live)
    cfg = _status_config(tmp_path, status)
    assert cli.main(["--config", cfg, "--mock-gpu", "status"]) == 0
    assert capsys.readouterr().out.strip() == live
    assert status.read_text() == live
    assert cli.main(["--config", cfg, "--mock-gpu", "status", "--snapshot", str(status)]) == 1
    assert "status_path" in capsys.readouterr().err
    assert status.read_text() == live
    snap = tmp_path / "snap.json"
    assert cli.main(["--config", cfg, "--mock-gpu", "status", "--snapshot", str(snap)]) == 0
    assert json.loads(snap.read_text())["pending"] == 0
    assert status.read_text() == live


def test_status_from_a_broker_that_stopped_says_so(tmp_path, capsys):
    """A dead broker's last status.json was printed as if it were current."""
    import datetime

    status = tmp_path / "status.json"
    cfg = _status_config(tmp_path, status)
    now = datetime.datetime.now(datetime.UTC).replace(microsecond=0)
    status.write_text(json.dumps({"ts": now.isoformat(), "pending": 0}))
    assert cli.main(["--config", cfg, "status"]) == 0
    assert "status written" in capsys.readouterr().err
    old = now - datetime.timedelta(seconds=cli.STATUS_STALE_S + 60)
    status.write_text(json.dumps({"ts": old.isoformat(), "pending": 0}))
    assert cli.main(["--config", cfg, "status"]) == 1
    captured = capsys.readouterr()
    assert '"pending": 0' in captured.out and "broker may have stopped" in captured.err


def test_status_names_why_the_gpu_is_degraded(tmp_path):
    from gpuworker import gpu
    from gpuworker.obs import write_status

    from helpers import cpu_jt, make_scheduler, registry

    g = gpu.MockGpu()
    sched = make_scheduler(tmp_path, registry(cpu_jt()), monitor=g)
    g.unresponsive = True
    sched._check_gpu_health()
    path = tmp_path / "snap.json"
    write_status(str(path), sched)
    ledger = json.loads(path.read_text())["ledger"]
    assert ledger["degraded"] and ledger["degraded_reason"] == "the GPU did not answer the monitor"
    g.unresponsive = False
    sched.tick()
    write_status(str(path), sched)
    assert json.loads(path.read_text())["ledger"]["degraded_reason"] == ""


def test_status_fails_while_the_gpu_is_degraded(tmp_path, capsys):
    """status exited 0 with the GPU degraded, so a health check built on it passed."""
    import datetime

    status = tmp_path / "status.json"
    cfg = _status_config(tmp_path, status)
    now = datetime.datetime.now(datetime.UTC).isoformat()
    ledger = {"degraded": True, "degraded_reason": "a stopped worker did not exit"}
    status.write_text(json.dumps({"ts": now, "ledger": ledger}))
    assert cli.main(["--config", cfg, "status"]) == 1
    assert "the GPU is degraded: a stopped worker did not exit" in capsys.readouterr().err
    ledger.update(degraded=False, degraded_reason="")
    status.write_text(json.dumps({"ts": now, "ledger": ledger}))
    assert cli.main(["--config", cfg, "status"]) == 0
    assert "degraded" not in capsys.readouterr().err


def test_status_without_a_running_broker_says_so(tmp_path, capsys):
    cfg = _status_config(tmp_path, tmp_path / "status.json")
    assert cli.main(["--config", cfg, "--mock-gpu", "status"]) == 1
    assert "is the broker running?" in capsys.readouterr().err


def test_the_readme_service_unit_sets_its_working_directory():
    """The example config's paths are relative and resolve against the working directory;
    the unit had none, so under systemd they resolved against /. It also restarted at
    once after a config error or a refused lock."""
    path = os.path.join(os.path.dirname(__file__), "..", "README.md")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    unit = text.split("```ini\n", 1)[1].split("```", 1)[0]
    keys = dict(line.split("=", 1) for line in unit.splitlines() if "=" in line)
    assert keys.get("WorkingDirectory", "").startswith("/")
    assert float(keys.get("RestartSec", "0")) > 0
