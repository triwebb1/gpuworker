import os
import shutil

import pytest

_STATE_FILES = ("footprints.json", "status.json", "state")


@pytest.fixture(autouse=True)
def _no_state_files_in_cwd():
    """Fail any test that writes broker state into the working directory (the repo root
    in development, a deployment directory in the worst case). Use helpers.make_config."""
    before = {f for f in _STATE_FILES if os.path.exists(f)}
    yield
    leaked = {f for f in _STATE_FILES if os.path.exists(f)} - before
    for f in leaked:
        shutil.rmtree(f) if os.path.isdir(f) else os.remove(f)
    assert not leaked, f"test wrote {sorted(leaked)} into the working directory"
