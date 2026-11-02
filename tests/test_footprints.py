import os

from gpuworker.footprints import Footprints
from gpuworker.module import JobType


def _jt(id="ex.m", vram=5000):
    return JobType(
        id=id,
        resource_class="vram",
        vram_mb=vram,
        env="container",
        entrypoint=["x"],
        image="i@sha256:" + "a" * 64,
        sandbox="standard",
    )


def test_effective_takes_max_of_declared_and_observed():
    fp = Footprints()
    jt = _jt(vram=5000)
    assert fp.effective(jt) == 5000  # nothing observed yet
    fp.record("ex.m", 6200)
    assert fp.effective(jt) == 6200  # reality > spec wins


def test_record_only_ratchets_up():
    fp = Footprints()
    assert fp.record("ex.m", 6000) is True
    assert fp.record("ex.m", 5500) is False  # lower sample ignored
    assert fp.observed("ex.m") == 6000
    assert fp.record("ex.m", 6500) is True
    assert fp.observed("ex.m") == 6500


def test_persists_and_reloads(tmp_path):
    path = str(tmp_path / "fp.json")
    fp = Footprints(path)
    fp.record("ex.m", 7000)
    assert os.path.exists(path)
    fp2 = Footprints(path)  # reload from disk
    assert fp2.observed("ex.m") == 7000


def test_ignores_nonpositive():
    fp = Footprints()
    assert fp.record("ex.m", 0) is False
    assert fp.record("ex.m", -10) is False


def test_image_change_resets_footprint():
    # A rebuild (new image) must invalidate the stale peak so a shrunk model can load
    # again instead of being latched to its old, larger footprint.
    fp = Footprints()
    jt = _jt(vram=14300)
    fp.record("ex.m", 15900, image="img@sha256:old")
    assert fp.effective(jt, "img@sha256:old") == 15900  # valid under the same image
    # new image: the old peak is stale -> admission falls back to declared
    assert fp.effective(jt, "img@sha256:new") == 14300
    assert fp.observed("ex.m", "img@sha256:new") is None
    # observing under the new image resets (records even though it's lower than 15900)
    assert fp.record("ex.m", 15260, image="img@sha256:new") is True
    assert fp.effective(jt, "img@sha256:new") == 15260
    # and it ratchets up again within the new image
    assert fp.record("ex.m", 15000, image="img@sha256:new") is False
    assert fp.record("ex.m", 15400, image="img@sha256:new") is True
    assert fp.effective(jt, "img@sha256:new") == 15400
