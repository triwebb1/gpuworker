import pytest

from gpuworker.config import Config


@pytest.mark.parametrize("bad", [-1, True, 1.5, "1"])
def test_config_rejects_bad_max_retries(tmp_path, bad):
    with pytest.raises(ValueError, match="max_retries"):
        Config(max_retries=bad)


@pytest.mark.parametrize(
    "field, bad",
    [
        ("cpu_slots", 0),  # no cpu job could ever run
        ("max_claims_per_tick", 0),  # nothing would ever be claimed
        ("tmpfs_mb", 0),
        ("serve_ready_timeout_s", 0),
        ("gpu_index", -1),
        ("cpu_slots", "2"),
        ("nvml_poll_ms", 2.5),
        ("reclaim_timeout_s", "30"),
        ("container_runtime", "lxc"),
        ("state_dir", ""),
        ("sandbox_image", 3),
        ("allow_unpinned_images", 1),
        ("queue_roots", "Q"),
        ("queue_roots", [""]),
        ("container_user", "0:0"),  # root
        ("container_user", "root"),
        ("container_user", "1000"),
        ("reclaim_timeout_s", float("nan")),  # every comparison with NaN is false
        ("serve_ready_timeout_s", float("inf")),
        ("allow_host_jobs", "yes"),
    ],
)
def test_config_rejects_bad_values(field, bad):
    """Config used to accept zero capacities and wrong types, which stranded jobs or
    failed later as a TypeError far from the setting."""
    with pytest.raises(ValueError, match=field):
        Config(**{field: bad})


def test_config_accepts_an_int_for_a_float_field():
    assert Config(reclaim_timeout_s=30).reclaim_timeout_s == 30
