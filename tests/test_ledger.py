from gpuworker.ledger import Ledger
from gpuworker.module import JobType


def jt(id, vram, co=False, lane="vram", profile="preallocated"):
    # Tests that expect packing use preallocated (the only co-residable profile).
    return JobType(
        id=id,
        resource_class=lane,
        vram_mb=vram,
        env="container",
        entrypoint=["x"],
        image="i@sha256:" + "a" * 64,
        co_resident=co,
        vram_profile=profile,
        sandbox="standard",
    )


def test_single_heavy_fits_then_blocks():
    led = Ledger(admit_budget_mb=15000, cpu_slots=2)
    big = jt("ex.large", 14300, co=False)
    assert led.can_fit(big)
    led.admit_vram(big, now=1.0)
    # A second model cannot co-reside with an exclusive resident.
    small = jt("ex.model", 5000, co=True)
    assert not led.can_fit(small)


def test_co_resident_small_models_pack():
    led = Ledger(admit_budget_mb=15000, cpu_slots=2)
    small = jt("ex.model", 5000, co=True)
    other = jt("ex.other", 6000, co=True)
    led.admit_vram(small, now=1.0)
    assert led.can_fit(other)  # 5000 + 6000 <= 15000, both preallocated
    led.admit_vram(other, now=2.0)
    assert led.resident_vram_mb == 11000


def test_growing_model_never_co_resides():
    # Even with co_resident=True and room to spare, a `growing` model gets the card
    # alone (its peak isn't known at admit), and blocks others from packing with it.
    led = Ledger(admit_budget_mb=15000, cpu_slots=2)
    grow = jt("ex.grow", 5000, co=True, profile="growing")
    small = jt("ex.small", 2000, co=True, profile="preallocated")
    assert led.can_fit(grow)  # alone on an empty card it fits
    led.admit_vram(grow, now=1.0)
    assert not led.can_fit(small)  # growing resident demands the card alone

    led2 = Ledger(admit_budget_mb=15000, cpu_slots=2)
    led2.admit_vram(small, now=1.0)
    assert not led2.can_fit(grow)  # growing newcomer won't pack onto a resident


def test_co_resident_blocked_by_budget():
    led = Ledger(admit_budget_mb=10000, cpu_slots=2)
    small = jt("ex.model", 5000, co=True)
    big = jt("x.big", 6000, co=True)
    led.admit_vram(small, now=1.0)
    assert not led.can_fit(big)  # 5000 + 6000 > 10000


def test_eviction_plan_frees_lru_idle():
    led = Ledger(admit_budget_mb=15000, cpu_slots=2)
    a = jt("a.m", 8000, co=True)
    b = jt("b.m", 6000, co=True)
    led.admit_vram(a, now=1.0)
    led.finish_vram("a.m", now=1.5, keep_warm_s=100)  # idle, lru
    led.admit_vram(b, now=2.0)
    led.finish_vram("b.m", now=2.5, keep_warm_s=100)
    # New exclusive heavy job needs the whole card -> evict both.
    heavy = jt("c.big", 14000, co=False)
    plan = led.eviction_plan(heavy)
    assert plan is not None and set(plan) == {"a.m", "b.m"}


def test_eviction_impossible_when_resident_busy():
    led = Ledger(admit_budget_mb=15000, cpu_slots=2)
    a = jt("a.m", 14000, co=False)
    led.admit_vram(a, now=1.0)  # busy (not finished)
    heavy = jt("c.big", 14000, co=False)
    assert led.eviction_plan(heavy) is None


def test_resident_reuse_is_free():
    led = Ledger(admit_budget_mb=15000, cpu_slots=2)
    a = jt("a.m", 8000, co=True)
    led.admit_vram(a, now=1.0)
    led.finish_vram("a.m", now=1.5, keep_warm_s=100)
    assert led.is_resident("a.m")
    # admitting another job of the same type adds no VRAM
    led.admit_vram(a, now=2.0)
    assert led.resident_vram_mb == 8000


def test_cpu_slots():
    led = Ledger(admit_budget_mb=15000, cpu_slots=2)
    assert led.can_admit_cpu()
    led.admit_cpu()
    led.admit_cpu()
    assert not led.can_admit_cpu()
    led.finish_cpu()
    assert led.can_admit_cpu()


def test_expired_warm():
    led = Ledger(admit_budget_mb=15000, cpu_slots=2)
    a = jt("a.m", 8000, co=True)
    led.admit_vram(a, now=1.0)
    led.finish_vram("a.m", now=2.0, keep_warm_s=10)
    assert led.expired_warm(now=5.0) == []
    assert led.expired_warm(now=20.0) == ["a.m"]
