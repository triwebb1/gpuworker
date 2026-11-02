from gpuworker import gpu


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


def test_reclaim_gate_succeeds_when_memory_frees():
    g = gpu.MockGpu(total_mb=16000, used_mb=14000)  # 2000 free
    clk = FakeClock()

    # After two polls, simulate the previous context releasing.
    calls = {"n": 0}
    real_free = g.free_mb

    def free_mb():
        calls["n"] += 1
        if calls["n"] >= 3:
            g.used = 1000
        return real_free()

    g.free_mb = free_mb  # type: ignore[assignment]
    ok = gpu.wait_for_free(
        g, needed_mb=10000, timeout_s=10, poll_s=1.0, sleep=clk.sleep, clock=clk.monotonic
    )
    assert ok


def test_reclaim_gate_times_out():
    g = gpu.MockGpu(total_mb=16000, used_mb=14000)  # stays at 2000 free
    clk = FakeClock()
    ok = gpu.wait_for_free(
        g, needed_mb=10000, timeout_s=5, poll_s=1.0, sleep=clk.sleep, clock=clk.monotonic
    )
    assert not ok


def test_reclaim_gate_immediate():
    g = gpu.MockGpu(total_mb=16000, used_mb=1000)  # 15000 free already
    ok = gpu.wait_for_free(g, needed_mb=10000, timeout_s=1)
    assert ok
