"""VRAM/lane admission ledger (SPEC 3, 5, 9).

Pure accounting state owned by the single-writer scheduler loop. No I/O, no threads;
fully unit-testable. The loop calls these methods to decide admission; it never
mutates the state from anywhere else (single-writer invariant, SPEC 3).

Lanes:
  * ``vram``: admitted against a VRAM budget with co-residency rules.
  * ``cpu``:  admitted against a fixed slot count; does not touch the VRAM budget, so a
              cpu job runs while a vram model is resident.
"""

from __future__ import annotations

from dataclasses import dataclass

from .module import LANE_VRAM, JobType


@dataclass
class Resident:
    """A vram-lane job type currently loaded (warm or actively running)."""

    jt: JobType
    running: int = 0  # how many jobs of this type are running right now
    last_used: float = 0.0  # monotonic stamp, for LRU eviction
    warm_until: float = 0.0  # keep-warm deadline once idle

    @property
    def busy(self) -> bool:
        # A type with >=1 running job must not be evicted (its container is live).
        return self.running > 0


class Ledger:
    def __init__(self, admit_budget_mb: int, cpu_slots: int):
        self.admit_budget_mb = admit_budget_mb
        self.cpu_slots = cpu_slots
        self.residents: dict[str, Resident] = {}
        self.cpu_active = 0

    # ---- vram lane ---------------------------------------------------------

    @property
    def resident_vram_mb(self) -> int:
        return sum(r.jt.vram_mb for r in self.residents.values())

    def is_resident(self, jt_id: str) -> bool:
        return jt_id in self.residents

    def _has_exclusive_resident(self) -> bool:
        # A resident that is not packable (growing, or co_resident=false) demands the
        # card alone and blocks any newcomer from co-residing.
        return any(not r.jt.packable for r in self.residents.values())

    def can_fit(self, jt: JobType) -> bool:
        """Could *jt* (not yet resident) be loaded alongside the current residents
        without eviction? Enforces co-residency (packable = co_resident and
        preallocated) + the VRAM budget. Only meaningful for vram-lane job types."""
        if jt.resource_class != LANE_VRAM:
            return True
        if self.residents:
            # A newcomer that isn't packable demands the card alone.
            if not jt.packable:
                return False
            # An existing exclusive resident blocks any newcomer.
            if self._has_exclusive_resident():
                return False
        return self.resident_vram_mb + jt.vram_mb <= self.admit_budget_mb

    def evictable(self) -> list[Resident]:
        """Idle residents, LRU first: candidates to free VRAM for a newcomer."""
        idle = [r for r in self.residents.values() if not r.busy]
        idle.sort(key=lambda r: r.last_used)
        return idle

    def eviction_plan(self, jt: JobType) -> list[str] | None:
        """Which resident job-type ids to evict so *jt* fits. Returns [] if it
        already fits, a list to evict, or None if impossible (would need to evict a
        busy resident). Greedy LRU."""
        if self.is_resident(jt.id) or self.can_fit(jt):
            return []
        # Simulate evicting idle residents LRU-first until jt fits.
        freed = 0
        plan: list[str] = []
        remaining = dict(self.residents)
        for r in self.evictable():
            plan.append(r.jt.id)
            freed += r.jt.vram_mb
            del remaining[r.jt.id]
            exclusive_left = any(not rr.jt.packable for rr in remaining.values())
            fits_budget = (self.resident_vram_mb - freed) + jt.vram_mb <= self.admit_budget_mb
            co_ok = jt.packable or not remaining
            no_exclusive = not exclusive_left
            if fits_budget and co_ok and no_exclusive:
                return plan
        return None  # cannot free enough without touching a busy resident

    def admit_vram(self, jt: JobType, now: float) -> None:
        r = self.residents.get(jt.id)
        if r is None:
            r = Resident(jt=jt)
            self.residents[jt.id] = r
        r.running += 1
        r.last_used = now

    def finish_vram(self, jt_id: str, now: float, keep_warm_s: float) -> None:
        r = self.residents.get(jt_id)
        if r is None:
            return
        r.running = max(0, r.running - 1)
        r.last_used = now
        # Only (re)start the keep-warm clock once the type is fully idle.
        if r.running == 0:
            r.warm_until = now + keep_warm_s

    def start_serve_job(self, jt_id: str, now: float) -> None:
        """A new job for an already-resident serve container: running++ only (no new
        VRAM: the model is already loaded). The resident must exist (the cold/first
        job created it via admit_vram). Do not call this for the first job."""
        r = self.residents.get(jt_id)
        if r is not None:
            r.running += 1
            r.last_used = now

    def finish_serve_job(self, jt_id: str, now: float, keep_warm_s: float) -> None:
        """A serve job finished: running--, (re)arm keep-warm only when fully idle.
        Same accounting as finish_vram; the resident persists (warm container)."""
        self.finish_vram(jt_id, now, keep_warm_s)

    def evict(self, jt_id: str) -> None:
        self.residents.pop(jt_id, None)

    def expired_warm(self, now: float) -> list[str]:
        return [r.jt.id for r in self.residents.values() if not r.busy and now >= r.warm_until]

    # ---- cpu lane ----------------------------------------------------------

    def can_admit_cpu(self) -> bool:
        return self.cpu_active < self.cpu_slots

    def admit_cpu(self) -> None:
        self.cpu_active += 1

    def finish_cpu(self) -> None:
        self.cpu_active = max(0, self.cpu_active - 1)
