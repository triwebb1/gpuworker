"""Observed-footprint reconciliation (SPEC 9).

A module *declares* `vram_mb`, but reality can exceed it (a model that peaks higher
than measured, a driver/runtime that grew, a dtype change). Admitting purely on the
declared number would over-commit the card and OOM.

So the broker measures actual peak VRAM when a model runs as the sole resident (clean
attribution) and persists the observation. Admission then uses
``max(declared, observed)``. Observations ratchet the effective footprint up *within
one image*, so a lucky low sample never under-counts a model that has peaked higher.

Each observation is stamped with the image it ran under. When the image changes, the
old observation is ignored and admission falls back to the declared number until it
re-learns. Otherwise a rebuild that shrinks a model would keep the old peak latched,
and the reclaim gate would keep demanding VRAM the model can never load to disprove.
"""

from __future__ import annotations

import json
import os
import threading


class Footprints:
    def __init__(self, path: str | None = None):
        self._path = path
        self._lock = threading.Lock()
        # jt_id -> {"mb": int, "image": str}
        self._observed: dict[str, dict] = {}
        if path and os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, dict) and isinstance(v.get("mb"), (int, float)):
                            self._observed[k] = {
                                "mb": int(v["mb"]),
                                "image": str(v.get("image", "")),
                            }
            except (ValueError, OSError):
                pass

    def observed(self, jt_id: str, image: str | None = None) -> int | None:
        """Last observed peak for *jt_id*. If *image* is given, return it only when it
        was observed under that same image (else None; it is stale across a rebuild)."""
        e = self._observed.get(jt_id)
        if e is None:
            return None
        if image is not None and e["image"] != image:
            return None
        return e["mb"]

    def effective(self, jt, image: str | None = None) -> int:
        """Admission footprint: the larger of declared and the worst observed peak that
        is still valid for *image* (a different image invalidates the observation)."""
        return max(jt.vram_mb, self.observed(jt.id, image) or 0)

    def record(self, jt_id: str, peak_mb: int, image: str = "") -> bool:
        """Record an observed peak. Ratchets up within the same image; a different
        image resets to this fresh observation (the old model's peak is irrelevant).
        Returns True if the stored value changed (caller logs the variance)."""
        if peak_mb <= 0:
            return False
        with self._lock:
            e = self._observed.get(jt_id)
            if e is None or e["image"] != image:
                self._observed[jt_id] = {"mb": peak_mb, "image": image}
                self._persist()
                return True
            if peak_mb <= e["mb"]:
                return False
            e["mb"] = peak_mb
            self._persist()
            return True

    def reset(self, jt_id: str) -> bool:
        """Forget *jt_id*'s observation. Returns True if there was one."""
        with self._lock:
            if self._observed.pop(jt_id, None) is None:
                return False
            self._persist()
            return True

    def _persist(self) -> None:
        if not self._path:
            return
        try:
            tmp = self._path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._observed, fh, indent=2, sort_keys=True)
            os.replace(tmp, self._path)
        except OSError:
            pass
