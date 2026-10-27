"""Hospital-selection policies. All expose `assign(view, rng) -> list[int]`."""
from __future__ import annotations

import numpy as np

from .sim import View

BIG = 1e6


def _free(view: View, inflight: np.ndarray | None = None) -> np.ndarray:
    """(n, H) published free beds of the pool each request needs."""
    pool = view.needs_icu.astype(int)
    f = view.free[pool]
    if inflight is not None:
        f = f - inflight[pool]
    return f


class Nearest:
    """Closest hospital by ETA; ignores beds entirely."""

    name = "nearest"

    def assign(self, view, rng):
        return list(np.argmin(view.eta, axis=1))


class NearestWithBeds:
    """Closest hospital whose *published* bed count is positive."""

    name = "nearest_beds"

    def assign(self, view, rng):
        eta = np.where(_free(view) > 0, view.eta, BIG)
        return list(np.argmin(eta, axis=1))


class PrototypeHeuristic:
    """The scoring rule of the original repo (src/main.py):
    0.6 * travel_seconds + 0.4 * 1000 / available_beds, over hospitals with beds.
    """

    name = "prototype_heuristic"

    def assign(self, view, rng):
        free = _free(view)
        score = 0.6 * view.eta * 60 + 0.4 * 1000 / np.maximum(free, 1)
        score = np.where(free > 0, score, BIG)
        return list(np.argmin(score, axis=1))


class CapacityHeuristic:
    """Paper's heuristic fallback: ETA plus a capacity-pressure term that
    accounts for ambulances already on the road (sequential within a batch)."""

    name = "capacity_heuristic"

    def __init__(self, weight: float = 30.0, use_inflight: bool = True):
        self.w = weight
        self.use_inflight = use_inflight  # False = ablation: ignore ambulances already on the road

    def assign(self, view, rng):
        inflight = view.inflight.astype(float).copy()
        if not self.use_inflight:
            inflight[:] = 0
        out = []
        for i in range(len(view.eta)):
            p = int(view.needs_icu[i])
            eff = view.free[p] - inflight[p]
            pressure = 1.0 / (1.0 + np.maximum(eff, 0))
            score = view.eta[i] + self.w * pressure + np.where(eff <= 0, BIG, 0)
            h = int(np.argmin(score))
            if eff[h] <= 0:  # everything looks full: fall back to nearest
                h = int(np.argmin(view.eta[i]))
            out.append(h)
            inflight[p, h] += 1
        return out
