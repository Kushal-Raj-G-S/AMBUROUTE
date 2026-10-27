"""Online adaptation to travel-time drift.

The planner's travel-time model is fitted once. When roads near some hospitals jam, trips there take longer
than planned. `OnlineETA` wraps any dispatch policy and learns a per-hospital correction from what finished
trips actually took (realised / planned minutes), as an exponentially weighted average of log-ratios. It uses
only information a dispatch centre has: arrival times of its own ambulances.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np


class OnlineETA:
    """Per-hospital correction learned from en-route progress reports. Only *relative* corrections are
    applied (hospital log-ratio minus the citywide average), so a uniform slowdown does not distort the
    balance between travel time and capacity that the base policy was tuned for."""

    def __init__(self, base, alpha: float = 0.3, n_hospitals: int = 18):
        self.base, self.alpha = base, alpha
        self.logc = np.full(n_hospitals, np.nan)  # per-hospital EWMA of log(realised / planned)
        self.g = 0.0  # citywide EWMA
        self._sim = None
        self.name = f"{base.name}+online(a={alpha:g})"

    def observe(self, obs, sim):
        if sim is not self._sim:  # a new episode: forget everything learned in the previous one
            self._sim, self.logc, self.g = sim, np.full_like(self.logc, np.nan), 0.0
        for h, ratio in obs:
            x = np.log(max(ratio, 1e-3))
            self.g += self.alpha * 0.5 * (x - self.g)  # slower citywide average
            self.logc[h] = x if np.isnan(self.logc[h]) else self.logc[h] + self.alpha * (x - self.logc[h])

    def assign(self, view, rng):
        rel = np.where(np.isnan(self.logc), 0.0, self.logc - self.g)
        return self.base.assign(replace(view, eta=view.eta * np.exp(rel)[None, :]), rng)


class OracleETA:
    """Upper bound: knows the true slowdown of every hospital at every moment. Not realisable."""

    def __init__(self, base):
        self.base = base
        self.mult = None
        self.name = f"{base.name}+oracle"

    def observe(self, obs, sim):
        on = sim.cfg.drift_n and sim.t >= sim.cfg.drift_start
        self.mult = sim.drift if on else np.ones(sim.H)

    def assign(self, view, rng):
        m = np.ones(view.eta.shape[1]) if self.mult is None else self.mult
        return self.base.assign(replace(view, eta=view.eta * m[None, :]), rng)
