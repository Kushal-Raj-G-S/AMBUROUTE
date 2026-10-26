"""Severity- and specialty-aware extension.

Patients are stable / urgent / critical; some need a specialty centre (cardiac, stroke, trauma).
A patient taken to a hospital without the needed capability is stabilised and then transferred,
which costs a fixed overhead plus the drive to the nearest capable hospital. The metric is the
time to *definitive care* (bed at a capable hospital, or completed transfer).

ALL clinical parameters (severity mix, specialty mix, capabilities, transfer overhead, severity
weights) are ASSUMPTIONS, not measured data. See docs/clinical_extension.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .nash import NashPolicy, nash_assign
from .policies import BIG

SEV_WEIGHTS = np.array([1.0, 1.5, 3.0])  # stable, urgent, critical (assumption)


def with_capabilities(df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Assign cardiac / stroke / trauma capability from hospital size. ASSUMED, not real data:
    large hospitals (>= 500 beds) are likelier to host each service, and every specialty exists
    in at least two hospitals."""
    rng = np.random.default_rng(seed)
    big = df["beds"].to_numpy() >= 500
    out = df.copy()
    for col, p_big, p_small in (("cap_cardiac", 0.7, 0.25), ("cap_stroke", 0.6, 0.20), ("cap_trauma", 0.7, 0.25)):
        cap = rng.random(len(df)) < np.where(big, p_big, p_small)
        if cap.sum() < 2:
            cap[np.argsort(-df["beds"].to_numpy())[:2]] = True
        out[col] = cap
    return out


class NearestCapable:
    """Nearest hospital that can give definitive care; ignores beds."""

    name = "nearest_capable"

    def assign(self, view, rng):
        return list(np.argmin(np.where(view.capable, view.eta, BIG), axis=1))


class ClinicalHeuristic:
    """Capacity heuristic limited to capable hospitals, most severe patient first, with the
    travel-time term weighted by severity."""

    name = "clinical_heuristic"

    def __init__(self, weight: float = 20.0, sev_weights=SEV_WEIGHTS):
        self.w = weight
        self.sw = np.asarray(sev_weights, dtype=float)

    def assign(self, view, rng):
        inflight = view.inflight.astype(float).copy()
        out = [0] * len(view.eta)
        for i in np.argsort(-view.severity, kind="stable"):
            p = int(view.needs_icu[i])
            eff = view.free[p] - inflight[p]
            ok = view.capable[i]
            score = self.sw[view.severity[i]] * view.eta[i] + self.w / (1.0 + np.maximum(eff, 0))
            score = score + np.where(eff <= 0, BIG, 0.0) + np.where(ok, 0.0, BIG)
            h = int(np.argmin(score))
            if eff[h] <= 0 or not ok[h]:  # everything looks full: nearest capable hospital
                h = int(np.argmin(np.where(ok, view.eta[i], BIG)))
            out[i] = h
            inflight[p, h] += 1
        return out


class ClinicalNash(NashPolicy):
    """Equilibrium assignment restricted to capable hospitals, with severity-weighted travel time.

    Strategy restriction and the weight on the travel-time term are player-specific additive
    terms, so Rosenthal's potential still works: Phi = sum_i w_i * tau_i,h(i) + (load terms), and a
    unilateral move changes Phi by exactly the mover's change in cost."""

    name = "clinical_nash"

    def __init__(self, beta: float = 30.0, power: float = 1.0, sev_weights=SEV_WEIGHTS):
        super().__init__(beta, power)
        self.sw = np.asarray(sev_weights, dtype=float)

    def assign(self, view, rng):
        w = self.sw[view.severity]
        asg, iters, ok = nash_assign(view, self.beta, rng, power=self.power, weights=w, allowed=view.capable)
        self.last_iters, self.last_converged = iters, ok
        self.iters_log.append(iters)
        self.converged_log.append(ok)
        return asg
