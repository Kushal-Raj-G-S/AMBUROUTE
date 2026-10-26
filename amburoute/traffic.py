"""Synthetic time-of-day traffic model.

Stands in for live Mapbox/OSRM traffic so experiments are reproducible and
free. The planner sees the deterministic ETA; the simulator realises a noisy
travel time, so ETA error is part of the experiment.
"""
from __future__ import annotations

import numpy as np


def _bump(hour: float, centre: float, width: float) -> float:
    return float(np.exp(-((hour - centre) ** 2) / (2 * width**2)))


class TrafficModel:
    def __init__(
        self,
        circuity: float = 1.35,  # road distance / straight-line distance
        free_speed_kmh: float = 38.0,  # urban free-flow speed
        ambulance_boost: float = 1.25,  # sirens, lane priority
        sigma: float = 0.18,  # lognormal sd of realised/estimated time
    ):
        self.circuity = circuity
        self.free_speed = free_speed_kmh
        self.boost = ambulance_boost
        self.sigma = sigma

    def slowdown(self, minute_of_day: float) -> float:
        h = (minute_of_day / 60.0) % 24
        return max(0.55 * _bump(h, 9.5, 1.3), 0.60 * _bump(h, 18.5, 1.6))

    def speed_kmh(self, minute_of_day: float) -> float:
        return self.free_speed * (1 - self.slowdown(minute_of_day)) * self.boost

    def eta_min(self, dist_km, minute_of_day: float):
        """Planner's estimate (minutes) for straight-line distance(s) in km."""
        return np.asarray(dist_km) * self.circuity / self.speed_kmh(minute_of_day) * 60.0

    def actual_min(self, dist_km, minute_of_day: float, rng: np.random.Generator):
        est = self.eta_min(dist_km, minute_of_day)
        noise = rng.lognormal(-0.5 * self.sigma**2, self.sigma, size=np.shape(est))
        return est * noise


class CalibratedTraffic(TrafficModel):
    """Travel time fitted to Mapbox `duration_typical` data (see
    experiments/calibrate_traffic.py). Two functional forms, chosen by
    cross-validation on the training pairs:

        linear: car_min = slope(hour) * km + intercept
        power : car_min = exp(log_c(hour)) * km ** p

    ambulance_min = car_min / ambulance_boost. `ambulance_boost` is an
    ASSUMPTION (Mapbox gives car times); `sigma` is the held-out residual
    spread of the fit, not a measurement of live variability.
    """

    def __init__(self, params: dict, ambulance_boost: float = 1.25, sigma: float | None = None):
        self.form = params.get("form", "linear")
        self.hours = np.array(params["hours"], dtype=float)
        self.coef = np.array(params["coef"], dtype=float)  # slope or log_c per hour
        self.intercept = float(params.get("intercept_min", 0.0))
        self.power = float(params.get("power", 1.0))
        self.boost = ambulance_boost
        self.sigma = float(params["sigma"]) if sigma is None else sigma
        # periodic extension so interpolation wraps around midnight
        self._h = np.concatenate([self.hours - 24, self.hours, self.hours + 24])
        self._c = np.tile(self.coef, 3)

    def car_min(self, dist_km, minute_of_day: float):
        c = float(np.interp((minute_of_day / 60.0) % 24, self._h, self._c))
        d = np.asarray(dist_km, dtype=float)
        if self.form == "power":
            return np.exp(c) * np.maximum(d, 1e-3) ** self.power
        return c * d + self.intercept

    def eta_min(self, dist_km, minute_of_day: float):
        return self.car_min(dist_km, minute_of_day) / self.boost

    def speed_kmh(self, minute_of_day: float) -> float:  # for completeness
        return 60.0 * 5.0 / float(self.car_min(5.0, minute_of_day)) * self.boost


def load_calibrated(path=None, ambulance_boost: float = 1.25, sigma: float | None = None):
    import json
    from pathlib import Path

    p = Path(path) if path else Path(__file__).resolve().parent.parent / "results" / "traffic_calibration.json"
    return CalibratedTraffic(json.loads(p.read_text()), ambulance_boost, sigma)
