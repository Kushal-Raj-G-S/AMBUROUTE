"""Realistic hospital capacity drawn from measured data.

`data/real_capacity_profile.csv` holds 14,202 anonymous hospital-weeks from the US federal facility-level capacity data
(size, general-bed occupancy, ICU share of beds, ICU occupancy; see experiments/fetch_hhs_capacity.py). For each hospital
of any city's table, `sample_capacity` picks one real hospital-week of similar size, so the simulator gets measured
occupancy and ICU shares that move with hospital size, instead of invented ones. Works for any table with a `beds` column.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_PROFILE: pd.DataFrame | None = None


def load_profile() -> pd.DataFrame:
    global _PROFILE
    if _PROFILE is None:
        _PROFILE = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "real_capacity_profile.csv")
    return _PROFILE


def sample_capacity(beds: np.ndarray, rng: np.random.Generator, k: int = 300):
    """For each hospital size, one real hospital-week among the `k` nearest in log size.
    Returns (general occupancy, ICU share of beds, ICU occupancy), each of length len(beds)."""
    p = load_profile()
    lb = np.log(p.beds.to_numpy())
    occ, share, iocc = [], [], []
    for b in np.asarray(beds, dtype=float):
        near = np.argpartition(np.abs(lb - np.log(max(b, 1.0))), k)[:k]
        j = int(rng.choice(near))
        occ.append(p.occ.iat[j])
        share.append(p.icu_share.iat[j])
        iocc.append(p.icu_occ.iat[j])
    return np.array(occ), np.array(share), np.array(iocc)
