"""Realistic hospital capacity drawn from measured data.

Three anonymous profiles are available (see docs/additional_studies.md, section 11):
  "2023-24"      14,202 US hospital-weeks, facility-level federal data (data/real_capacity_profile.csv)
  "2022"         14,955 US hospital-weeks (data/real_capacity_profile_2022.csv)
  "england-2026" 124 English acute hospital trusts, six-month means April-September 2026 (data/real_capacity_profile_england_2026.csv)
Each holds size, general-bed occupancy, ICU share of beds and ICU occupancy. For each hospital of any city's table,
`sample_capacity` picks one real record of similar size, so the simulator gets measured occupancy and ICU shares that
move with hospital size, instead of invented ones. Works for any table with a `beds` column.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROFILES = {
    "2023-24": "real_capacity_profile.csv",
    "2022": "real_capacity_profile_2022.csv",
    "england-2026": "real_capacity_profile_england_2026.csv",
}
_CACHE: dict[str, pd.DataFrame] = {}


def load_profile(name: str | bool = "2023-24") -> pd.DataFrame:
    key = "2023-24" if name is True else str(name)
    if key not in PROFILES:
        raise ValueError(f"unknown capacity profile {key!r}; choose from {list(PROFILES)}")
    if key not in _CACHE:
        _CACHE[key] = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / PROFILES[key])
    return _CACHE[key]


def sample_capacity(beds: np.ndarray, rng: np.random.Generator, profile: str | bool = "2023-24", k: int = 300):
    """For each hospital size, one real record among the `k` nearest in log size (fewer for a small profile).
    Returns (general occupancy, ICU share of beds, ICU occupancy), each of length len(beds)."""
    p = load_profile(profile)
    k = min(k, max(10, len(p) // 4))
    lb = np.log(p.beds.to_numpy())
    occ, share, iocc = [], [], []
    for b in np.asarray(beds, dtype=float):
        near = np.argpartition(np.abs(lb - np.log(max(b, 1.0))), k)[:k]
        j = int(rng.choice(near))
        occ.append(p.occ.iat[j])
        share.append(p.icu_share.iat[j])
        iocc.append(p.icu_occ.iat[j])
    return np.array(occ), np.array(share), np.array(iocc)
