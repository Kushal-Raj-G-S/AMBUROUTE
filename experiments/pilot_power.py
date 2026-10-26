"""Rough sample-size estimate for a paired pilot, using the simulator's day-to-day variability.

A paired design compares the recommended policy and the baseline on the SAME days. The number of
days needed to detect a mean difference `delta` (two-sided alpha=0.05, power=0.80) is
n = ((z_a + z_b) * sd_diff / delta)^2, where sd_diff is the standard deviation of the paired daily
difference. Variances here come from SIMULATION and must be re-estimated from real pilot data.

    python experiments/pilot_power.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
raw = pd.read_csv(ROOT / "results" / "final_raw.csv")
Z = 1.959964 + 0.841621  # z_(1-alpha/2) + z_(power)
rows = []
for sc in ("offpeak", "peak", "surge"):
    for a, b in (("nash", "nearest"), ("nash", "capacity_heuristic")):
        da = raw[(raw.scenario == sc) & (raw.policy == a)].sort_values("seed").ttb_mean.to_numpy()
        db = raw[(raw.scenario == sc) & (raw.policy == b)].sort_values("seed").ttb_mean.to_numpy()
        d = da - db
        delta = abs(d.mean())
        sd = d.std(ddof=1)
        n = int(np.ceil((Z * sd / delta) ** 2)) if delta > 0 else None
        rows.append(dict(scenario=sc, comparison=f"{a} vs {b}", mean_diff_min=round(d.mean(), 2),
                         sd_daily_diff=round(sd, 2), days_needed=n))
out = pd.DataFrame(rows)
out.to_csv(ROOT / "results" / "pilot_power.csv", index=False)
print(out.to_string(index=False))
