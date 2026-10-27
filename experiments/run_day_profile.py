"""A day in the life: call volume follows the real hourly profile of the NYC EMS log (data/real_nyc_ems_hourly.csv,
shape only, not the level) while travel times follow the calibrated Bengaluru hourly traffic model. Mean call rate is
an ASSUMPTION (0.85 per minute, between the paper's off-peak and peak settings).

For each start hour, a 2-hour window is simulated (30 seeds) under nearest, the tuned heuristic and tuned Nash.

    python experiments/run_day_profile.py
Outputs: results/day_profile.csv, docs/images/day_profile.png
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from amburoute.geo import load_hospitals
from amburoute.nash import NashPolicy
from amburoute.policies import CapacityHeuristic, Nearest
from amburoute.sim import Scenario, SimConfig, run_episode
from amburoute.traffic import load_calibrated

RES = ROOT / "results"
base = json.loads((RES / "best_baseline_params.json").read_text())
H, T = load_hospitals(), load_calibrated()
hourly = pd.read_csv(ROOT / "data" / "real_nyc_ems_hourly.csv").set_index("hr").n
rel = hourly / hourly.mean()  # 1.0 = average hour
MEAN_RATE = 0.85
pols = {"nearest": Nearest(), "heuristic": CapacityHeuristic(base["capacity_heuristic"]["w"]),
        "nash": NashPolicy(base["nash"]["beta"], base["nash"]["power"])}
rows = []
for hr in range(24):
    lam = MEAN_RATE * float(np.mean([rel[hr], rel[(hr + 1) % 24]]))
    sc = Scenario(f"h{hr}", lam, 120, hr * 60)
    for name, p in pols.items():
        r = [run_episode(H, sc, s, p, SimConfig(), T) for s in range(1000, 1030)]
        rows.append(dict(hour=hr, rate=lam, policy=name, ttb=np.mean([x["ttb_mean"] for x in r]),
                         rejections=np.mean([x["rejections"] for x in r])))
df = pd.DataFrame(rows).round(3)
df.to_csv(RES / "day_profile.csv", index=False)
pv = df.pivot(index="hour", columns="policy", values="ttb")
print(pv.round(2).to_string())
print("mean gain of heuristic over nearest across the day: %.1f %%" % (100 * (1 - pv.heuristic.mean() / pv.nearest.mean())))
print("worst hour for nearest:", int(pv.nearest.idxmax()), "gain there: %.1f %%" % (100 * (1 - pv.heuristic[pv.nearest.idxmax()] / pv.nearest.max())))

fig, ax = plt.subplots(figsize=(6.4, 3.4))
for name in pv:
    ax.plot(pv.index, pv[name], marker="o", ms=3, label=name)
ax.set_xlabel("Start hour of a 2-hour window")
ax.set_ylabel("Mean time to bed (min)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(ROOT / "docs" / "images" / "day_profile.png", dpi=200)
