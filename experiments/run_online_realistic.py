"""Online adaptation under jam sizes taken from REAL day-to-day variation (results/gmaps_variability.json), instead of the
2x jams assumed in run_online.py. In the real Google-Maps readings, 95 % of readings are within 1.20x of the typical
time for that route and hour (1.24x at weekday rush hours), 99 % within 1.39x (1.48x at rush hour), and only 0.04 % reach 2x.

Drift cases (from minute 30): four random hospitals 1.25x slower to reach; four hospitals 1.5x slower; every trip 1.24x slower.
Same learner as run_online.py (alpha 0.3, paper's tuned heuristic), 50 test seeds.

    python experiments/run_online_realistic.py
Output: results/online_realistic.csv
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from amburoute.geo import load_hospitals
from amburoute.online import OnlineETA, OracleETA
from amburoute.policies import CapacityHeuristic
from amburoute.sim import SCENARIOS, SimConfig, run_episode
from amburoute.traffic import load_calibrated

w = json.loads((ROOT / "results" / "best_baseline_params.json").read_text())["capacity_heuristic"]["w"]
H, T = load_hospitals(), load_calibrated()
CASES = {"local jams 1.25x (4 hospitals)": SimConfig(drift_n=4, drift_factor=1.25),
         "local jams 1.5x (4 hospitals)": SimConfig(drift_n=4, drift_factor=1.5),
         "citywide slowdown 1.24x": SimConfig(drift_n=-1, drift_factor=1.24)}
rows = []
for name, cfg in CASES.items():
    for sc in ("peak", "surge"):
        for kind in ("fixed model", "online (learned)", "oracle"):
            pol = {"fixed model": CapacityHeuristic(w), "online (learned)": OnlineETA(CapacityHeuristic(w), 0.3),
                   "oracle": OracleETA(CapacityHeuristic(w))}[kind]
            for s in range(1000, 1050):
                rows.append(dict(case=name, scenario=sc, variant=kind, seed=s, ttb=run_episode(H, SCENARIOS[sc], s, pol, cfg, T)["ttb_mean"]))
    print("done", name, flush=True)
raw = pd.DataFrame(rows)
out = []
for (c, sc), g in raw.groupby(["case", "scenario"]):
    fixed = g[g.variant == "fixed model"].sort_values("seed").ttb.to_numpy()
    for v, gg in g.groupby("variant"):
        x = gg.sort_values("seed").ttb.to_numpy()
        d = x - fixed
        out.append(dict(case=c, scenario=sc, variant=v, ttb=x.mean(), vs_fixed_pct=100 * d.mean() / fixed.mean(),
                        p_raw=wilcoxon(d).pvalue if v != "fixed model" and np.any(d) else np.nan))
res = pd.DataFrame(out)
k = res.p_raw.notna().sum()
res["p_bonf"] = np.minimum(1, res.p_raw * k)
res.round(4).to_csv(ROOT / "results" / "online_realistic.csv", index=False)
pd.set_option("display.width", 200)
print(res.round(3).to_string(index=False))
