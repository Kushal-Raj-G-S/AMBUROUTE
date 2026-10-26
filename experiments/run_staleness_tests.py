"""Paired tests for the bed-data staleness ablation (test seeds 1000-1049)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from amburoute.geo import load_hospitals
from amburoute.nash import NashPolicy
from amburoute.policies import CapacityHeuristic
from amburoute.sim import SCENARIOS, SimConfig, run_episode
from amburoute.traffic import load_calibrated

best = json.loads((ROOT / "results" / "best_baseline_params.json").read_text())
h, T = load_hospitals(), load_calibrated()
SEEDS = list(range(1000, 1050))
cap, nash = CapacityHeuristic(best["capacity_heuristic"]["w"]), NashPolicy(best["nash"]["beta"], best["nash"]["power"])
rows, raw = [], {}
for sc in ("peak", "surge"):
    for ri in (1, 5, 15, 30):
        cfg = SimConfig(report_interval=ri)
        for name, p in (("capacity_heuristic", cap), ("nash", nash)):
            raw[(sc, ri, name)] = np.array([run_episode(h, SCENARIOS[sc], s, p, cfg, T)["ttb_mean"] for s in SEEDS])
        d = raw[(sc, ri, "nash")] - raw[(sc, ri, "capacity_heuristic")]
        rows.append(dict(scenario=sc, refresh_min=ri, test="nash - heuristic", mean_diff=d.mean(),
                         p_raw=wilcoxon(d).pvalue if np.any(d) else 1.0))
    for name in ("capacity_heuristic", "nash"):
        d = raw[(sc, 30, name)] - raw[(sc, 1, name)]
        rows.append(dict(scenario=sc, refresh_min=30, test=f"{name}: 30min - 1min", mean_diff=d.mean(),
                         p_raw=wilcoxon(d).pvalue))
df = pd.DataFrame(rows)
df["p_holm"] = np.minimum(1, df.p_raw * len(df))  # conservative Bonferroni over all 12
df.round(4).to_csv(ROOT / "results" / "staleness_tests.csv", index=False)
print(df.round(4).to_string(index=False))
