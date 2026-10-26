"""Paired comparison of the selected PPO policy against the prototype rule (test seeds, from final_raw.csv)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

raw = pd.read_csv(ROOT / "results" / "final_raw.csv")
rows = []
for sc in ("offpeak", "peak", "surge"):
    a = raw[(raw.scenario == sc) & (raw.policy == "ppo")].sort_values("seed").ttb_mean.to_numpy()
    b = raw[(raw.scenario == sc) & (raw.policy == "prototype_heuristic")].sort_values("seed").ttb_mean.to_numpy()
    d = a - b
    p = wilcoxon(d).pvalue
    rows.append(dict(scenario=sc, mean_diff=d.mean(), seeds_ppo_better=int((d < 0).sum()), n=len(d),
                     p_raw=p, p_bonferroni3=min(1.0, 3 * p)))
out = pd.DataFrame(rows).round(4)
out.to_csv(ROOT / "results" / "ppo_vs_prototype.csv", index=False)
print(out.to_string(index=False))
