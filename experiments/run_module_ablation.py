"""Module ablation: what does each part of the system contribute?

Full system = calibrated travel times + in-flight load + congestion game (Nash).
Each variant removes one module and is run on the same TEST seeds (1000-1049), with the
policy parameters tuned for the full system (nothing is re-tuned for the ablated variants).

    python experiments/run_module_ablation.py
Outputs: results/module_ablation.csv, results/module_ablation_raw.csv, results/module_ablation_tests.csv
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
from amburoute.nash import NashPolicy
from amburoute.policies import CapacityHeuristic, Nearest
from amburoute.sim import SCENARIOS, run_episode
from amburoute.traffic import TrafficModel, load_calibrated

SEEDS = list(range(1000, 1050))
best = json.loads((ROOT / "results" / "best_baseline_params.json").read_text())
w, beta, gam = best["capacity_heuristic"]["w"], best["nash"]["beta"], best["nash"]["power"]
H = load_hospitals()
T = load_calibrated()
RAW_TRAFFIC = TrafficModel()  # the first prototype's hand-set model, used as the planner without calibration

# name -> (policy factory, planner traffic)
VARIANTS = {
    "full (calibration + in-flight + game)": (lambda: NashPolicy(beta, gam), None),
    "no congestion game (heuristic instead)": (lambda: CapacityHeuristic(w), None),
    "no in-flight load": (lambda: NashPolicy(beta, gam, use_inflight=False), None),
    "no traffic calibration": (lambda: NashPolicy(beta, gam), RAW_TRAFFIC),
    "none of the three (nearest hospital)": (lambda: Nearest(), RAW_TRAFFIC),
}

rows = []
for sc in ("peak", "surge"):
    for name, (mk, plan) in VARIANTS.items():
        pol = mk()
        for s in SEEDS:
            r = run_episode(H, SCENARIOS[sc], s, pol, traffic=T, planner_traffic=plan)
            rows.append(dict(scenario=sc, variant=name, seed=s, ttb_mean=r["ttb_mean"], rejections=r["rejections"]))
    print("done", sc, flush=True)
raw = pd.DataFrame(rows)
raw.to_csv(ROOT / "results" / "module_ablation_raw.csv", index=False)

full = next(iter(VARIANTS))
tests, summ = [], []
for sc in ("peak", "surge"):
    a = raw[(raw.scenario == sc) & (raw.variant == full)].sort_values("seed")
    for name in VARIANTS:
        b = raw[(raw.scenario == sc) & (raw.variant == name)].sort_values("seed")
        d = b.ttb_mean.to_numpy() - a.ttb_mean.to_numpy()
        p = wilcoxon(d).pvalue if np.any(d) else 1.0
        summ.append(dict(scenario=sc, variant=name, ttb_mean=b.ttb_mean.mean(), rejections=b.rejections.mean(),
                         delta_min=d.mean(), delta_pct=100 * d.mean() / a.ttb_mean.mean(), p_raw=p))
t = pd.DataFrame(summ)
mask = t.variant != full
t.loc[mask, "p_holm"] = np.minimum(1, t.loc[mask, "p_raw"] * mask.sum() / 2)  # Bonferroni over the variants in each scenario
t.round(4).to_csv(ROOT / "results" / "module_ablation.csv", index=False)
pd.set_option("display.width", 200)
print(t.round(3).to_string(index=False))
