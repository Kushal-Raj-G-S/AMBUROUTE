"""Do the headline results survive realistic call locations? Calls follow OSM building density instead of being
uniform over the map. Policy parameters are the paper's (tuned for uniform calls, NOT re-tuned); the PPO policy,
trained on uniform calls, is also scored out of distribution.

    python experiments/run_spatial_demand.py
Outputs: results/spatial_demand.csv, results/spatial_demand_tests.csv
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
from amburoute.rl_env import PPOPolicy
from amburoute.sim import SCENARIOS, SimConfig, run_episode
from amburoute.traffic import load_calibrated

RES = ROOT / "results"
best = json.loads((RES / "best_baseline_params.json").read_text())
H, T = load_hospitals(), load_calibrated()
SEEDS = list(range(1000, 1050))
pols = {"nearest": Nearest(), "heuristic": CapacityHeuristic(best["capacity_heuristic"]["w"]),
        "nash": NashPolicy(best["nash"]["beta"], best["nash"]["power"]),
        "ppo (trained on uniform calls)": PPOPolicy(RES / "models" / "B_s1.pt", name="ppo")}
rows = []
for demand, cfg in (("uniform", SimConfig()), ("building density", SimConfig(spatial_demand=True))):
    for sc in ("offpeak", "peak", "surge"):
        for name, p in pols.items():
            for s in SEEDS:
                r = run_episode(H, SCENARIOS[sc], s, p, cfg, T)
                rows.append(dict(demand=demand, scenario=sc, policy=name, seed=s, ttb=r["ttb_mean"],
                                 rejections=r["rejections"], jain=r["jain_load_balance"]))
    print("done", demand, flush=True)
raw = pd.DataFrame(rows)
summ = raw.groupby(["demand", "scenario", "policy"])[["ttb", "rejections", "jain"]].mean().round(3)
summ.to_csv(RES / "spatial_demand.csv")
pd.set_option("display.width", 200)
print(summ.to_string())
tests = []
for sc in ("peak", "surge"):
    g = lambda pol: raw[(raw.demand == "building density") & (raw.scenario == sc) & (raw.policy == pol)].sort_values("seed").ttb.to_numpy()
    for a, b in (("heuristic", "nearest"), ("nash", "nearest"), ("nash", "heuristic"), ("ppo (trained on uniform calls)", "heuristic")):
        d = g(a) - g(b)
        tests.append(dict(scenario=sc, a=a, b=b, rel_pct=100 * d.mean() / g(b).mean(), p_raw=wilcoxon(d).pvalue))
t = pd.DataFrame(tests)
t["p_bonf"] = np.minimum(1, t.p_raw * len(t))
t.round(4).to_csv(RES / "spatial_demand_tests.csv", index=False)
print(t.round(4).to_string(index=False))
