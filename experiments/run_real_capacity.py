"""Bengaluru map and hospitals with MEASURED capacity: occupancy and ICU share drawn from real hospitals of similar size
(data/real_capacity_profile.csv, US federal facility data), instead of the paper's assumed 88-99 % occupancy for every
hospital. Each seed draws its own capacities, so the result does not rest on one arbitrary assignment.

1. Policies keep the paper's tuned parameters (robustness check) and are also re-tuned on VALIDATION seeds (500-529).
2. Test seeds 1000-1049, scored once, three scenarios. PPO (trained on the assumed capacities) is run without retraining.
3. Episodes are split into terciles by how much general capacity is free at the start, to show how the benefit depends on
   capacity pressure.

    python experiments/run_real_capacity.py [2023-24 | 2022 | england-2026]
Outputs (suffix _<profile> unless 2023-24): results/real_capacity_tuning.csv, results/best_real_capacity_params.json,
         results/real_capacity_summary.csv, results/real_capacity_tests.csv, results/real_capacity_pressure.csv
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
H, T = load_hospitals(), load_calibrated()
PROFILE = sys.argv[1] if len(sys.argv) > 1 else "2023-24"
SUF = "" if PROFILE == "2023-24" else "_" + PROFILE
CFG = SimConfig(real_capacity=PROFILE)
VAL, TEST = list(range(500, 530)), list(range(1000, 1050))
base = json.loads((RES / "best_baseline_params.json").read_text())
w0, b0, g0 = base["capacity_heuristic"]["w"], base["nash"]["beta"], base["nash"]["power"]


def val_score(make):
    return float(np.mean([run_episode(H, SCENARIOS[sc], s, make(), CFG, T)["ttb_mean"] for sc in ("peak", "surge") for s in VAL]))


rows = [dict(policy="heuristic", w=w, beta=np.nan, power=np.nan, ttb=val_score(lambda w=w: CapacityHeuristic(w))) for w in (5, 10, 20, 40, 80)]
rows += [dict(policy="nash", w=np.nan, beta=b, power=p, ttb=val_score(lambda b=b, p=p: NashPolicy(b, p)))
         for b in (10, 20, 40) for p in (1.0, 2.0)]
tune = pd.DataFrame(rows)
tune.round(4).to_csv(RES / f"real_capacity_tuning{SUF}.csv", index=False)
bh = tune[tune.policy == "heuristic"].sort_values("ttb").iloc[0]
bn = tune[tune.policy == "nash"].sort_values("ttb").iloc[0]
best = {"heuristic": {"w": float(bh.w)}, "nash": {"beta": float(bn.beta), "power": float(bn.power)}}
(RES / f"best_real_capacity_params{SUF}.json").write_text(json.dumps(best))
print("re-tuned:", best, "| paper's:", dict(w=w0, beta=b0, power=g0), flush=True)

h2, n2 = CapacityHeuristic(best["heuristic"]["w"]), NashPolicy(best["nash"]["beta"], best["nash"]["power"])
h2.name, n2.name = "heuristic_retuned", "nash_retuned"
pols = [Nearest(), CapacityHeuristic(w0), NashPolicy(b0, g0), h2, n2, PPOPolicy(RES / "models" / "B_s1.pt", name="ppo")]
raw = []
for sc in ("offpeak", "peak", "surge"):
    for p in pols:
        for s in TEST:
            r = run_episode(H, SCENARIOS[sc], s, p, CFG, T)
            raw.append(dict(scenario=sc, policy=p.name, seed=s, ttb=r["ttb_mean"], rej=r["rejections"], free0=r["free_frac0"]))
    print("done", sc, flush=True)
raw = pd.DataFrame(raw)
summ = raw.groupby(["scenario", "policy"])[["ttb", "rej", "free0"]].mean().round(3)
summ.to_csv(RES / f"real_capacity_summary{SUF}.csv")
pd.set_option("display.width", 200)
print(summ.to_string())

tests = []
for sc in ("offpeak", "peak", "surge"):
    g = lambda pol: raw[(raw.scenario == sc) & (raw.policy == pol)].sort_values("seed").ttb.to_numpy()
    for a, b in (("capacity_heuristic", "nearest"), ("nash", "nearest"), ("nash", "capacity_heuristic"),
                 ("heuristic_retuned", "capacity_heuristic"), ("ppo", "capacity_heuristic")):
        d = g(a) - g(b)
        tests.append(dict(scenario=sc, a=a, b=b, rel_pct=100 * d.mean() / g(b).mean(), p_raw=wilcoxon(d).pvalue if np.any(d) else 1.0))
t = pd.DataFrame(tests)
t["p_bonf"] = np.minimum(1, t.p_raw * len(t))
t.round(4).to_csv(RES / f"real_capacity_tests{SUF}.csv", index=False)
print(t.round(4).to_string(index=False))

# benefit against capacity pressure: terciles of free capacity at the start of each episode
press = []
for sc in ("peak", "surge"):
    n = raw[(raw.scenario == sc) & (raw.policy == "nearest")].set_index("seed")
    h = raw[(raw.scenario == sc) & (raw.policy == "capacity_heuristic")].set_index("seed")
    df = pd.DataFrame({"free0": n.free0, "nearest": n.ttb, "heuristic": h.ttb, "rej_n": n.rej, "rej_h": h.rej})
    df["pressure"] = pd.qcut(df.free0, 3, labels=["tight", "medium", "loose"])
    for lab, g in df.groupby("pressure", observed=True):
        press.append(dict(scenario=sc, pressure=lab, free_capacity=round(g.free0.mean(), 3), nearest=g.nearest.mean(),
                          heuristic=g.heuristic.mean(), gain_pct=100 * (1 - g.heuristic.mean() / g.nearest.mean()),
                          rejections_nearest=g.rej_n.mean(), rejections_heuristic=g.rej_h.mean()))
pr = pd.DataFrame(press).round(3)
pr.to_csv(RES / f"real_capacity_pressure{SUF}.csv", index=False)
print(pr.to_string(index=False))
