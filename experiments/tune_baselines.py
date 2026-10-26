"""Tune the hand-designed policies on VALIDATION seeds (500-529), same budget
as PPO gets. Test seeds (1000+) are never touched here."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from amburoute.geo import load_hospitals
from amburoute.nash import NashPolicy
from amburoute.policies import CapacityHeuristic
from amburoute.sim import SCENARIOS, run_episode
from amburoute.traffic import load_calibrated

VAL_SEEDS = list(range(500, 530))
h = load_hospitals()
T = load_calibrated()
rows = []


def evaluate(make, label, **params):
    for sc in SCENARIOS.values():
        vals = [run_episode(h, sc, s, make(), traffic=T) for s in VAL_SEEDS]
        rows.append(dict(policy=label, scenario=sc.name, **params,
                         ttb_mean=np.mean([v["ttb_mean"] for v in vals]),
                         rejections=np.mean([v["rejections"] for v in vals])))


for w in (5, 10, 20, 30, 50, 80):
    evaluate(lambda w=w: CapacityHeuristic(w), "capacity_heuristic", w=w)
for beta in (5, 10, 20, 30, 45):
    for power in (1.0, 2.0):
        evaluate(lambda b=beta, p=power: NashPolicy(b, p), "nash", beta=beta, power=power)

df = pd.DataFrame(rows)
df.to_csv(ROOT / "results" / "tuning_baselines.csv", index=False)
best = {}
for pol, keys in (("capacity_heuristic", ["w"]), ("nash", ["beta", "power"])):
    d = df[df.policy == pol].groupby(keys).ttb_mean.mean().sort_values()
    print(pol, "\n", d.head(5).round(3).to_string())
    k = d.index[0]
    best[pol] = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
(ROOT / "results" / "best_baseline_params.json").write_text(json.dumps(best, default=float))
print(best)
