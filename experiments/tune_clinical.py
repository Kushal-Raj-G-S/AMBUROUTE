"""Tune the clinical policies on VALIDATION seeds (500-529) so Nash and the heuristic get equal effort.
Objective: mean time to definitive care of CRITICAL patients, averaged over peak and surge.

    python experiments/tune_clinical.py     ->  results/best_clinical_params.json, results/tuning_clinical.csv
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from amburoute.clinical import ClinicalHeuristic, ClinicalNash, with_capabilities
from amburoute.geo import load_hospitals
from amburoute.sim import SCENARIOS, SimConfig, run_episode
from amburoute.traffic import load_calibrated

H = with_capabilities(load_hospitals())
T = load_calibrated()
CFG = SimConfig(clinical=True)
VAL = list(range(500, 530))
rows = []


def score(make, label, **params):
    vals = []
    for sc in ("peak", "surge"):
        vals.append(np.mean([run_episode(H, SCENARIOS[sc], s, make(), CFG, T)["crit_ttdc_mean"] for s in VAL]))
    rows.append(dict(policy=label, **params, crit_ttdc_mean=float(np.mean(vals))))


for w in (5, 10, 20, 40, 80):
    score(lambda w=w: ClinicalHeuristic(w), "clinical_heuristic", w=w)
for beta in (5, 10, 20, 40, 80):
    for power in (1.0, 2.0):
        score(lambda b=beta, p=power: ClinicalNash(b, p), "clinical_nash", beta=beta, power=power)

df = pd.DataFrame(rows)
df.to_csv(ROOT / "results" / "tuning_clinical.csv", index=False)
best = {}
for pol, keys in (("clinical_heuristic", ["w"]), ("clinical_nash", ["beta", "power"])):
    d = df[df.policy == pol].sort_values("crit_ttdc_mean")
    print(d.head(4).round(3).to_string(index=False))
    r = d.iloc[0]
    best[pol] = {k: float(r[k]) for k in keys}
(ROOT / "results" / "best_clinical_params.json").write_text(json.dumps(best))
print(best)
