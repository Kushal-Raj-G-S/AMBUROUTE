"""Tune the clinical policies on VALIDATION seeds (500-529) with the RESEARCHED hospital capabilities
(data/hospital_capabilities_researched.csv) and the real NYC severity mix, so Nash and the heuristic get equal effort.
The grids extend beyond the ones used for the assumed capabilities, whose best weight sat at the grid edge.
Objective: mean time to definitive care of CRITICAL patients, averaged over peak and surge.

    python experiments/tune_clinical_researched.py
        ->  results/best_clinical_params_researched.json, results/tuning_clinical_researched.csv
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from amburoute.clinical import ClinicalHeuristic, ClinicalNash
from amburoute.geo import load_hospitals
from amburoute.sim import SCENARIOS, SimConfig, run_episode
from amburoute.traffic import load_calibrated

H = load_hospitals().copy()
cap = pd.read_csv(ROOT / "data" / "hospital_capabilities_researched.csv").set_index("id")
for c in ("cardiac", "stroke", "trauma"):
    H["cap_" + c] = cap.loc[H["id"], c].to_numpy().astype(bool)
T = load_calibrated()
sev = pd.read_csv(ROOT / "data" / "real_nyc_ems_severity.csv").set_index("sev").n
mix = np.array([sev.loc[5:].sum(), sev.loc[3:4].sum(), sev.loc[1:2].sum()], dtype=float)
CFG = SimConfig(clinical=True, sev_probs=tuple(round(float(x), 3) for x in mix / mix.sum()))
VAL = list(range(500, 530))
rows = []


def score(make, label, **params):
    vals = []
    for sc in ("peak", "surge"):
        vals.append(np.mean([run_episode(H, SCENARIOS[sc], s, make(), CFG, T)["crit_ttdc_mean"] for s in VAL]))
    rows.append(dict(policy=label, **params, crit_ttdc_mean=float(np.mean(vals))))


for w in (10, 20, 40, 80, 160, 320):
    score(lambda w=w: ClinicalHeuristic(w), "clinical_heuristic", w=w)
for beta in (10, 20, 40, 80, 160):
    for power in (1.0, 2.0):
        score(lambda b=beta, p=power: ClinicalNash(b, p), "clinical_nash", beta=beta, power=power)

df = pd.DataFrame(rows)
df.to_csv(ROOT / "results" / "tuning_clinical_researched.csv", index=False)
best = {}
for pol, keys in (("clinical_heuristic", ["w"]), ("clinical_nash", ["beta", "power"])):
    d = df[df.policy == pol].sort_values("crit_ttdc_mean")
    print(d.head(4).round(3).to_string(index=False))
    r = d.iloc[0]
    best[pol] = {k: float(r[k]) for k in keys}
(ROOT / "results" / "best_clinical_params_researched.json").write_text(json.dumps(best))
print(best)
