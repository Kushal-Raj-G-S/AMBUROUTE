"""The clinical study with capabilities taken from public sources instead of assumed
(data/hospital_capabilities_researched.csv: a service is counted only if a public source states it; evidence and
URLs are in the file; sources were read from search-result summaries and still need checking at the source).

Cardiac 4 hospitals, stroke 3, trauma 4 of 18. Real severity mix from the NYC log. 30 test seeds. Clinical policies
appear twice: with the parameters tuned for the assumed capabilities ("clinical_*") and re-tuned on validation seeds
for the researched capabilities ("*_retuned", results/best_clinical_params_researched.json).

    python experiments/run_clinical_researched.py
Outputs: results/clinical_researched.csv, results/clinical_researched_tests.csv
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from amburoute.clinical import ClinicalHeuristic, ClinicalNash, NearestCapable
from amburoute.geo import load_hospitals
from amburoute.nash import NashPolicy
from amburoute.policies import CapacityHeuristic, Nearest
from amburoute.sim import SCENARIOS, SimConfig, run_episode
from amburoute.traffic import load_calibrated

RES = ROOT / "results"
base = json.loads((RES / "best_baseline_params.json").read_text())
cb = json.loads((RES / "best_clinical_params.json").read_text())
cap = pd.read_csv(ROOT / "data" / "hospital_capabilities_researched.csv").set_index("id")
H = load_hospitals().copy()
for c in ("cardiac", "stroke", "trauma"):
    H["cap_" + c] = cap.loc[H["id"], c].to_numpy().astype(bool)
print("capable hospitals:", {c: int(H["cap_" + c].sum()) for c in ("cardiac", "stroke", "trauma")})
T = load_calibrated()
sev = pd.read_csv(ROOT / "data" / "real_nyc_ems_severity.csv").set_index("sev").n
mix = np.array([sev.loc[5:].sum(), sev.loc[3:4].sum(), sev.loc[1:2].sum()], dtype=float)
mix = tuple(round(float(x), 3) for x in mix / mix.sum())
cfg = SimConfig(clinical=True, sev_probs=mix)
SEEDS = list(range(1000, 1030))
rt = json.loads((RES / "best_clinical_params_researched.json").read_text())
h2, n2 = ClinicalHeuristic(rt["clinical_heuristic"]["w"]), ClinicalNash(rt["clinical_nash"]["beta"], rt["clinical_nash"]["power"])
h2.name, n2.name = "clinical_heuristic_retuned", "clinical_nash_retuned"
pols = [h2, n2, Nearest(), NearestCapable(), CapacityHeuristic(base["capacity_heuristic"]["w"]),
        NashPolicy(base["nash"]["beta"], base["nash"]["power"]),
        ClinicalHeuristic(cb["clinical_heuristic"]["w"]), ClinicalNash(cb["clinical_nash"]["beta"], cb["clinical_nash"]["power"])]
rows = []
for sc in ("peak", "surge"):
    for p in pols:
        for s in SEEDS:
            r = run_episode(H, SCENARIOS[sc], s, p, cfg, T)
            rows.append(dict(scenario=sc, policy=p.name, seed=s, crit=r["crit_ttdc_mean"], spec=r["spec_ttdc_mean"],
                             transfer=r["transfer_rate"], rejections=r["rejections"]))
raw = pd.DataFrame(rows)
summ = raw.groupby(["scenario", "policy"])[["crit", "spec", "transfer", "rejections"]].mean().round(3)
summ.to_csv(RES / "clinical_researched.csv")
pd.set_option("display.width", 200)
print(summ.to_string())
tests = []
for sc in ("peak", "surge"):
    g = lambda pol: raw[(raw.scenario == sc) & (raw.policy == pol)].sort_values("seed").crit.to_numpy()
    for a, b in (("clinical_nash_retuned", "nearest_capable"), ("clinical_heuristic_retuned", "nearest_capable"),
                 ("clinical_nash_retuned", "nash"), ("clinical_heuristic_retuned", "capacity_heuristic"),
                 ("clinical_nash_retuned", "clinical_heuristic_retuned"), ("clinical_heuristic_retuned", "clinical_heuristic"),
                 ("clinical_nash_retuned", "clinical_nash")):
        d = g(a) - g(b)
        tests.append(dict(scenario=sc, a=a, b=b, rel_pct=100 * d.mean() / g(b).mean(), p_raw=wilcoxon(d).pvalue))
t = pd.DataFrame(tests)
t["p_bonf"] = np.minimum(1, t.p_raw * len(t))
t.round(4).to_csv(RES / "clinical_researched_tests.csv", index=False)
print(t.round(4).to_string(index=False))
