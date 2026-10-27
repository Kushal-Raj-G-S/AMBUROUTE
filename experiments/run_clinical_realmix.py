"""Sensitivity of the clinical study to the severity mix, using the share of dispatch priority levels in the
real NYC EMS log (data/real_nyc_ems_severity.csv): levels 1-2 -> critical, 3-4 -> urgent, 5+ -> stable.
Policy parameters are the ones tuned on validation seeds for the ORIGINAL assumed mix (not re-tuned).

    python experiments/run_clinical_realmix.py
Outputs: results/clinical_realmix.csv, results/clinical_realmix_tests.csv
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from amburoute.clinical import ClinicalHeuristic, ClinicalNash, NearestCapable, with_capabilities
from amburoute.geo import load_hospitals
from amburoute.nash import NashPolicy
from amburoute.policies import CapacityHeuristic, Nearest
from amburoute.sim import SCENARIOS, SimConfig, run_episode
from amburoute.traffic import load_calibrated

RES = ROOT / "results"
sev = pd.read_csv(ROOT / "data" / "real_nyc_ems_severity.csv").set_index("sev").n
mix = (sev.loc[1:2].sum(), sev.loc[3:4].sum(), sev.loc[5:].sum())
mix = tuple(round(float(x), 3) for x in np.array(mix[::-1]) / sum(mix))  # (stable, urgent, critical)
print("severity mix from real log (stable, urgent, critical):", mix)

base = json.loads((RES / "best_baseline_params.json").read_text())
cb = json.loads((RES / "best_clinical_params.json").read_text())
H, T = with_capabilities(load_hospitals()), load_calibrated()
CFG = SimConfig(clinical=True, sev_probs=mix)
SEEDS = list(range(1000, 1030))
pols = [Nearest(), NearestCapable(), CapacityHeuristic(base["capacity_heuristic"]["w"]),
        NashPolicy(base["nash"]["beta"], base["nash"]["power"]),
        ClinicalHeuristic(cb["clinical_heuristic"]["w"]), ClinicalNash(cb["clinical_nash"]["beta"], cb["clinical_nash"]["power"])]
rows = []
for sc in ("peak", "surge"):
    for p in pols:
        for s in SEEDS:
            r = run_episode(H, SCENARIOS[sc], s, p, CFG, T)
            rows.append(dict(scenario=sc, policy=p.name, seed=s, crit=r["crit_ttdc_mean"], spec=r["spec_ttdc_mean"],
                             surv=r["surv_mean"], rejections=r["rejections"], n_spec=r["n_spec"]))
raw = pd.DataFrame(rows)
summ = raw.groupby(["scenario", "policy"])[["crit", "spec", "surv", "rejections", "n_spec"]].mean().round(3)
summ.to_csv(RES / "clinical_realmix.csv")
pd.set_option("display.width", 200)
print(summ.to_string())

tests = []
for sc in ("peak", "surge"):
    for a, b in (("clinical_nash", "nearest_capable"), ("clinical_heuristic", "nearest_capable"),
                 ("clinical_nash", "nash"), ("clinical_nash", "clinical_heuristic")):
        da = raw[(raw.scenario == sc) & (raw.policy == a)].sort_values("seed").crit.to_numpy()
        db = raw[(raw.scenario == sc) & (raw.policy == b)].sort_values("seed").crit.to_numpy()
        d = da - db
        tests.append(dict(scenario=sc, a=a, b=b, mean_a=da.mean(), mean_b=db.mean(), rel_pct=100 * d.mean() / db.mean(),
                          p_raw=wilcoxon(d).pvalue))
t = pd.DataFrame(tests)
t["p_bonf"] = np.minimum(1, t.p_raw * len(t))
t.round(4).to_csv(RES / "clinical_realmix_tests.csv", index=False)
print(t.round(4).to_string(index=False))
