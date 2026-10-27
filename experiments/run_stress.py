"""Stress tests: (A) call-rate sweep, (B) everything going wrong at once for the clinical policies.

(A) call rate 0.4-3.5 per minute, evening traffic, paper's tuned policies.
(B) clinical mode + bed reports every 15 min + local traffic jams (4 hospitals x2 from minute 30), with and without
    the online ETA correction. Parameters are the ones tuned on validation seeds for the clean setting.

    python experiments/run_stress.py
Outputs: results/stress_rate_sweep.csv, results/stress_combined.csv, results/stress_combined_tests.csv, docs/images/stress.png
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from amburoute.clinical import ClinicalHeuristic, ClinicalNash, NearestCapable, with_capabilities
from amburoute.geo import load_hospitals
from amburoute.nash import NashPolicy
from amburoute.online import OnlineETA
from amburoute.policies import CapacityHeuristic, Nearest
from amburoute.sim import SCENARIOS, Scenario, SimConfig, run_episode
from amburoute.traffic import load_calibrated

RES = ROOT / "results"
base = json.loads((RES / "best_baseline_params.json").read_text())
cb = json.loads((RES / "best_clinical_params.json").read_text())
H0, T = load_hospitals(), load_calibrated()
SEEDS = list(range(1000, 1030))

# (A) rate sweep
rows = []
pols = {"nearest": Nearest(), "heuristic": CapacityHeuristic(base["capacity_heuristic"]["w"]),
        "nash": NashPolicy(base["nash"]["beta"], base["nash"]["power"])}
for lam in (0.4, 0.8, 1.2, 1.8, 2.5, 3.5):
    sc = Scenario(f"rate{lam}", lam, 90, 17 * 60 + 30)
    for name, p in pols.items():
        r = [run_episode(H0, sc, s, p, SimConfig(), T) for s in SEEDS]
        rows.append(dict(rate=lam, policy=name, ttb=np.mean([x["ttb_mean"] for x in r]),
                         rejections=np.mean([x["rejections"] for x in r])))
    print("rate", lam, flush=True)
sweep = pd.DataFrame(rows).round(3)
sweep.to_csv(RES / "stress_rate_sweep.csv", index=False)
print(sweep.pivot(index="rate", columns="policy", values="ttb").to_string())

# (B) combined stress
H = with_capabilities(H0)
cfg = SimConfig(clinical=True, report_interval=15, drift_n=4, drift_factor=2.0)
mk = {"nearest capable": lambda: NearestCapable(),
      "clinical heuristic": lambda: ClinicalHeuristic(cb["clinical_heuristic"]["w"]),
      "clinical Nash": lambda: ClinicalNash(cb["clinical_nash"]["beta"], cb["clinical_nash"]["power"]),
      "clinical heuristic + online ETA": lambda: OnlineETA(ClinicalHeuristic(cb["clinical_heuristic"]["w"]), 0.3),
      "clinical Nash + online ETA": lambda: OnlineETA(ClinicalNash(cb["clinical_nash"]["beta"], cb["clinical_nash"]["power"]), 0.3)}
rows = []
for sc in ("peak", "surge"):
    for name, f in mk.items():
        p = f()
        for s in SEEDS:
            r = run_episode(H, SCENARIOS[sc], s, p, cfg, T)
            rows.append(dict(scenario=sc, policy=name, seed=s, crit=r["crit_ttdc_mean"], spec=r["spec_ttdc_mean"],
                             surv=r["surv_mean"], rejections=r["rejections"]))
    print("combined", sc, flush=True)
raw = pd.DataFrame(rows)
summ = raw.groupby(["scenario", "policy"])[["crit", "spec", "surv", "rejections"]].mean().round(3)
summ.to_csv(RES / "stress_combined.csv")
pd.set_option("display.width", 200)
print(summ.to_string())
tests = []
for sc in ("peak", "surge"):
    g = lambda pol: raw[(raw.scenario == sc) & (raw.policy == pol)].sort_values("seed").crit.to_numpy()
    for a, b in (("clinical heuristic", "nearest capable"), ("clinical Nash", "nearest capable"),
                 ("clinical heuristic + online ETA", "clinical heuristic"), ("clinical Nash + online ETA", "clinical Nash")):
        d = g(a) - g(b)
        tests.append(dict(scenario=sc, a=a, b=b, rel_pct=100 * d.mean() / g(b).mean(), p_raw=wilcoxon(d).pvalue))
t = pd.DataFrame(tests)
t["p_bonf"] = np.minimum(1, t.p_raw * len(t))
t.round(4).to_csv(RES / "stress_combined_tests.csv", index=False)
print(t.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(5.4, 3.5))
for name, g in sweep.groupby("policy"):
    ax.plot(g.rate, g.ttb, marker="o", label=name)
ax.set_xlabel("Ambulance calls per minute (evening traffic)")
ax.set_ylabel("Mean time to bed (min)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(ROOT / "docs" / "images" / "stress.png", dpi=200)
