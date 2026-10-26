"""Clinical extension: severity and specialty (cardiac, stroke, trauma). ALL clinical inputs are assumed.

    python experiments/run_clinical.py
Outputs: data/hospital_capabilities.csv, results/clinical_summary.csv, results/clinical_tests.csv,
         docs/images/clinical.png
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
from amburoute.policies import CapacityHeuristic, Nearest
from amburoute.sim import SCENARIOS, SimConfig, run_episode
from amburoute.traffic import load_calibrated

RES = ROOT / "results"
SEEDS = list(range(1000, 1030))
best = json.loads((RES / "best_baseline_params.json").read_text())
w, beta, gam = best["capacity_heuristic"]["w"], best["nash"]["beta"], best["nash"]["power"]
H = with_capabilities(load_hospitals())
H.to_csv(ROOT / "data" / "hospital_capabilities.csv", index=False)
T = load_calibrated()
CFG = SimConfig(clinical=True)
LABEL = {"nearest": "Nearest", "nearest_capable": "Nearest capable", "capacity_heuristic": "Capacity heuristic (specialty-blind)",
         "nash": "Nash (specialty-blind)", "clinical_heuristic": "Clinical heuristic", "clinical_nash": "Clinical Nash"}


cb = json.loads((RES / "best_clinical_params.json").read_text())  # tuned on validation seeds
cw, cbeta, cgam = cb["clinical_heuristic"]["w"], cb["clinical_nash"]["beta"], cb["clinical_nash"]["power"]
ONES = np.ones(3)


def pols():
    ablation = [ClinicalHeuristic(cw, ONES), ClinicalNash(cbeta, cgam, ONES)]
    ablation[0].name, ablation[1].name = "clinical_heuristic_unweighted", "clinical_nash_unweighted"
    return [Nearest(), NearestCapable(), CapacityHeuristic(w), NashPolicy(beta, gam),
            ClinicalHeuristic(cw), ClinicalNash(cbeta, cgam)] + ablation


METRICS = ["ttb_mean", "crit_ttdc_mean", "crit_ttdc_p95", "spec_ttdc_mean", "transfer_rate", "rejections"]
rows = []
for sc in ("peak", "surge"):
    for p in pols():
        for s in SEEDS:
            rows.append(dict(scenario=sc, policy=p.name, seed=s, **run_episode(H, SCENARIOS[sc], s, p, CFG, T)))
    print("done", sc, flush=True)
raw = pd.DataFrame(rows)
raw.to_csv(RES / "clinical_raw.csv", index=False)
summ = raw.groupby(["scenario", "policy"])[METRICS].mean().round(3)
summ.to_csv(RES / "clinical_summary.csv")

pairs = [("clinical_nash", "nearest"), ("clinical_nash", "nearest_capable"), ("clinical_nash", "nash"),
         ("clinical_heuristic", "capacity_heuristic"), ("clinical_nash", "clinical_heuristic"),
         ("clinical_heuristic", "clinical_heuristic_unweighted"), ("clinical_nash", "clinical_nash_unweighted")]
tests = []
for sc in ("peak", "surge"):
    for m in ("crit_ttdc_mean", "spec_ttdc_mean", "ttb_mean"):
        for a, b in pairs:
            da = raw[(raw.scenario == sc) & (raw.policy == a)].sort_values("seed")[m].to_numpy()
            db = raw[(raw.scenario == sc) & (raw.policy == b)].sort_values("seed")[m].to_numpy()
            d = da - db
            tests.append(dict(scenario=sc, metric=m, a=a, b=b, mean_a=da.mean(), mean_b=db.mean(), mean_diff=d.mean(),
                              rel_pct=100 * d.mean() / db.mean(), p_raw=wilcoxon(d).pvalue if np.any(d) else 1.0))
t = pd.DataFrame(tests)
t["p_holm_over_all"] = np.minimum(1, t.p_raw * len(t))  # conservative Bonferroni over every test shown
t.round(4).to_csv(RES / "clinical_tests.csv", index=False)

pd.set_option("display.width", 220)
print(summ.round(2).to_string())

order = [k for k in LABEL if k in raw.policy.unique()]
cols = {"nearest": "#999999", "nearest_capable": "#C4A052", "capacity_heuristic": "#4C72B0", "nash": "#8172B3",
        "clinical_heuristic": "#55A868", "clinical_nash": "#2f9e6b"}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, sc in zip(axes, ("peak", "surge")):
    x = np.arange(2)
    for i, k in enumerate(order):
        vals = [summ.loc[(sc, k), "crit_ttdc_mean"], summ.loc[(sc, k), "spec_ttdc_mean"]]
        ax.bar(x + (i - 2.5) * 0.14, vals, 0.14, label=LABEL[k], color=cols[k])
    ax.set_xticks(x, ["Critical patients", "Specialty patients"])
    ax.set_title("Evening peak" if sc == "peak" else "Surge")
axes[0].set_ylabel("Mean time to definitive care (min)")
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8, frameon=False)
fig.tight_layout(rect=(0, 0.14, 1, 1))
(ROOT / "docs" / "images").mkdir(parents=True, exist_ok=True)
fig.savefig(ROOT / "docs" / "images" / "clinical.png", dpi=200)
print("saved")
