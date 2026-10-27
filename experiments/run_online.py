"""Online adaptation study: local traffic jams that the fitted travel-time model does not know about.

Scenarios of drift (from minute 30): none; four random hospitals 2x slower to reach; all trips 1.5x slower.
The adaptation rate alpha is chosen on VALIDATION seeds (500-529); everything is then scored once on TEST seeds
(1000-1049). Base policy parameters are the paper's tuned ones (not re-tuned for the drift).

    python experiments/run_online.py
Outputs: results/online_validation.csv, results/online_raw.csv, results/online_summary.csv, docs/images/online.png
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

from amburoute.geo import load_hospitals
from amburoute.nash import NashPolicy
from amburoute.online import OnlineETA, OracleETA
from amburoute.policies import CapacityHeuristic
from amburoute.sim import SCENARIOS, SimConfig, run_episode
from amburoute.traffic import load_calibrated

RES = ROOT / "results"
best = json.loads((RES / "best_baseline_params.json").read_text())
w, beta, gam = best["capacity_heuristic"]["w"], best["nash"]["beta"], best["nash"]["power"]
H, T = load_hospitals(), load_calibrated()
DRIFTS = {"no drift": SimConfig(), "local jams (4 hospitals x2)": SimConfig(drift_n=4, drift_factor=2.0),
          "citywide slowdown x1.5": SimConfig(drift_n=-1, drift_factor=1.5)}
VAL, TEST = list(range(500, 530)), list(range(1000, 1050))


def mean_ttb(pol, cfg, sc, seeds):
    return float(np.mean([run_episode(H, SCENARIOS[sc], s, pol, cfg, T)["ttb_mean"] for s in seeds]))


# 1) choose alpha on validation seeds (under local jams, peak) using the heuristic as base
val = [dict(alpha=a, ttb=mean_ttb(OnlineETA(CapacityHeuristic(w), a), DRIFTS["local jams (4 hospitals x2)"], "peak", VAL))
       for a in (0.05, 0.1, 0.2, 0.3, 0.5, 0.8)]
val = pd.DataFrame(val)
val.round(4).to_csv(RES / "online_validation.csv", index=False)
alpha = float(val.loc[val.ttb.idxmin(), "alpha"])
print(val.round(3).to_string(index=False), "-> alpha", alpha, flush=True)

# 2) one-shot test
mk = {"heuristic": lambda: CapacityHeuristic(w), "nash": lambda: NashPolicy(beta, gam)}
rows = []
for dname, cfg in DRIFTS.items():
    for sc in ("peak", "surge"):
        for base, f in mk.items():
            for kind in ("fixed model", "online (learned)", "oracle"):
                pol = {"fixed model": f(), "online (learned)": OnlineETA(f(), alpha), "oracle": OracleETA(f())}[kind]
                for s in TEST:
                    r = run_episode(H, SCENARIOS[sc], s, pol, cfg, T)
                    rows.append(dict(drift=dname, scenario=sc, base=base, variant=kind, seed=s,
                                     ttb_mean=r["ttb_mean"], rejections=r["rejections"]))
        print("done", dname, sc, flush=True)
raw = pd.DataFrame(rows)
raw.to_csv(RES / "online_raw.csv", index=False)

summ = []
for (d, sc, b), g in raw.groupby(["drift", "scenario", "base"]):
    fixed = g[g.variant == "fixed model"].sort_values("seed").ttb_mean.to_numpy()
    for v, gg in g.groupby("variant"):
        x = gg.sort_values("seed")
        diff = x.ttb_mean.to_numpy() - fixed
        summ.append(dict(drift=d, scenario=sc, base=b, variant=v, ttb_mean=x.ttb_mean.mean(),
                         rejections=x.rejections.mean(), delta_min=diff.mean(), delta_pct=100 * diff.mean() / fixed.mean(),
                         p_raw=wilcoxon(diff).pvalue if np.any(diff) and v != "fixed model" else np.nan))
S = pd.DataFrame(summ)
k = S.p_raw.notna().sum()
S["p_bonf"] = np.minimum(1, S.p_raw * k)
S.round(4).to_csv(RES / "online_summary.csv", index=False)
pd.set_option("display.width", 220)
print(S.round(3).to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
cols = {"fixed model": "#999999", "online (learned)": "#2f9e6b", "oracle": "#C4A052"}
for ax, d in zip(axes, DRIFTS):
    sub = S[(S.drift == d) & (S.scenario == "peak") & (S.base == "heuristic")]
    ax.bar(sub.variant, sub.ttb_mean, color=[cols[v] for v in sub.variant])
    ax.set_title(d, fontsize=9)
    ax.tick_params(axis="x", labelrotation=20, labelsize=8)
axes[0].set_ylabel("Mean time to bed, evening peak (min)")
axes[0].set_ylim(20, None)
fig.tight_layout()
fig.savefig(ROOT / "docs" / "images" / "online.png", dpi=200)
print("saved")
