"""Multi-objective study: time, survival, door waiting and fairness together. ALL clinical inputs and the
survival curve are assumed.

Weights (w, s, lam) of a scalarised policy are swept on VALIDATION seeds (500-529); the non-dominated
set is found there and its representative points are then scored once on TEST seeds (1000-1029).

    python experiments/run_multiobjective.py
Outputs: results/mo_validation.csv, results/mo_test.csv, results/mo_pareto.png (docs/images copy)
"""
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from amburoute.clinical import ClinicalHeuristic, MultiObjectiveHeuristic, NearestCapable, with_capabilities
from amburoute.geo import load_hospitals
from amburoute.policies import CapacityHeuristic
from amburoute.sim import SCENARIOS, SimConfig, run_episode
from amburoute.traffic import load_calibrated

RES = ROOT / "results"
VAL, TEST = list(range(500, 530)), list(range(1000, 1030))
H = with_capabilities(load_hospitals())
T = load_calibrated()
CFG = SimConfig(clinical=True)
# objectives: (column, direction) -> everything below is converted to "lower is better"
OBJ = {"ttdc": ("ttb_mean", 1), "surv": ("surv_mean", -1), "door_wait": ("door_wait_mean", 1), "jain": ("jain_load_balance", -1)}


def score(pol, seeds, sc):
    r = [run_episode(H, SCENARIOS[sc], s, pol, CFG, T) for s in seeds]
    return {k: float(np.mean([x[c] for x in r])) for k, (c, _) in OBJ.items()} | {
        "crit_ttdc": float(np.mean([x["crit_ttdc_mean"] for x in r])),
        "rejections": float(np.mean([x["rejections"] for x in r]))}


def dominated(M):
    """M: (n, d) minimise-all. Returns boolean mask of dominated rows."""
    n = len(M)
    return np.array([any(np.all(M[j] <= M[i]) and np.any(M[j] < M[i]) for j in range(n) if j != i) for i in range(n)])


def minimise(df):
    return np.column_stack([df[k] * OBJ[k][1] for k in OBJ])


rows = []
for sc in ("peak", "surge"):
    for w, s, lam in itertools.product((0, 10, 20, 40, 80), (0, 1, 2), (0, 15, 40, 80)):
        rows.append(dict(scenario=sc, w=w, s=s, lam=lam, **score(MultiObjectiveHeuristic(w, s, lam), VAL, sc)))
    print("validation sweep done", sc, flush=True)
val = pd.DataFrame(rows)
val["pareto"] = False
for sc in ("peak", "surge"):
    m = val.scenario == sc
    val.loc[m, "pareto"] = ~dominated(minimise(val[m]))
val.round(4).to_csv(RES / "mo_validation.csv", index=False)

# Named operating points picked on VALIDATION seeds from the peak front (nothing is read from test seeds).
front = val[(val.scenario == "peak") & val.pareto].copy()
M = minimise(front)
z = pd.Series(((M - M.min(axis=0)) / np.where(np.ptp(M, axis=0) == 0, 1, np.ptp(M, axis=0))).sum(axis=1), index=front.index)
picks = {"fastest": front.loc[front.ttdc.idxmin()], "best survival": front.loc[front.surv.idxmax()],
         "fairest": front.loc[front.jain.idxmax()], "balanced": front.loc[z.idxmin()]}
print(pd.DataFrame(picks).T[["w", "s", "lam", "ttdc", "surv", "door_wait", "jain"]].round(3).to_string())

out = []
for sc in ("peak", "surge"):
    refs = [("nearest capable", NearestCapable()), ("clinical heuristic (paper extension)", ClinicalHeuristic(80.0))]
    refs += [(f"multi-objective: {k}", MultiObjectiveHeuristic(float(p.w), float(p.s), float(p.lam))) for k, p in picks.items()]
    for name, pol in refs:
        out.append(dict(scenario=sc, policy=name, **score(pol, TEST, sc)))
test = pd.DataFrame(out)
test.round(4).to_csv(RES / "mo_test.csv", index=False)
pd.set_option("display.width", 220)
print(test.round(3).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
sub = val[val.scenario == "peak"]
for ax, (xk, yk, xl, yl) in zip(axes, [("ttdc", "surv", "Mean time to definitive care (min)", "Expected survival (assumed curve)"),
                                       ("door_wait", "jain", "Mean wait after reaching a hospital (min)", "Load balance across hospitals (Jain)")]):
    ax.scatter(sub[xk], sub[yk], s=14, color="#bbbbbb", label="all weightings")
    pf = sub[sub.pareto]
    ax.scatter(pf[xk], pf[yk], s=26, color="#2f9e6b", label="non-dominated")
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
(ROOT / "docs" / "images").mkdir(parents=True, exist_ok=True)
fig.savefig(ROOT / "docs" / "images" / "multiobjective.png", dpi=200)
print("saved")
