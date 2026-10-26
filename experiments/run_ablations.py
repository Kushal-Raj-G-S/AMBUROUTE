"""Ablations on information quality, using tuned policies on TEST seeds.

1. Bed-data freshness: report_interval in {1, 5, 15, 30} minutes.
2. Traffic outage: planner falls back to free-flow ETAs (routing API down).
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from amburoute.geo import load_hospitals
from amburoute.nash import NashPolicy
from amburoute.policies import CapacityHeuristic, Nearest, NearestWithBeds
from amburoute.sim import SCENARIOS, SimConfig, run_episode
from amburoute.traffic import load_calibrated

best = json.loads((ROOT / "results" / "best_baseline_params.json").read_text())
SEEDS = list(range(1000, 1050))
h = load_hospitals()
T = load_calibrated()
pols = lambda: [Nearest(), NearestWithBeds(),
                CapacityHeuristic(best["capacity_heuristic"]["w"]),
                NashPolicy(best["nash"]["beta"], best["nash"]["power"])]
rows = []
variants = [("freshness", f"report_every_{r}min", SimConfig(report_interval=r), T) for r in (1, 5, 15, 30)]
variants += [("traffic", "traffic_aware", SimConfig(), T),
             ("traffic", "traffic_outage", SimConfig(traffic_aware_planner=False), T)]
from amburoute.traffic import load_calibrated as _lc
variants += [("ambulance_speedup", f"boost_{b}", SimConfig(), _lc(ambulance_boost=b)) for b in (1.0, 1.25, 1.5)]
variants += [("eta_noise", f"sigma_x{m}", SimConfig(), _lc(sigma=T.sigma * m)) for m in (0.5, 1.0, 2.0)]
for study, label, cfg, TT in variants:
    for sc in (SCENARIOS["peak"], SCENARIOS["surge"]):
        for p in pols():
            vals = [run_episode(h, sc, s, p, cfg, TT) for s in SEEDS]
            rows.append(dict(study=study, variant=label, scenario=sc.name, policy=p.name,
                             ttb_mean=np.mean([v["ttb_mean"] for v in vals]),
                             ttb_p95=np.mean([v["ttb_p95"] for v in vals]),
                             rejections=np.mean([v["rejections"] for v in vals])))
df = pd.DataFrame(rows).round(3)
df.to_csv(ROOT / "results" / "ablations.csv", index=False)
pd.set_option("display.width", 200)
print(df.pivot_table(index=["study", "variant", "scenario"], columns="policy",
                     values="ttb_mean").round(2).to_string())
