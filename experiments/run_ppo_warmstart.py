"""PPO warm-start ablation: the same 1M-step PPO trained with (tag B, 150 behaviour-cloning episodes) and without
(tag C) imitation of the heuristic. Each tag's seed is chosen on VALIDATION seeds (500-529); the chosen
checkpoints are then scored once on TEST seeds (1000-1049). Mean over all three seeds of each tag is reported as
well, to show training variance.

    python experiments/run_ppo_warmstart.py
Outputs: results/ppo_warmstart_validation.csv, results/ppo_warmstart_test.csv, results/ppo_warmstart_tests.csv
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from amburoute.geo import load_hospitals
from amburoute.policies import CapacityHeuristic
from amburoute.rl_env import PPOPolicy
from amburoute.sim import SCENARIOS, run_episode
from amburoute.traffic import load_calibrated

RES = ROOT / "results"
H, T = load_hospitals(), load_calibrated()
best = json.loads((RES / "best_baseline_params.json").read_text())
VAL, TEST = list(range(500, 530)), list(range(1000, 1050))
models = {f"{tag}_s{s}": RES / "models" / f"{tag}_s{s}.pt" for tag in ("B", "C") for s in (0, 1, 2)}


def mean_ttb(pol, seeds, sc):
    return float(np.mean([run_episode(H, SCENARIOS[sc], s, pol, traffic=T)["ttb_mean"] for s in seeds]))


val = []
for name, path in models.items():
    pol = PPOPolicy(path, name=name)
    per = {sc: mean_ttb(pol, VAL, sc) for sc in SCENARIOS}
    val.append(dict(model=name, tag=name[0], **per, mean_all=np.mean(list(per.values()))))
    print(val[-1], flush=True)
val = pd.DataFrame(val).round(3)
val.to_csv(RES / "ppo_warmstart_validation.csv", index=False)
chosen = {tag: val[val.tag == tag].sort_values("mean_all").iloc[0].model for tag in ("B", "C")}
print("chosen on validation:", chosen, flush=True)

rows = []
heur = CapacityHeuristic(best["capacity_heuristic"]["w"])
for sc in ("peak", "surge"):
    for name, pol, role in [("heuristic (tuned)", heur, "reference")] + [
            (m, PPOPolicy(models[m], name=m), "chosen" if m in chosen.values() else "other") for m in models]:
        for s in TEST:
            r = run_episode(H, SCENARIOS[sc], s, pol, traffic=T)
            rows.append(dict(scenario=sc, model=name, role=role, seed=s, ttb_mean=r["ttb_mean"], rejections=r["rejections"]))
    print("done", sc, flush=True)
raw = pd.DataFrame(rows)
summ = raw.groupby(["scenario", "model", "role"])[["ttb_mean", "rejections"]].mean().round(3).reset_index()
summ.to_csv(RES / "ppo_warmstart_test.csv", index=False)
pd.set_option("display.width", 200)
print(summ.to_string(index=False))

tests = []
for sc in ("peak", "surge"):
    g = lambda m: raw[(raw.scenario == sc) & (raw.model == m)].sort_values("seed").ttb_mean.to_numpy()
    for a, b in ((chosen["C"], chosen["B"]), (chosen["C"], "heuristic (tuned)"), (chosen["B"], "heuristic (tuned)")):
        d = g(a) - g(b)
        tests.append(dict(scenario=sc, a=a, b=b, mean_a=g(a).mean(), mean_b=g(b).mean(), rel_pct=100 * d.mean() / g(b).mean(),
                          p_raw=wilcoxon(d).pvalue))
t = pd.DataFrame(tests)
t["p_bonf"] = np.minimum(1, t.p_raw * len(t))
t.round(4).to_csv(RES / "ppo_warmstart_tests.csv", index=False)
print(t.round(4).to_string(index=False))
