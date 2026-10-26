"""Score every trained PPO checkpoint on VALIDATION seeds (500-529) and pick
the winner. Test seeds are not used here."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from amburoute.geo import load_hospitals
from amburoute.policies import CapacityHeuristic
from amburoute.rl_env import PPOPolicy
from amburoute.sim import SCENARIOS, run_episode
from amburoute.traffic import load_calibrated

VAL_SEEDS = list(range(500, 530))
h = load_hospitals()
T = load_calibrated()
best = json.loads((ROOT / "results" / "best_baseline_params.json").read_text())
rows = []


def score(policy, label):
    per = {}
    for sc in SCENARIOS.values():
        r = [run_episode(h, sc, s, policy, traffic=T) for s in VAL_SEEDS]
        per[sc.name] = np.mean([x["ttb_mean"] for x in r])
    rows.append(dict(model=label, **per, mean_all=np.mean(list(per.values()))))
    print(rows[-1], flush=True)


score(CapacityHeuristic(best["capacity_heuristic"]["w"]), "capacity_heuristic(tuned)")
for pt in sorted((ROOT / "results" / "models").glob("*.pt")):
    score(PPOPolicy(pt, name=pt.stem), pt.stem)
df = pd.DataFrame(rows).round(3).sort_values("mean_all")
df.to_csv(ROOT / "results" / "validation_ppo.csv", index=False)
print(df.to_string(index=False))
ppo = df[~df.model.str.startswith("capacity")]
if len(ppo):
    (ROOT / "results" / "selected_ppo.txt").write_text(ppo.iloc[0].model)
