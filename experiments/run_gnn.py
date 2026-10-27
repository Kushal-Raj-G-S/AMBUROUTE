"""Learned congestion cost with a graph network (mam's point 3).

1. Collect exploratory dispatches on TRAIN seeds (>= 100000): tuned heuristic with a random weight and some random
   choices among the five nearest hospitals, so the data covers many load states. Label = minutes lost at the door
   (first arrival to bed) of the hospital actually chosen.
2. Train the graph network (and a copy with the edges removed, as an ablation) on those labels.
3. Choose the cost scale on VALIDATION seeds (500-529); score once on TEST seeds (1000-1049).

    python experiments/run_gnn.py
Outputs: results/gnn_summary.csv, results/gnn_tests.csv, results/gnn_validation.csv, results/models/gnn_*.pt
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from amburoute.geo import load_hospitals
from amburoute.gnn import GNNPolicy, HospitalGNN, knn_adjacency, node_features
from amburoute.nash import NashPolicy
from amburoute.policies import CapacityHeuristic, Nearest
from amburoute.sim import SCENARIOS, Sim, SimConfig, run_episode
from amburoute.traffic import load_calibrated

torch.set_num_threads(2)
RES = ROOT / "results"
H, T = load_hospitals(), load_calibrated()
best = json.loads((RES / "best_baseline_params.json").read_text())
w, beta, gam = best["capacity_heuristic"]["w"], best["nash"]["beta"], best["nash"]["power"]
A = knn_adjacency(H, 4)
cfg = SimConfig()


class Explorer:
    """Behaviour policy for data collection: heuristic with a random weight, plus random nearby choices."""

    def __init__(self, rng):
        self.base = CapacityHeuristic(float(rng.choice([0, 5, 20, 60])))
        self.eps = float(rng.choice([0.1, 0.3]))
        self.name = "explorer"

    def assign(self, view, rng):
        out = self.base.assign(view, rng)
        for i in range(len(out)):
            if rng.random() < self.eps:
                out[i] = int(rng.choice(np.argsort(view.eta[i])[:5]))
        return out


def collect(seed, sc):
    sim = Sim(H, SCENARIOS[sc], seed, cfg, T)
    prng = np.random.default_rng(seed + 10_000)
    pol = Explorer(np.random.default_rng(seed + 20_000))
    rec = []  # (features, chosen hospital, request id)
    while (batch := sim.next_batch()) is not None:
        view = sim.make_view(batch)
        asg = pol.assign(view, prng)
        infl = view.inflight.astype(float).copy()
        for i, (r, h) in enumerate(zip(batch, asg)):
            rec.append((node_features(view, i, infl), int(h), r.rid))
            infl[r.pool, int(h)] += 1
            sim.dispatch(r, int(h))
    sim.finish()
    X = np.stack([r[0] for r in rec])
    hsel = np.array([r[1] for r in rec])
    y = np.array([min(sim.reqs[r[2]].t_done - sim.reqs[r[2]].t_first_arrival, 60.0) for r in rec], dtype=np.float32)
    return X, hsel, y


def train(use_graph, X, hsel, y, epochs=40, seed=0):
    torch.manual_seed(seed)
    m = HospitalGNN(A, use_graph=use_graph)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    X, hsel, y = torch.tensor(X), torch.tensor(hsel), torch.tensor(y)
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 512):
            idx = perm[i:i + 512]
            pred = m(X[idx])[torch.arange(len(idx)), hsel[idx]]
            loss = torch.nn.functional.huber_loss(pred, y[idx], delta=10.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        if ep % 10 == 9:
            print(f"  graph={use_graph} epoch {ep + 1} loss {tot / n:.3f}", flush=True)
    return m


parts = [collect(100_000 + s, sc) for s in range(220) for sc in ("offpeak", "peak", "surge")]
X = np.concatenate([p[0] for p in parts])
hsel = np.concatenate([p[1] for p in parts])
y = np.concatenate([p[2] for p in parts])
print(f"training samples {len(X)}, mean extra delay {y.mean():.2f} min, share > 1 min {(y > 1).mean():.3f}", flush=True)

models = {"gnn": train(True, X, hsel, y), "gnn_no_edges": train(False, X, hsel, y)}
for k, m in models.items():
    torch.save(m.state_dict(), RES / "models" / f"{k}.pt")

# scale chosen on validation seeds, under peak and surge together
VAL, TEST = list(range(500, 530)), list(range(1000, 1050))
val = []
for k, m in models.items():
    for s in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
        pol = GNNPolicy(m, s, k)
        t = np.mean([run_episode(H, SCENARIOS[sc], sd, pol, cfg, T)["ttb_mean"] for sc in ("peak", "surge") for sd in VAL])
        val.append(dict(model=k, scale=s, ttb=t))
val = pd.DataFrame(val)
val.round(4).to_csv(RES / "gnn_validation.csv", index=False)
print(val.round(3).to_string(index=False), flush=True)
pick = {k: float(val[val.model == k].sort_values("ttb").iloc[0].scale) for k in models}

pols = {"nearest": Nearest(), "heuristic (tuned)": CapacityHeuristic(w), "nash (tuned)": NashPolicy(beta, gam),
        "gnn (learned cost)": GNNPolicy(models["gnn"], pick["gnn"], "gnn"),
        "gnn without edges": GNNPolicy(models["gnn_no_edges"], pick["gnn_no_edges"], "gnn_no_edges")}
rows = []
for sc in ("offpeak", "peak", "surge"):
    for name, p in pols.items():
        for sd in TEST:
            r = run_episode(H, SCENARIOS[sc], sd, p, cfg, T)
            rows.append(dict(scenario=sc, policy=name, seed=sd, ttb_mean=r["ttb_mean"], rejections=r["rejections"],
                             jain=r["jain_load_balance"], decision_ms=r["decision_ms"]))
    print("done", sc, flush=True)
raw = pd.DataFrame(rows)
summ = raw.groupby(["scenario", "policy"])[["ttb_mean", "rejections", "jain", "decision_ms"]].mean().round(3)
summ.to_csv(RES / "gnn_summary.csv")
pd.set_option("display.width", 200)
print(summ.to_string())
tests = []
for sc in ("offpeak", "peak", "surge"):
    for a, b in (("gnn (learned cost)", "heuristic (tuned)"), ("gnn (learned cost)", "nash (tuned)"),
                 ("gnn (learned cost)", "gnn without edges")):
        da = raw[(raw.scenario == sc) & (raw.policy == a)].sort_values("seed").ttb_mean.to_numpy()
        db = raw[(raw.scenario == sc) & (raw.policy == b)].sort_values("seed").ttb_mean.to_numpy()
        d = da - db
        tests.append(dict(scenario=sc, a=a, b=b, mean_a=da.mean(), mean_b=db.mean(), delta=d.mean(),
                          rel_pct=100 * d.mean() / db.mean(), p_raw=wilcoxon(d).pvalue))
t = pd.DataFrame(tests)
t["p_bonf"] = np.minimum(1, t.p_raw * len(t))
t.round(4).to_csv(RES / "gnn_tests.csv", index=False)
print(t.round(4).to_string(index=False))
