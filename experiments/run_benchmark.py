"""FINAL test benchmark on held-out seeds (1000-1049). Run once after tuning.

Policies are configured from results/best_baseline_params.json and
results/selected_ppo.txt, both chosen on VALIDATION seeds only.

    python experiments/run_benchmark.py
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
from amburoute.policies import CapacityHeuristic, Nearest, NearestWithBeds, PrototypeHeuristic
from amburoute.sim import SCENARIOS, run_episode
from amburoute.traffic import load_calibrated

RES = ROOT / "results"
SEEDS = list(range(1000, 1050))
LABEL = {"nearest": "Nearest", "nearest_beds": "Nearest+beds", "prototype_heuristic": "Prototype rule",
         "capacity_heuristic": "Capacity heuristic", "nash": "Nash", "ppo": "PPO"}
METRICS = ["ttb_mean", "ttb_median", "ttb_p95", "diversion_rate", "icu_diversion_rate", "rejections",
           "max_rejections_one_hospital", "jain_load_balance", "decision_ms"]


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


def boot_ci(x, n=5000, seed=0):
    r = np.random.default_rng(seed)
    means = r.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def main():
    h = load_hospitals()
    T = load_calibrated()
    best = json.loads((RES / "best_baseline_params.json").read_text())
    policies = [Nearest(), NearestWithBeds(), PrototypeHeuristic(),
                CapacityHeuristic(best["capacity_heuristic"]["w"]),
                NashPolicy(best["nash"]["beta"], best["nash"]["power"])]
    sel = RES / "selected_ppo.txt"
    if sel.exists():
        from amburoute.rl_env import PPOPolicy
        policies.append(PPOPolicy(RES / "models" / f"{sel.read_text().strip()}.pt"))
    else:
        print("WARNING: no selected PPO model; PPO omitted")

    rows = []
    for sc in SCENARIOS.values():
        for p in policies:
            for s in SEEDS:
                rows.append(dict(scenario=sc.name, policy=p.name, seed=s,
                                 **run_episode(h, sc, s, p, traffic=T)))
    raw = pd.DataFrame(rows)
    raw.to_csv(RES / "final_raw.csv", index=False)
    g = raw.groupby(["scenario", "policy"])[METRICS]
    summ = g.mean().join(g.std(), rsuffix="_sd")
    ci = raw.groupby(["scenario", "policy"]).ttb_mean.apply(lambda x: boot_ci(x.to_numpy()))
    summ["ttb_ci_lo"] = [ci[k][0] for k in summ.index]
    summ["ttb_ci_hi"] = [ci[k][1] for k in summ.index]
    summ.round(3).to_csv(RES / "final_summary.csv")

    # ---- paired tests: difference in mean time-to-bed (a - b), Wilcoxon + Holm
    pairs = [("capacity_heuristic", "nearest"), ("nash", "nearest"), ("nash", "capacity_heuristic"),
             ("capacity_heuristic", "prototype_heuristic"), ("nash", "prototype_heuristic"),
             ("ppo", "capacity_heuristic"), ("ppo", "nash"), ("ppo", "nearest")]
    tests = []
    for sc in SCENARIOS:
        for metric in ("ttb_mean", "rejections"):
            block = []
            for a, b in pairs:
                da = raw[(raw.scenario == sc) & (raw.policy == a)].sort_values("seed")
                db = raw[(raw.scenario == sc) & (raw.policy == b)].sort_values("seed")
                if da.empty or db.empty:
                    continue
                diff = da[metric].to_numpy() - db[metric].to_numpy()
                p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
                lo, hi = boot_ci(diff)
                block.append(dict(scenario=sc, metric=metric, a=a, b=b, mean_a=da[metric].mean(),
                                  mean_b=db[metric].mean(), mean_diff=diff.mean(), diff_ci_lo=lo,
                                  diff_ci_hi=hi, rel_change_pct=100 * diff.mean() / db[metric].mean(),
                                  p_raw=p))
            adj = holm([t["p_raw"] for t in block])
            for t, a_ in zip(block, adj):
                t["p_holm"] = a_
            tests += block
    pd.DataFrame(tests).round(4).to_csv(RES / "final_tests.csv", index=False)

    # ---- Nash beta sweep (test seeds) + convergence statistics
    sweep, conv = [], []
    for beta in (0, 5, 10, 20, 40):
        for sc in (SCENARIOS["peak"], SCENARIOS["surge"]):
            pol = NashPolicy(beta, best["nash"]["power"])
            for s in SEEDS:
                sweep.append(dict(beta=beta, scenario=sc.name, seed=s, **run_episode(h, sc, s, pol, traffic=T)))
            if beta == best["nash"]["beta"]:
                it = np.array(pol.iters_log)
                conv.append(dict(scenario=sc.name, batches=len(it),
                                 converged_pct=100 * np.mean(pol.converged_log),
                                 mean_rounds=it.mean(), max_rounds=int(it.max())))
    sw = pd.DataFrame(sweep)
    sw.groupby(["scenario", "beta"])[["ttb_mean", "ttb_p95", "rejections", "jain_load_balance"]].mean().round(3).to_csv(RES / "final_beta_sweep.csv")
    pd.DataFrame(conv).round(3).to_csv(RES / "final_nash_convergence.csv", index=False)

    # ---- figures
    order = [k for k in LABEL if k in raw.policy.unique()]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, sc in zip(axes, SCENARIOS):
        m = [summ.loc[(sc, k), "ttb_mean"] for k in order]
        lo = [summ.loc[(sc, k), "ttb_mean"] - summ.loc[(sc, k), "ttb_ci_lo"] for k in order]
        hi = [summ.loc[(sc, k), "ttb_ci_hi"] - summ.loc[(sc, k), "ttb_mean"] for k in order]
        ax.bar(range(len(order)), m, yerr=[lo, hi], color="#4C72B0", capsize=3)
        ax.set_xticks(range(len(order)), [LABEL[k] for k in order], rotation=40, ha="right", fontsize=8)
        ax.set_title(sc); ax.set_ylabel("Mean time to bed (min)")
    fig.tight_layout(); fig.savefig(RES / "fig_final_time_to_bed.png", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, sc in zip(axes, SCENARIOS):
        ax.bar(range(len(order)), [summ.loc[(sc, k), "rejections"] for k in order], color="#C44E52")
        ax.set_xticks(range(len(order)), [LABEL[k] for k in order], rotation=40, ha="right", fontsize=8)
        ax.set_title(sc); ax.set_ylabel("Rejected arrivals / episode")
    fig.tight_layout(); fig.savefig(RES / "fig_final_rejections.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for sc in ("peak", "surge"):
        d = sw[sw.scenario == sc].groupby("beta").ttb_mean.mean()
        ax.plot(d.index, d.values, marker="o", label=sc)
    ax.set_xlabel(r"congestion weight $\beta$"); ax.set_ylabel("Mean time to bed (min)"); ax.legend()
    fig.tight_layout(); fig.savefig(RES / "fig_final_beta_sweep.png", dpi=200); plt.close(fig)

    if sel.exists():
        cur = RES / "models" / f"{sel.read_text().strip()}_curve.csv"
        if cur.exists():
            c = pd.read_csv(cur)
            fig, ax = plt.subplots(figsize=(4.6, 3.4))
            ax.plot(c.timesteps, c.mean_episode_reward)
            ax.set_xlabel("Training steps"); ax.set_ylabel("Mean episode reward (shaped)")
            fig.tight_layout(); fig.savefig(RES / "fig_final_ppo_training.png", dpi=200); plt.close(fig)

    pd.set_option("display.width", 220)
    print(summ[["ttb_mean", "ttb_ci_lo", "ttb_ci_hi", "ttb_p95", "rejections", "jain_load_balance"]].round(2).to_string())


if __name__ == "__main__":
    main()
