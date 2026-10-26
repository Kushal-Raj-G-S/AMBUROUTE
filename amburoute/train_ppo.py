"""Train a PPO hospital-selection policy.

    python -m amburoute.train_ppo --tag v2 --seed 0 --steps 1000000 --k 12 --herding 0.3 --bc 150

Outputs (never overwritten across tags/seeds):
    results/models/<tag>_s<seed>.pt / .json    policy weights + config
    results/models/<tag>_s<seed>_curve.csv     training reward curve
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from .geo import load_hospitals
from .rl_env import DispatchEnv, save_policy

ROOT = Path(__file__).resolve().parent.parent
TRAIN_SEED0 = 100_000  # training episodes use seeds far from validation/test


class RewardLog(BaseCallback):
    def __init__(self, path):
        super().__init__()
        self.path = Path(path)
        self.rows = []

    def _on_rollout_end(self):
        buf = self.model.ep_info_buffer
        if buf:
            self.rows.append((self.num_timesteps, sum(e["r"] for e in buf) / len(buf)))

    def _on_step(self):
        return True

    def _on_training_end(self):
        with open(self.path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timesteps", "mean_episode_reward"])
            w.writerows(self.rows)


def behaviour_clone(model, env: DispatchEnv, episodes: int, epochs: int = 30):
    """Warm start: imitate the capacity heuristic with cross-entropy."""
    xs, ys = [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            a = env.teacher_action()
            xs.append(obs)
            ys.append(a)
            obs, _, done, _, _ = env.step(a)
    X = torch.tensor(np.array(xs))
    y = torch.tensor(ys)
    opt = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
    n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            loss = -model.policy.get_distribution(X[idx]).log_prob(y[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        acc = (model.policy.get_distribution(X).distribution.probs.argmax(1) == y).float().mean()
    print(f"behaviour cloning: {n} samples, imitation accuracy {acc:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--herding", type=float, default=0.0)
    ap.add_argument("--bc", type=int, default=0, help="episodes of heuristic imitation")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--calibrated", action="store_true", help="train on the Mapbox-calibrated traffic model")
    args = ap.parse_args()

    torch.set_num_threads(1)
    out = ROOT / "results" / "models"
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{args.tag}_s{args.seed}"
    hosp = load_hospitals()
    traffic = None
    if args.calibrated:
        from .traffic import load_calibrated

        traffic = load_calibrated()

    env = Monitor(DispatchEnv(hosp, seed=TRAIN_SEED0 + args.seed, k=args.k,
                              herding_coef=args.herding, traffic=traffic))
    model = PPO(
        "MlpPolicy", env, seed=args.seed, verbose=0, device="cpu",
        learning_rate=lambda frac: args.lr * max(frac, 0.1),
        n_steps=2048, batch_size=256, gamma=0.99, gae_lambda=0.95,
        clip_range=args.clip, ent_coef=args.ent,
        policy_kwargs=dict(net_arch=[128, 128]),
    )
    if args.bc:
        behaviour_clone(model, DispatchEnv(hosp, seed=TRAIN_SEED0 + 5000 + args.seed,
                                           k=args.k, herding_coef=args.herding, traffic=traffic), args.bc)
    model.learn(args.steps, callback=RewardLog(str(stem) + "_curve.csv"))
    save_policy(model, stem, args.k, args.herding)
    print("saved", stem, flush=True)


if __name__ == "__main__":
    main()
