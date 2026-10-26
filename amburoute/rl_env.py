"""Gymnasium environment: one decision = pick a hospital for one ambulance.

Observation: features of the k nearest hospitals (by planner ETA) + globals.
Action: index into those k candidates.
Reward (training only): -(ETA/10) - overflow penalty (+ optional herding term),
where overflow means the hospital's true free beds are already covered by
ambulances on the road. Evaluation never uses this shaped reward: it uses
realised time-to-bed from the simulator.
"""
from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .sim import SCENARIOS, Sim, SimConfig, View

N_FEAT = 7
N_GLOBAL = 5
OVERFLOW_PENALTY = 1.5  # in reward units (= 15 minutes)
TEACHER_WEIGHT = 30.0  # capacity-heuristic pressure weight used for warm start


def obs_dim(k: int) -> int:
    return k * N_FEAT + N_GLOBAL


def build_obs(eta_row, needs_icu, free, inflight, total, minute_of_day,
              n_remaining=0, k=8):
    """Return (obs, candidate_indices) for one request."""
    p = int(needs_icu)
    cand = np.argsort(eta_row)[:k]
    f = free[p, cand].astype(float)
    inf = inflight[p, cand].astype(float)
    tot = np.maximum(total[p, cand].astype(float), 1.0)
    feats = np.stack(
        [
            eta_row[cand] / 30.0,
            np.log1p(np.maximum(f, 0)) / 5.0,
            f / tot,
            inf / (np.maximum(f, 0) + 1.0),
            np.clip((f - inf) / 20.0, -2, 2),
            (f - inf <= 0).astype(float),
            np.arange(len(cand)) / max(k - 1, 1),
        ],
        axis=1,
    )
    city_free = max(float(free[p].sum() - inflight[p].sum()), 0.0)
    ang = 2 * np.pi * (minute_of_day % 1440) / 1440
    glob = [float(p), np.sin(ang), np.cos(ang), min(n_remaining, 20) / 10.0,
            np.log1p(city_free) / 8.0]
    return np.concatenate([feats.ravel(), glob]).astype(np.float32), cand


class DispatchEnv(gym.Env):
    def __init__(self, hospitals, scenarios=None, cfg: SimConfig = SimConfig(),
                 seed=0, k: int = 8, herding_coef: float = 0.0, traffic=None):
        super().__init__()
        self.hospitals = hospitals
        self.scenarios = scenarios or list(SCENARIOS.values())
        self.cfg = cfg
        self.k = k
        self.herding_coef = herding_coef
        self.traffic = traffic
        self.observation_space = spaces.Box(-5, 5, shape=(obs_dim(k),), dtype=np.float32)
        self.action_space = spaces.Discrete(k)
        self._rng = np.random.default_rng(seed)
        self.sim: Sim | None = None
        self.queue: list = []
        self.cand = np.zeros(k, dtype=int)

    def _next_request(self):
        while not self.queue:
            batch = self.sim.next_batch()
            if batch is None:
                return None
            self.queue = list(batch)
        return self.queue.pop(0)

    def _observe(self):
        r = self._cur
        v = self.sim.make_view([r])
        obs, self.cand = build_obs(
            v.eta[0], r.needs_icu, v.free, v.inflight, v.total, v.minute_of_day,
            len(self.queue), self.k,
        )
        self._view = v
        return obs

    def teacher_action(self) -> int:
        """Capacity-heuristic choice, restricted to the k candidates."""
        v, p = self._view, self._cur.pool
        eff = (v.free[p] - v.inflight[p])[self.cand].astype(float)
        score = v.eta[0][self.cand] + TEACHER_WEIGHT / (1.0 + np.maximum(eff, 0))
        score = score + np.where(eff <= 0, 1e6, 0)
        return int(np.argmin(score)) if (eff > 0).any() else 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        sc = self.scenarios[int(self._rng.integers(len(self.scenarios)))]
        self.sim = Sim(self.hospitals, sc, int(self._rng.integers(2**31)), self.cfg, self.traffic)
        self.queue = []
        self._cur = self._next_request()
        if self._cur is None:  # empty episode (rare): try again
            return self.reset()
        return self._observe(), {}

    def step(self, action):
        r = self._cur
        h = int(self.cand[int(action)])
        p = r.pool
        eta = float(self._view.eta[0, h])
        true_free = self.sim.total[p, h] - self.sim.occ[p, h]
        load = self.sim.inflight[p, h] + 1
        overflow = float(load > true_free)
        herd = min(load / (true_free + 1.0), 3.0)
        reward = -eta / 10.0 - OVERFLOW_PENALTY * overflow - self.herding_coef * herd
        self.sim.dispatch(r, h)
        self._cur = self._next_request()
        if self._cur is None:
            self.sim.finish()
            return np.zeros(self.observation_space.shape, dtype=np.float32), reward, True, False, {}
        return self._observe(), reward, False, False, {}


def save_policy(model, path: Path, k: int, herding_coef: float):
    """Save weights with plain torch.save (SB3's zip loader breaks on torch 2.14)."""
    import torch

    path = Path(path)
    torch.save(model.policy.state_dict(), path.with_suffix(".pt"))
    path.with_suffix(".json").write_text(json.dumps(dict(k=k, herding_coef=herding_coef)))


class PPOPolicy:
    """Batch-policy wrapper around a trained Stable-Baselines3 PPO model.
    Requests in a batch are dispatched sequentially so the model sees the
    in-flight counts updated by its own earlier choices."""

    name = "ppo"

    def __init__(self, model_path, name: str | None = None):
        import torch
        from stable_baselines3 import PPO

        from .geo import load_hospitals

        path = Path(model_path)
        meta = json.loads(path.with_suffix(".json").read_text())
        self.k = meta["k"]
        if name:
            self.name = name
        self.model = PPO("MlpPolicy", DispatchEnv(load_hospitals(), k=self.k),
                         device="cpu", policy_kwargs=dict(net_arch=[128, 128]))
        state = torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=True)
        self.model.policy.load_state_dict(state)
        self.model.policy.set_training_mode(False)

    def assign(self, view: View, rng):
        inflight = view.inflight.copy()
        n = len(view.eta)
        out = []
        for i in range(n):
            obs, cand = build_obs(
                view.eta[i], view.needs_icu[i], view.free, inflight, view.total,
                view.minute_of_day, n - i - 1, self.k,
            )
            a, _ = self.model.predict(obs, deterministic=True)
            h = int(cand[int(a)])
            out.append(h)
            inflight[int(view.needs_icu[i]), h] += 1
        return out
