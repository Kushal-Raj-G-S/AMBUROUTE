"""A graph network that learns the congestion cost instead of using a hand-set term.

Hospitals are nodes; edges join each hospital to its k nearest neighbours (where turned-away ambulances end up,
so load spreads along them). Node features describe what a dispatcher can see for one request: the ETA to the
hospital, published free beds, ambulances already on the way, size and bed pool. Two rounds of message passing
let a node's score depend on how crowded its neighbours are. The network predicts the extra minutes the patient
will lose at the door (diversions, handovers, boarding) if sent there; the policy sends each ambulance to the
hospital with the smallest ETA plus predicted extra delay, updating the in-flight counts as it goes.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .geo import pairwise_km

N_FEAT = 7


def knn_adjacency(hospitals, k: int = 4) -> np.ndarray:
    d = pairwise_km(hospitals).copy()
    np.fill_diagonal(d, np.inf)
    A = np.zeros_like(d)
    for i in range(len(d)):
        A[i, np.argsort(d[i])[:k]] = 1.0 / k
    return A


def node_features(view, i: int, inflight: np.ndarray) -> np.ndarray:
    """(H, N_FEAT) features of every hospital for request i."""
    p = int(view.needs_icu[i])
    free = view.free[p].astype(float)
    tot = np.maximum(view.total[p].astype(float), 1.0)
    infl = inflight[p].astype(float)
    eff = free - infl
    return np.stack([
        view.eta[i] / 30.0,
        np.log1p(np.maximum(free, 0.0)) / 4.0,
        np.clip(infl / 10.0, 0.0, 3.0),
        np.clip(eff / 10.0, -3.0, 3.0),
        np.clip(free / tot, 0.0, 1.0),
        np.log(tot) / 6.0,
        np.full(len(free), float(p)),
    ], axis=1).astype(np.float32)


class HospitalGNN(nn.Module):
    def __init__(self, adjacency: np.ndarray, hidden: int = 48, use_graph: bool = True):
        super().__init__()
        self.register_buffer("A", torch.tensor(adjacency if use_graph else np.zeros_like(adjacency), dtype=torch.float32))
        self.inp = nn.Linear(N_FEAT, hidden)
        self.self1, self.nbr1 = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden, bias=False)
        self.self2, self.nbr2 = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden, bias=False)
        self.glob = nn.Linear(hidden, hidden, bias=False)  # city-wide load context
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):  # x: (B, H, N_FEAT) -> predicted extra minutes (B, H)
        z = torch.relu(self.inp(x))
        z = torch.relu(self.self1(z) + self.nbr1(self.A @ z))
        z = torch.relu(self.self2(z) + self.nbr2(self.A @ z) + self.glob(z.mean(dim=1, keepdim=True)))
        return nn.functional.softplus(self.out(z).squeeze(-1))


class GNNPolicy:
    def __init__(self, model: HospitalGNN, scale: float = 1.0, name: str = "gnn"):
        self.model, self.scale, self.name = model.eval(), scale, name

    @torch.no_grad()
    def assign(self, view, rng):
        infl = view.inflight.astype(float).copy()
        out = []
        for i in range(len(view.eta)):
            x = torch.tensor(node_features(view, i, infl))[None]
            pred = self.model(x)[0].numpy()
            h = int(np.argmin(view.eta[i] + self.scale * pred))
            out.append(h)
            infl[int(view.needs_icu[i]), h] += 1
        return out
