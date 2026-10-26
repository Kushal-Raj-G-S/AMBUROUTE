"""Multi-ambulance assignment as a congestion game solved by best-response.

Each ambulance i picks a hospital h. Its cost is

    c_i(h) = ETA_ih + beta * L_r / max(F_r, 1),      r = (pool of i, h)

where L_r is the number of ambulances on resource r (already in flight plus
those assigned in this batch, including i) and F_r the published free beds.
Resources of different pools (general / ICU) are separate.

Because the congestion term depends only on the load of the chosen resource
and the rest of the cost is a per-player constant, this is an exact potential
game (Rosenthal): Phi = sum_i ETA_i,h(i) + sum_r sum_{k<=L_r} beta*k/F_r.
Every strictly improving unilateral move lowers Phi, so best-response dynamics
terminate at a pure Nash equilibrium. `nash_assign` returns whether it did.
"""
from __future__ import annotations

import numpy as np

from .sim import View

PENALTY = 1e3  # cost added to resources with no published free bed


def _cost_row(eta_i, load_r, free_r, beta, power=1.0):
    cost = eta_i + beta * ((load_r + 1) / np.maximum(free_r, 1)) ** power
    return cost + np.where(free_r <= 0, PENALTY, 0.0)


def potential(eta, needs_icu, free, inflight, assignment, beta, power=1.0) -> float:
    pool = needs_icu.astype(int)
    phi = float(sum(eta[i, h] for i, h in enumerate(assignment)))
    load = inflight.astype(float).copy()
    for p in (0, 1):
        for h in range(eta.shape[1]):
            n = sum(1 for i, a in enumerate(assignment) if a == h and pool[i] == p)
            base = load[p, h]
            f = max(free[p, h], 1)
            phi += sum(beta * ((base + k) / f) ** power for k in range(1, n + 1))
            phi += PENALTY * n * (free[p, h] <= 0)
    return phi


def nash_assign(
    view: View, beta: float, rng: np.random.Generator, max_iter: int = 100,
    power: float = 1.0,
) -> tuple[list[int], int, bool]:
    n, H = view.eta.shape
    pool = view.needs_icu.astype(int)
    load = view.inflight.astype(float).copy()  # (2, H)
    asg = np.zeros(n, dtype=int)

    # Greedy initialisation (sequential best response in arrival order).
    for i in range(n):
        p = pool[i]
        h = int(np.argmin(_cost_row(view.eta[i], load[p], view.free[p], beta, power)))
        asg[i] = h
        load[p, h] += 1

    # Best-response rounds until no ambulance can improve alone.
    for it in range(1, max_iter + 1):
        moved = False
        for i in rng.permutation(n):
            p = pool[i]
            cur = asg[i]
            load[p, cur] -= 1
            cost = _cost_row(view.eta[i], load[p], view.free[p], beta, power)
            best = int(np.argmin(cost))
            if cost[best] < cost[cur] - 1e-9:
                asg[i] = best
                moved = True
            load[p, asg[i]] += 1
        if not moved:
            return list(asg), it, True
    return list(asg), max_iter, False


class NashPolicy:
    name = "nash"

    def __init__(self, beta: float = 30.0, power: float = 1.0):
        self.beta = beta
        self.power = power
        self.last_iters = 0
        self.last_converged = True
        self.iters_log: list[int] = []
        self.converged_log: list[bool] = []

    def assign(self, view, rng):
        asg, iters, ok = nash_assign(view, self.beta, rng, power=self.power)
        self.last_iters, self.last_converged = iters, ok
        self.iters_log.append(iters)
        self.converged_log.append(ok)
        return asg
