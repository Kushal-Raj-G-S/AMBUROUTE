import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from amburoute.clinical import ClinicalHeuristic, ClinicalNash, NearestCapable, SEV_WEIGHTS, with_capabilities
from amburoute.geo import load_hospitals
from amburoute.nash import _cost_row, nash_assign, potential
from amburoute.policies import CapacityHeuristic, Nearest
from amburoute.sim import SCENARIOS, SimConfig, View, run_episode

H = with_capabilities(load_hospitals())
CFG = SimConfig(clinical=True)


def test_clinical_off_has_no_clinical_metrics():
    r = run_episode(load_hospitals(), SCENARIOS["peak"], 5, CapacityHeuristic())
    assert "crit_ttdc_mean" not in r


def test_capabilities_exist_for_every_specialty():
    assert H[["cap_cardiac", "cap_stroke", "cap_trauma"]].sum().min() >= 2


def test_clinical_run_and_transfers():
    r_blind = run_episode(H, SCENARIOS["peak"], 3, Nearest(), CFG)
    r_cap = run_episode(H, SCENARIOS["peak"], 3, NearestCapable(), CFG)
    for r in (r_blind, r_cap):
        assert r["n_spec"] > 0 and 0.0 <= r["transfer_rate"] <= 1.0
    # Sending specialty patients to a capable hospital cannot cause more transfers than ignoring capability.
    assert r_cap["transfer_rate"] <= r_blind["transfer_rate"]


def test_clinical_policies_only_pick_capable_hospitals():
    rng = np.random.default_rng(0)
    n, Hn = 8, len(H)
    sev = rng.integers(0, 3, n)
    cap = rng.random((n, Hn)) < 0.4
    cap[np.arange(n), rng.integers(0, Hn, n)] = True  # at least one capable hospital each
    v = View(720.0, rng.uniform(3, 30, (n, Hn)), rng.random(n) < 0.3, np.full((2, Hn), 20),
             np.zeros((2, Hn), dtype=int), np.full((2, Hn), 100), capable=cap, severity=sev)
    for pol in (NearestCapable(), ClinicalHeuristic(), ClinicalNash(10.0, 2.0)):
        asg = pol.assign(v, np.random.default_rng(1))
        assert all(cap[i, h] for i, h in enumerate(asg)), pol.name


def test_weighted_restricted_game_is_an_exact_potential_game_with_no_profitable_deviation():
    rng = np.random.default_rng(4)
    for seed in range(15):
        r = np.random.default_rng(seed)
        n, Hn = 10, 9
        sev = r.integers(0, 3, n)
        w = SEV_WEIGHTS[sev]
        cap = r.random((n, Hn)) < 0.5
        cap[np.arange(n), r.integers(0, Hn, n)] = True
        v = View(720.0, r.uniform(3, 30, (n, Hn)), r.random(n) < 0.3, r.integers(2, 15, (2, Hn)),
                 r.integers(0, 3, (2, Hn)), np.full((2, Hn), 100), capable=cap, severity=sev)
        asg, _, ok = nash_assign(v, 10.0, np.random.default_rng(seed), power=2.0, weights=w, allowed=cap)
        assert ok
        pool = v.needs_icu.astype(int)
        load = v.inflight.astype(float).copy()
        for i, h in enumerate(asg):
            load[pool[i], h] += 1
        for i, h in enumerate(asg):  # no ambulance can lower its own cost by switching alone
            p = pool[i]
            load[p, h] -= 1
            cost = _cost_row(v.eta[i], load[p], v.free[p], 10.0, 2.0, w[i], cap[i])
            assert cost[h] <= cost.min() + 1e-6
            load[p, h] += 1
        # exact potential: moving one ambulance changes Phi by exactly its own cost change
        i = int(rng.integers(0, n))
        alt = int(np.flatnonzero(cap[i])[0])
        a2 = list(asg)
        a2[i] = alt

        def own_cost(a):
            ld = v.inflight.astype(float).copy()
            for j, hh in enumerate(a):
                ld[pool[j], hh] += 1
            p = pool[i]
            ld[p, a[i]] -= 1  # remove i itself, then cost_row adds it back (+1)
            return _cost_row(v.eta[i], ld[p], v.free[p], 10.0, 2.0, w[i], cap[i])[a[i]]

        d_phi = potential(v.eta, v.needs_icu, v.free, v.inflight, a2, 10.0, 2.0, w, cap) - potential(
            v.eta, v.needs_icu, v.free, v.inflight, list(asg), 10.0, 2.0, w, cap)
        assert abs(d_phi - (own_cost(a2) - own_cost(list(asg)))) < 1e-6
