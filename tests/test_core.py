import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from fastapi.testclient import TestClient

from amburoute.geo import haversine_km, load_hospitals
from amburoute.nash import nash_assign, potential
from amburoute.sim import SCENARIOS, Sim, View, run_episode
from amburoute.policies import Nearest, CapacityHeuristic


def test_haversine_known_distance():
    # Bengaluru centre -> Kempegowda airport is ~ 30 km straight line
    d = haversine_km(12.9716, 77.5946, 13.1986, 77.7066)
    assert 25 < d < 30


def test_hospital_table():
    h = load_hospitals()
    assert len(h) > 10 and (h.beds > 0).all() and (h.icu_beds >= 4).all()


def test_sim_invariants():
    h = load_hospitals()
    sim = Sim(h, SCENARIOS["surge"], seed=3)
    pol = Nearest()
    rng = np.random.default_rng(0)
    while (batch := sim.next_batch()) is not None:
        for r, a in zip(batch, pol.assign(sim.make_view(batch), rng)):
            sim.dispatch(r, int(a))
        assert (sim.occ >= 0).all() and (sim.occ <= sim.total).all()
        assert (sim.inflight >= 0).all()
    m = sim.finish()
    assert (sim.inflight == 0).all()
    assert m["n"] == len(sim.reqs)  # every patient eventually gets a bed


def test_sim_deterministic():
    h = load_hospitals()
    a = run_episode(h, SCENARIOS["peak"], 5, CapacityHeuristic())
    b = run_episode(h, SCENARIOS["peak"], 5, CapacityHeuristic())
    a.pop("decision_ms"); b.pop("decision_ms")
    assert a == b


def _random_view(n, H, seed):
    rng = np.random.default_rng(seed)
    return View(
        720.0, rng.uniform(3, 30, (n, H)), rng.random(n) < 0.3,
        rng.integers(0, 15, (2, H)), rng.integers(0, 3, (2, H)),
        np.full((2, H), 100),
    )


def test_nash_converges_and_is_equilibrium():
    for seed in range(20):
        v = _random_view(12, 10, seed)
        asg, iters, ok = nash_assign(v, 30.0, np.random.default_rng(seed))
        assert ok
        # No ambulance can lower its own cost by a unilateral move.
        pool = v.needs_icu.astype(int)
        load = v.inflight.astype(float).copy()
        for i, h in enumerate(asg):
            load[pool[i], h] += 1
        for i, h in enumerate(asg):
            p = pool[i]
            load[p, h] -= 1
            cost = v.eta[i] + 30.0 * (load[p] + 1) / np.maximum(v.free[p], 1)
            cost += np.where(v.free[p] <= 0, 1e3, 0)
            assert cost[h] <= cost.min() + 1e-6
            load[p, h] += 1


def test_nash_improvement_lowers_potential():
    v = _random_view(10, 8, 1)
    greedy_like = [int(np.argmin(v.eta[i])) for i in range(10)]
    asg, _, _ = nash_assign(v, 30.0, np.random.default_rng(0))
    args = (v.eta, v.needs_icu, v.free, v.inflight)
    assert potential(*args, asg, 30.0) <= potential(*args, greedy_like, 30.0) + 1e-9


def test_api_endpoints():
    from amburoute.api import app

    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    hosp = c.get("/hospitals").json()
    assert len(hosp) > 10
    r = c.post("/hospitals/0/beds", json={"free_beds": 7})
    assert r.json()["available_beds"] == 7
    assert c.post("/hospitals/999/beds", json={"free_beds": 1}).status_code == 404
    assert c.post("/hospitals/simulate-updates").status_code == 200
    body = {"ambulance": {"lat": 12.97, "lon": 77.59}, "needs_icu": False}
    r = c.post("/route", json=body)
    assert r.status_code == 200 and r.json()["route"]["geometry"]
    assert c.post("/route/drl", json=body).status_code == 200
    assert c.post("/route", json={"ambulance": {"lat": 999, "lon": 0}}).status_code == 422
    r = c.post("/assign", json={"ambulances": [{"lat": 12.97, "lon": 77.59}] * 6})
    assert r.status_code == 200 and r.json()["converged"]
