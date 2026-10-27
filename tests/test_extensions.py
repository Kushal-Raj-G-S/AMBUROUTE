import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from amburoute.geo import load_hospitals
from amburoute.nash import NashPolicy
from amburoute.online import OnlineETA
from amburoute.policies import CapacityHeuristic
from amburoute.sim import SCENARIOS, SimConfig, run_episode
from amburoute.traffic import load_calibrated

H = load_hospitals()
T = load_calibrated()
ROOT = Path(__file__).resolve().parent.parent


def test_new_hooks_leave_saved_paper_results_unchanged():
    saved = pd.read_csv(ROOT / "results" / "final_raw.csv")
    row = saved[(saved.scenario == "peak") & (saved.policy == "capacity_heuristic") & (saved.seed == 1000)].iloc[0]
    w = __import__("json").loads((ROOT / "results" / "best_baseline_params.json").read_text())["capacity_heuristic"]["w"]
    r = run_episode(H, SCENARIOS["peak"], 1000, CapacityHeuristic(w), traffic=T)
    assert abs(r["ttb_mean"] - row.ttb_mean) < 1e-9 and r["rejections"] == row.rejections


def test_in_flight_ablation_changes_decisions_and_default_does_not():
    a = run_episode(H, SCENARIOS["surge"], 3, NashPolicy(10, 2), traffic=T)
    b = run_episode(H, SCENARIOS["surge"], 3, NashPolicy(10, 2, use_inflight=False), traffic=T)
    c = run_episode(H, SCENARIOS["surge"], 3, NashPolicy(10, 2, use_inflight=True), traffic=T)
    assert a["ttb_mean"] == c["ttb_mean"] and a["ttb_mean"] != b["ttb_mean"]


def test_drift_slows_trips_and_is_off_by_default():
    base = run_episode(H, SCENARIOS["peak"], 4, CapacityHeuristic(20), traffic=T)
    off = run_episode(H, SCENARIOS["peak"], 4, CapacityHeuristic(20), SimConfig(drift_n=0), T)
    slow = run_episode(H, SCENARIOS["peak"], 4, CapacityHeuristic(20), SimConfig(drift_n=-1, drift_factor=1.5), T)
    assert base["ttb_mean"] == off["ttb_mean"] and slow["ttb_mean"] > base["ttb_mean"]


def test_online_learner_forgets_between_episodes_and_is_reproducible():
    pol = OnlineETA(CapacityHeuristic(20), 0.3)
    cfg = SimConfig(drift_n=4, drift_factor=2.0)
    a = run_episode(H, SCENARIOS["peak"], 6, pol, cfg, T)["ttb_mean"]
    run_episode(H, SCENARIOS["peak"], 7, pol, cfg, T)
    b = run_episode(H, SCENARIOS["peak"], 6, pol, cfg, T)["ttb_mean"]
    assert a == b


def test_clinical_run_reports_survival_and_door_wait():
    from amburoute.clinical import ClinicalHeuristic, with_capabilities

    r = run_episode(with_capabilities(H), SCENARIOS["peak"], 2, ClinicalHeuristic(80), SimConfig(clinical=True), T)
    assert 0 < r["surv_mean"] < 1 and r["door_wait_mean"] >= 0


def test_explicit_arrivals_replace_poisson_calls():
    arr = np.array([1.2, 1.9, 5.5, 5.6, 30.0])
    from amburoute.sim import Sim

    sim = Sim(H, SCENARIOS["peak"], 1, SimConfig(), T, arrivals=arr)
    n = 0
    while (batch := sim.next_batch()) is not None:
        n += len(batch)
        for r in batch:
            sim.dispatch(r, 0)
    assert n == len(arr)


def test_api_clinical_routing_and_online_feedback():
    from fastapi.testclient import TestClient

    from amburoute.api import app

    c = TestClient(app)
    c.post("/hospitals/simulate-updates", params={"steps": 1})
    body = {"ambulance": {"lat": 12.97, "lon": 77.59}, "severity": 2, "specialty": "cardiac"}
    r = c.post("/route", json=body)
    assert r.status_code == 200 and r.json()["meta"]["selection_method"] == "clinical_heuristic"
    assert r.json()["hospital"]["capabilities"]["cardiac"] is True
    assert c.post("/route", json={**body, "specialty": "dentist"}).status_code == 422
    r = c.post("/assign", json={"ambulances": [{"lat": 12.97, "lon": 77.59}] * 4, "severity": [2, 2, 1, 0],
                                "specialty": ["stroke", "cardiac", "none", "none"]})
    assert r.status_code == 200 and r.json()["converged"]
    a = r.json()["assignments"]
    assert a[0]["hospital"]["capabilities"]["stroke"] and a[1]["hospital"]["capabilities"]["cardiac"]
    assert c.post("/assign", json={"ambulances": [{"lat": 12.97, "lon": 77.59}] * 2, "severity": [1]}).status_code == 422
    before = c.post("/feedback", json={"hospital_id": 3, "planned_min": 10, "observed_min": 10}).json()["correction_factors"]
    for _ in range(6):
        out = c.post("/feedback", json={"hospital_id": 3, "planned_min": 10, "observed_min": 20}).json()["correction_factors"]
    assert out[3] > 1.1 and out[3] > before[3] and c.post("/feedback", json={"hospital_id": 99, "planned_min": 1, "observed_min": 1}).status_code == 404
    assert c.post("/route", json={"ambulance": {"lat": 12.97, "lon": 77.59}, "use_online_correction": True}).status_code == 200


def test_real_capacity_is_off_by_default_and_reproducible():
    from amburoute.sim import Sim

    a = Sim(H, SCENARIOS["peak"], 5, SimConfig(), T)
    b = Sim(H, SCENARIOS["peak"], 5, SimConfig(real_capacity=False), T)
    assert (a.total == b.total).all() and (a.target == b.target).all()
    c = Sim(H, SCENARIOS["peak"], 5, SimConfig(real_capacity=True), T)
    d = Sim(H, SCENARIOS["peak"], 5, SimConfig(real_capacity=True), T)
    e = Sim(H, SCENARIOS["peak"], 6, SimConfig(real_capacity=True), T)
    assert (c.total == d.total).all() and (c.target == d.target).all()  # same seed, same draw
    assert not (c.target == e.target).all()  # another seed draws different hospitals
    assert (c.target >= 0).all() and (c.target <= 1).all() and (c.total[1] >= 2).all()
    assert (c.total[0] == a.total[0]).all()  # Bengaluru's real bed totals are kept


def test_capacity_sampler_matches_size_and_works_for_any_city():
    import numpy as np

    from amburoute.capacity import sample_capacity

    rng = np.random.default_rng(0)
    occ, share, iocc = sample_capacity(np.array([60, 250, 900] * 40), rng)
    assert len(occ) == 120 and ((0 <= occ) & (occ <= 1)).all() and ((0 < share) & (share <= 0.6)).all()
    # larger hospitals are fuller on average in the measured data
    assert occ[2::3].mean() > occ[0::3].mean()


def test_capacity_profiles_select_by_name():
    import numpy as np

    from amburoute.capacity import PROFILES, sample_capacity
    from amburoute.sim import Sim

    for name in PROFILES:
        occ, share, iocc = sample_capacity(np.array([100, 500, 1500]), np.random.default_rng(1), name)
        assert len(occ) == 3 and (occ <= 1).all() and (share > 0).all()
    a = Sim(H, SCENARIOS["peak"], 3, SimConfig(real_capacity="england-2026"), T)
    b = Sim(H, SCENARIOS["peak"], 3, SimConfig(real_capacity="2022"), T)
    assert a.target[0].mean() > b.target[0].mean()  # English trusts run fuller than US hospitals
    try:
        Sim(H, SCENARIOS["peak"], 3, SimConfig(real_capacity="nowhere"), T)
        raise AssertionError("expected an error for an unknown profile")
    except ValueError:
        pass
