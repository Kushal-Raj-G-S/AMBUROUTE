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
