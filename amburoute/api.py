"""FastAPI service. Run: uvicorn amburoute.api:app --reload"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .geo import haversine_km, load_hospitals
from .nash import nash_assign
from .policies import CapacityHeuristic
from .routing import get_route
from .sim import View
from .traffic import TrafficModel

ROOT = Path(__file__).resolve().parent.parent


def _default_model() -> Path:
    sel = ROOT / "results" / "selected_ppo.txt"
    name = sel.read_text().strip() if sel.exists() else "none"
    return ROOT / "results" / "models" / f"{name}.pt"


MODEL_PATH = Path(os.getenv("PPO_MODEL", _default_model()))

app = FastAPI(title="AmbuRoute", version="0.2.0")


def _make_traffic():
    cal = ROOT / "results" / "traffic_calibration.json"
    if cal.exists():
        from .traffic import load_calibrated

        return load_calibrated()
    return TrafficModel()


_traffic = _make_traffic()
_rng = np.random.default_rng(0)

# In-memory hospital state. Bed counts are SIMULATED (see README).
_hosp = load_hospitals()
_free = np.stack([
    (_hosp["beds"] * np.random.default_rng(1).uniform(0.02, 0.12, len(_hosp))).astype(int),
    (_hosp["icu_beds"] * np.random.default_rng(2).uniform(0.02, 0.15, len(_hosp))).astype(int),
])
_total = np.stack([_hosp["beds"].to_numpy(), _hosp["icu_beds"].to_numpy()])
_inflight = np.zeros_like(_free)
_ppo = None


def _get_ppo():
    global _ppo
    if _ppo is None and MODEL_PATH.exists():
        from .rl_env import PPOPolicy

        _ppo = PPOPolicy(MODEL_PATH)
    return _ppo


class Location(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class RouteRequest(BaseModel):
    ambulance: Location
    needs_icu: bool = False
    min_beds: int = Field(1, ge=0)
    minute_of_day: float = Field(12 * 60, ge=0, le=1440)


class AssignRequest(BaseModel):
    ambulances: list[Location] = Field(min_length=1, max_length=200)
    needs_icu: list[bool] | None = None
    beta: float = Field(30.0, ge=0)
    minute_of_day: float = Field(12 * 60, ge=0, le=1440)


class BedUpdate(BaseModel):
    free_beds: int | None = Field(None, ge=0)
    free_icu_beds: int | None = Field(None, ge=0)


def _view(locs, needs_icu, mod) -> View:
    lat = np.array([l.lat for l in locs])[:, None]
    lon = np.array([l.lon for l in locs])[:, None]
    d = haversine_km(lat, lon, _hosp["lat"].to_numpy()[None, :], _hosp["lon"].to_numpy()[None, :])
    return View(mod, _traffic.eta_min(d, mod), np.array(needs_icu, dtype=bool),
                _free.copy(), _inflight.copy(), _total.copy())


def _hospital_json(i: int) -> dict:
    h = _hosp.iloc[i]
    return dict(id=int(h["id"]), name=h["name"], lat=float(h["lat"]), lon=float(h["lon"]),
                total_beds=int(_total[0, i]), available_beds=int(_free[0, i]),
                total_icu_beds=int(_total[1, i]), available_icu_beds=int(_free[1, i]))


def _respond(req: RouteRequest, h: int, method: str, t0: float) -> dict:
    t1 = time.perf_counter()
    route = get_route((req.ambulance.lat, req.ambulance.lon),
                      (float(_hosp.loc[h, "lat"]), float(_hosp.loc[h, "lon"])), req.minute_of_day)
    t2 = time.perf_counter()
    if not route["geometry"]:
        raise HTTPException(502, "routing failed")
    return dict(hospital=_hospital_json(h), route=route,
                meta=dict(selection_method=method, infer_ms=(t1 - t0) * 1000,
                          route_ms=(t2 - t1) * 1000,
                          timestamp=datetime.now(timezone.utc).isoformat()))


def _candidates(req: RouteRequest) -> View:
    v = _view([req.ambulance], [req.needs_icu], req.minute_of_day)
    pool = int(req.needs_icu)
    ok = _free[pool] >= max(req.min_beds, 1)
    if not ok.any():
        raise HTTPException(404, "no hospital satisfies the bed requirements")
    v.free = v.free.copy()
    v.free[pool, ~ok] = 0  # hide hospitals below the requested minimum
    return v


@app.get("/health")
def health():
    return dict(status="ok", hospitals=len(_hosp), ppo_model_loaded=_get_ppo() is not None)


@app.get("/hospitals")
def list_hospitals():
    return [_hospital_json(i) for i in range(len(_hosp))]


@app.post("/hospitals/{hospital_id}/beds")
def update_beds(hospital_id: int, upd: BedUpdate):
    if not 0 <= hospital_id < len(_hosp):
        raise HTTPException(404, "unknown hospital")
    if upd.free_beds is not None:
        _free[0, hospital_id] = min(upd.free_beds, _total[0, hospital_id])
    if upd.free_icu_beds is not None:
        _free[1, hospital_id] = min(upd.free_icu_beds, _total[1, hospital_id])
    return _hospital_json(hospital_id)


@app.post("/hospitals/simulate-updates")
def simulate_updates(steps: int = 1):
    """Random-walk the bed counts (stand-in for a live hospital feed)."""
    for _ in range(max(1, min(steps, 1000))):
        delta = _rng.integers(-3, 4, size=_free.shape)
        _free[:] = np.clip(_free + delta, 0, _total)
    return dict(updated=len(_hosp))


@app.post("/route")
def route_heuristic(req: RouteRequest):
    """Hospital selection with the non-learned capacity heuristic."""
    t0 = time.perf_counter()
    v = _candidates(req)
    h = CapacityHeuristic().assign(v, _rng)[0]
    return _respond(req, int(h), "capacity_heuristic", t0)


@app.post("/route/drl")
def route_drl(req: RouteRequest):
    """PPO selection; falls back to Nash/heuristic if no model is available."""
    t0 = time.perf_counter()
    v = _candidates(req)
    ppo = _get_ppo()
    if ppo is not None:
        return _respond(req, int(ppo.assign(v, _rng)[0]), "ppo", t0)
    asg, _, _ = nash_assign(v, 30.0, _rng)
    return _respond(req, int(asg[0]), "nash_fallback", t0)


@app.post("/assign")
def assign_many(req: AssignRequest):
    """Joint assignment of several ambulances at a Nash equilibrium."""
    n = len(req.ambulances)
    needs = req.needs_icu or [False] * n
    if len(needs) != n:
        raise HTTPException(422, "needs_icu must match ambulances")
    v = _view(req.ambulances, needs, req.minute_of_day)
    t0 = time.perf_counter()
    asg, iters, converged = nash_assign(v, req.beta, _rng)
    ms = (time.perf_counter() - t0) * 1000
    load = {}
    for h in asg:
        load[int(h)] = load.get(int(h), 0) + 1
    return dict(
        assignments=[dict(ambulance=i, hospital=_hospital_json(int(h)),
                          eta_min=float(v.eta[i, h])) for i, h in enumerate(asg)],
        hospital_load=load, iterations=iters, converged=converged, solve_ms=ms)


# ----------------------------------------------------------------- console API
class PathRequest(BaseModel):
    ambulance: Location
    hospital_id: int
    minute_of_day: float = Field(12 * 60, ge=0, le=1440)


@app.get("/config")
def config():
    """Public map token for the browser (a Mapbox `pk.` token is public by design)."""
    return dict(mapbox_token=os.getenv("MAPBOX_TOKEN", ""),
                traffic_model="calibrated" if (ROOT / "results" / "traffic_calibration.json").exists() else "synthetic",
                ppo_available=_get_ppo() is not None)


@app.post("/route/path")
def route_to_hospital(req: PathRequest):
    if not 0 <= req.hospital_id < len(_hosp):
        raise HTTPException(404, "unknown hospital")
    h = _hosp.loc[req.hospital_id]
    return get_route((req.ambulance.lat, req.ambulance.lon), (float(h["lat"]), float(h["lon"])),
                     req.minute_of_day)


@app.get("/compare")
def compare(scenario: str = "surge", seeds: int = 8):
    """Run the simulator for several policies on the same random days."""
    from .nash import NashPolicy
    from .policies import Nearest, NearestWithBeds
    from .sim import SCENARIOS, run_episode

    if scenario not in SCENARIOS:
        raise HTTPException(404, f"scenario must be one of {list(SCENARIOS)}")
    seeds = max(1, min(seeds, 30))
    pols = [Nearest(), NearestWithBeds(), CapacityHeuristic(20.0), NashPolicy(20.0, 2.0)]
    out = []
    for p in pols:
        rs = [run_episode(_hosp, SCENARIOS[scenario], 7000 + s, p, traffic=_traffic) for s in range(seeds)]
        out.append(dict(policy=p.name,
                        mean_time_to_bed_min=float(np.mean([r["ttb_mean"] for r in rs])),
                        p95_time_to_bed_min=float(np.mean([r["ttb_p95"] for r in rs])),
                        rejected_arrivals=float(np.mean([r["rejections"] for r in rs])),
                        load_balance_jain=float(np.mean([r["jain_load_balance"] for r in rs]))))
    return dict(scenario=scenario, seeds=seeds, results=out)


@app.get("/")
def index():
    return FileResponse(Path(__file__).resolve().parent / "web" / "index.html")


app.mount("/static", StaticFiles(directory=str(Path(__file__).resolve().parent / "web")), name="static")
