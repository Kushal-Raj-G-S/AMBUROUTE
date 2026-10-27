"""FastAPI service. Run: uvicorn amburoute.api:app --reload"""
from __future__ import annotations

import hmac
import json
import logging
import os
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .clinical import ClinicalHeuristic, ClinicalNash, with_capabilities
from .geo import haversine_km, load_hospitals
from .nash import nash_assign
from .online import OnlineETA
from .policies import CapacityHeuristic
from .routing import get_route
from .sim import View
from .traffic import TrafficModel

ROOT = Path(__file__).resolve().parent.parent

# --- configuration (environment only; see .env.example) ---
API_KEY = os.getenv("API_KEY", "")  # if set, write endpoints require the X-API-Key header
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
MAX_AMBULANCES = int(os.getenv("MAX_AMBULANCES", "200"))
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "0"))  # per client address; 0 = off
STALE_MINUTES = float(os.getenv("STALE_MINUTES", "15"))  # bed reports older than this are flagged
STATE_PATH = os.getenv("STATE_PATH", "")  # JSON file that keeps bed state and learned corrections across restarts
SHADOW_LOG = os.getenv("SHADOW_LOG", "")  # JSONL file of recommendations vs dispatcher decisions

log = logging.getLogger("amburoute")
if not log.handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")


def _default_model() -> Path:
    sel = ROOT / "results" / "selected_ppo.txt"
    name = sel.read_text().strip() if sel.exists() else "none"
    return ROOT / "results" / "models" / f"{name}.pt"


MODEL_PATH = Path(os.getenv("PPO_MODEL", _default_model()))

app = FastAPI(title="AmbuRoute", version="0.3.0")
if CORS_ORIGINS:
    app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type", "X-API-Key"])
_hits: dict = defaultdict(deque)


@app.middleware("http")
async def guard_and_log(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    t0 = time.perf_counter()
    if RATE_LIMIT_PER_MIN and request.url.path not in ("/health", "/ready"):
        q = _hits[request.client.host if request.client else "?"]
        now = time.monotonic()
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_LIMIT_PER_MIN:
            return JSONResponse({"detail": "rate limit exceeded"}, status_code=429, headers={"X-Request-ID": rid})
        q.append(now)
    resp = await call_next(request)
    resp.headers["X-Request-ID"] = rid
    log.info(json.dumps(dict(ts=datetime.now(timezone.utc).isoformat(), id=rid, method=request.method,
                             path=request.url.path, status=resp.status_code,
                             ms=round((time.perf_counter() - t0) * 1000, 1))))
    return resp


def require_key(x_api_key: str | None = Header(None)):
    if API_KEY and not hmac.compare_digest(x_api_key or "", API_KEY):
        raise HTTPException(401, "missing or invalid X-API-Key")


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

# Specialty capabilities per hospital: from public sources when data/hospital_capabilities_researched.csv exists
# (a service is counted only if a source states it; see its `verification` column), otherwise ASSUMED from hospital size.
SPECIALTIES = {"none": 0, "cardiac": 1, "stroke": 2, "trauma": 3}
_cap_file = ROOT / "data" / "hospital_capabilities_researched.csv"
if _cap_file.exists():
    import pandas as _pd

    _c = _pd.read_csv(_cap_file).set_index("id")
    _cols = [_c.loc[_hosp["id"], k].to_numpy().astype(bool) for k in ("cardiac", "stroke", "trauma")]
    CAP_SOURCE = "public sources"
else:
    _c = with_capabilities(_hosp)
    _cols = [_c[k].to_numpy(bool) for k in ("cap_cardiac", "cap_stroke", "cap_trauma")]
    CAP_SOURCE = "assumed"
_cap = np.stack([np.ones(len(_hosp), dtype=bool)] + _cols, axis=1)
# Online correction learned from observed trip times (POST /feedback); relative per-hospital factors.
_online = OnlineETA(CapacityHeuristic(), 0.3, len(_hosp))
_online_session = object()
_online.observe([], _online_session)
_updated = np.full(len(_hosp), time.time())  # when each hospital last reported beds


def _save_state():
    if not STATE_PATH:
        return
    state = dict(free=_free.tolist(), updated=_updated.tolist(), g=float(_online.g),
                 logc=[None if np.isnan(x) else float(x) for x in _online.logc])
    tmp = Path(STATE_PATH).with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    os.replace(tmp, STATE_PATH)  # atomic: a crash never leaves a half-written file


def _load_state():
    if not STATE_PATH or not Path(STATE_PATH).exists():
        return
    st = json.loads(Path(STATE_PATH).read_text())
    if np.array(st["free"]).shape == _free.shape:
        _free[:] = np.minimum(np.array(st["free"], dtype=int), _total)
        _updated[:] = st["updated"]
        _online.g = float(st["g"])
        _online.logc[:] = [np.nan if x is None else x for x in st["logc"]]


_load_state()


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
    severity: int = Field(0, ge=0, le=2, description="0 stable, 1 urgent, 2 critical")
    specialty: str = Field("none", description="none, cardiac, stroke or trauma")
    use_online_correction: bool = False


class AssignRequest(BaseModel):
    ambulances: list[Location] = Field(min_length=1, max_length=200)
    needs_icu: list[bool] | None = None
    beta: float = Field(30.0, ge=0)
    minute_of_day: float = Field(12 * 60, ge=0, le=1440)
    severity: list[int] | None = Field(None, description="per ambulance: 0 stable, 1 urgent, 2 critical")
    specialty: list[str] | None = Field(None, description="per ambulance: none, cardiac, stroke or trauma")
    use_online_correction: bool = False


class Feedback(BaseModel):
    hospital_id: int
    planned_min: float = Field(gt=0)
    observed_min: float = Field(gt=0)


class BedUpdate(BaseModel):
    free_beds: int | None = Field(None, ge=0)
    free_icu_beds: int | None = Field(None, ge=0)


def _online_factors() -> np.ndarray:
    rel = np.where(np.isnan(_online.logc), 0.0, _online.logc - _online.g)
    return np.exp(rel)


def _view(locs, needs_icu, mod, severity=None, specialty=None, online=False) -> View:
    lat = np.array([p.lat for p in locs])[:, None]
    lon = np.array([p.lon for p in locs])[:, None]
    d = haversine_km(lat, lon, _hosp["lat"].to_numpy()[None, :], _hosp["lon"].to_numpy()[None, :])
    eta = _traffic.eta_min(d, mod)
    if online:
        eta = eta * _online_factors()[None, :]
    v = View(mod, eta, np.array(needs_icu, dtype=bool), _free.copy(), _inflight.copy(), _total.copy())
    if severity is not None:
        for sp in specialty:
            if sp not in SPECIALTIES:
                raise HTTPException(422, f"specialty must be one of {list(SPECIALTIES)}")
        v.severity = np.array(severity, dtype=int)
        v.capable = np.stack([_cap[:, SPECIALTIES[sp]] for sp in specialty])
    return v


def _hospital_json(i: int) -> dict:
    h = _hosp.iloc[i]
    return dict(id=int(h["id"]), name=h["name"], lat=float(h["lat"]), lon=float(h["lon"]),
                total_beds=int(_total[0, i]), available_beds=int(_free[0, i]),
                total_icu_beds=int(_total[1, i]), available_icu_beds=int(_free[1, i]),
                updated_at=datetime.fromtimestamp(float(_updated[i]), timezone.utc).isoformat(),
                stale=bool((time.time() - _updated[i]) / 60 > STALE_MINUTES),
                capabilities=dict(cardiac=bool(_cap[i, 1]), stroke=bool(_cap[i, 2]), trauma=bool(_cap[i, 3]),
                                  source=CAP_SOURCE))


def _explain(req: RouteRequest, h: int) -> dict:
    """The three nearest hospitals by ETA plus the chosen one, with what the dispatcher needs to judge the choice."""
    v = _view([req.ambulance], [req.needs_icu], req.minute_of_day)
    pool = int(req.needs_icu)
    sp = SPECIALTIES.get(req.specialty, 0)
    order = [int(i) for i in np.argsort(v.eta[0])[:3]]
    ids = order + ([h] if h not in order else [])
    alt = [dict(hospital=_hosp.loc[i, "name"], eta_min=round(float(v.eta[0, i]), 1), free_beds=int(_free[pool, i]),
                ambulances_on_the_way=int(_inflight[pool, i]), capable=bool(_cap[i, sp]),
                stale=bool((time.time() - _updated[i]) / 60 > STALE_MINUTES), chosen=(i == h)) for i in ids]
    near = order[0]
    if h == near:
        why = "nearest hospital that fits the request"
    else:
        why = (f"skipped the nearest ({_hosp.loc[near, 'name']}): {int(_free[pool, near])} free "
               f"{'ICU ' if pool else ''}beds, {int(_inflight[pool, near])} ambulances already on the way"
               + ("" if _cap[near, sp] else ", and it lacks the needed specialty"))
    return dict(alternatives=alt, reason=why)


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
                          timestamp=datetime.now(timezone.utc).isoformat(),
                          notice="Recommendation only; the dispatcher decides.",
                          **_explain(req, h)))


def _clinical(req: RouteRequest) -> bool:
    return req.severity > 0 or req.specialty != "none"


def _candidates(req: RouteRequest) -> View:
    v = _view([req.ambulance], [req.needs_icu], req.minute_of_day,
              [req.severity] if _clinical(req) else None, [req.specialty], req.use_online_correction)
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


@app.get("/ready")
def ready():
    checks = dict(traffic_model="calibrated" if (ROOT / "results" / "traffic_calibration.json").exists() else "synthetic",
                  hospitals=len(_hosp) > 0,
                  state_writable=(not STATE_PATH) or os.access(Path(STATE_PATH).parent, os.W_OK),
                  shadow_log_writable=(not SHADOW_LOG) or os.access(Path(SHADOW_LOG).parent, os.W_OK),
                  stale_hospitals=int(((time.time() - _updated) / 60 > STALE_MINUTES).sum()))
    ok = checks["hospitals"] and checks["state_writable"] and checks["shadow_log_writable"]
    return JSONResponse(dict(status="ready" if ok else "not_ready", **checks), status_code=200 if ok else 503)


class Decision(BaseModel):
    recommended_id: int
    chosen_id: int
    reason: str = Field("", max_length=300)


@app.post("/decision", dependencies=[Depends(require_key)])
def decision(d: Decision):
    """The dispatcher's final choice next to our recommendation (for shadow-mode evaluation). No patient data."""
    for i in (d.recommended_id, d.chosen_id):
        if not 0 <= i < len(_hosp):
            raise HTTPException(404, "unknown hospital")
    rec = dict(ts=datetime.now(timezone.utc).isoformat(), recommended=d.recommended_id, chosen=d.chosen_id,
               agreed=d.recommended_id == d.chosen_id, reason=d.reason)
    if SHADOW_LOG:
        with open(SHADOW_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    return rec


@app.post("/feedback", dependencies=[Depends(require_key)])
def feedback(f: Feedback):
    """Report how long a finished (or in-progress) trip took against the planned time. The service learns a relative
    per-hospital correction from these reports and applies it when a request sets use_online_correction."""
    if not 0 <= f.hospital_id < len(_hosp):
        raise HTTPException(404, "unknown hospital")
    _online.observe([(f.hospital_id, f.observed_min / f.planned_min)], _online_session)
    _save_state()
    return dict(hospital_id=f.hospital_id, correction_factors=[round(float(x), 3) for x in _online_factors()])


@app.get("/hospitals")
def list_hospitals():
    return [_hospital_json(i) for i in range(len(_hosp))]


@app.post("/hospitals/{hospital_id}/beds", dependencies=[Depends(require_key)])
def update_beds(hospital_id: int, upd: BedUpdate):
    if not 0 <= hospital_id < len(_hosp):
        raise HTTPException(404, "unknown hospital")
    if upd.free_beds is not None:
        _free[0, hospital_id] = min(upd.free_beds, _total[0, hospital_id])
    if upd.free_icu_beds is not None:
        _free[1, hospital_id] = min(upd.free_icu_beds, _total[1, hospital_id])
    _updated[hospital_id] = time.time()
    _save_state()
    return _hospital_json(hospital_id)


@app.post("/hospitals/simulate-updates", dependencies=[Depends(require_key)])
def simulate_updates(steps: int = 1):
    """Random-walk the bed counts (stand-in for a live hospital feed)."""
    for _ in range(max(1, min(steps, 1000))):
        delta = _rng.integers(-3, 4, size=_free.shape)
        _free[:] = np.clip(_free + delta, 0, _total)
    _updated[:] = time.time()
    _save_state()
    return dict(updated=len(_hosp))


@app.post("/route")
def route_heuristic(req: RouteRequest):
    """Hospital selection with the non-learned capacity heuristic."""
    t0 = time.perf_counter()
    v = _candidates(req)
    if _clinical(req):
        h = ClinicalHeuristic(80.0).assign(v, _rng)[0]
        return _respond(req, int(h), "clinical_heuristic", t0)
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
    if n > MAX_AMBULANCES:
        raise HTTPException(422, f"at most {MAX_AMBULANCES} ambulances per request")
    needs = req.needs_icu or [False] * n
    if len(needs) != n:
        raise HTTPException(422, "needs_icu must match ambulances")
    clinical = req.severity is not None or req.specialty is not None
    sev = req.severity or [0] * n
    spec = req.specialty or ["none"] * n
    if len(sev) != n or len(spec) != n:
        raise HTTPException(422, "severity and specialty must match ambulances")
    v = _view(req.ambulances, needs, req.minute_of_day, sev if clinical else None, spec, req.use_online_correction)
    t0 = time.perf_counter()
    if clinical:
        pol = ClinicalNash(req.beta, 1.0)
        asg = pol.assign(v, _rng)
        iters, converged = pol.last_iters, pol.last_converged
    else:
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
