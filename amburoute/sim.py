"""Time-stepped (1 min) simulator of ambulance dispatch to hospitals.

What makes the problem hard (and what the paper's Nash step is meant to fix):
  * Hospitals publish bed counts only every `report_interval` minutes, and a
    published count never includes ambulances already on the road.
  * Hospitals also admit walk-in patients, so beds fill and free up by
    themselves (birth-death process around each hospital's initial occupancy).
  * When an ambulance arrives and no bed of the needed type is free, it is
    diverted to another hospital, paying extra travel + handover time.

Two bed pools per hospital: 0 = general, 1 = ICU.
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .geo import BBOX, haversine_km, pairwise_km
from .traffic import TrafficModel


@dataclass(frozen=True)
class SimConfig:
    window_min: int = 2  # requests are dispatched in batches every N minutes
    report_interval: int = 5  # bed counts refresh every N minutes
    handover_min: float = 5.0  # time lost on each rejected arrival
    max_diversions: int = 4
    boarding_penalty_min: float = 60.0  # after max_diversions: forced boarding
    icu_prob: float = 0.25  # share of patients needing ICU
    los_min: tuple = (4320.0, 4320.0)  # mean length of stay (general, ICU)
    occ_range: tuple = (0.88, 0.99)  # initial occupancy range per hospital
    traffic_aware_planner: bool = True  # False = routing API down: free-flow ETAs
    # --- clinical extension (off by default; see amburoute/clinical.py). All values are ASSUMPTIONS.
    clinical: bool = False
    sev_probs: tuple = (0.55, 0.30, 0.15)  # stable, urgent, critical
    spec_prob: tuple = (0.0, 0.25, 0.85)  # P(needs a specialty centre | severity)
    spec_mix: tuple = (0.4, 0.3, 0.3)  # cardiac, stroke, trauma
    icu_by_severity: tuple = (0.05, 0.30, 0.80)  # P(needs ICU | severity), about 24 % overall
    transfer_delay_min: float = 15.0  # overhead of an inter-hospital transfer
    # Survival proxy S = exp(-hazard * minutes_to_definitive_care / 60) by severity. ASSUMED shape and rates,
    # used only to compare policies (see docs/multiobjective.md).
    hazard_per_hour: tuple = (0.05, 0.25, 0.90)
    # Drift study (off by default): from `drift_start` minutes on, trips to `drift_n` randomly chosen hospitals
    # take `drift_factor` times longer than the planner's model says (local jams); drift_n = -1 slows every trip.
    drift_n: int = 0
    drift_factor: float = 1.0
    drift_start: int = 30
    obs_frac: float = 0.35  # online learners see a trip's realised/planned time this far into the trip (GPS tracking, assumed)


@dataclass(frozen=True)
class Scenario:
    name: str
    lam: float  # ambulance requests per minute (citywide)
    duration: int  # minutes during which requests arrive
    start_min: int  # minute of day at start (drives traffic)


SCENARIOS = {
    "offpeak": Scenario("offpeak", lam=0.5, duration=120, start_min=13 * 60),
    "peak": Scenario("peak", lam=1.2, duration=120, start_min=17 * 60 + 30),
    "surge": Scenario("surge", lam=2.5, duration=60, start_min=18 * 60),
}


@dataclass
class Request:
    rid: int
    t_call: float
    lat: float
    lon: float
    needs_icu: bool
    t_dispatch: float = float("nan")
    first_hospital: int = -1
    diversions: int = 0
    t_done: float = float("nan")
    boarded: bool = False
    tried: set = field(default_factory=set)
    severity: int = 0  # 0 stable, 1 urgent, 2 critical (clinical mode)
    spec: int = 0  # 0 none, 1 cardiac, 2 stroke, 3 trauma (clinical mode)
    transferred: bool = False
    t_first_arrival: float = float("nan")  # first time an ambulance reached a hospital door

    @property
    def pool(self) -> int:
        return int(self.needs_icu)


@dataclass
class View:
    """What a policy is allowed to see at dispatch time."""

    minute_of_day: float
    eta: np.ndarray  # (n, H) planner ETA in minutes
    needs_icu: np.ndarray  # (n,) bool
    free: np.ndarray  # (2, H) *published* free beds
    inflight: np.ndarray  # (2, H) ambulances dispatched, not yet arrived
    total: np.ndarray  # (2, H) bed capacity
    capable: np.ndarray | None = None  # (n, H) hospital can give definitive care (clinical mode)
    severity: np.ndarray | None = None  # (n,) 0 stable, 1 urgent, 2 critical (clinical mode)


class Sim:
    def __init__(
        self,
        hospitals: pd.DataFrame,
        scenario: Scenario,
        seed: int,
        cfg: SimConfig = SimConfig(),
        traffic: TrafficModel | None = None,
        planner_traffic: TrafficModel | None = None,
    ):
        self.h = hospitals
        self.H = len(hospitals)
        self.sc = scenario
        self.cfg = cfg
        self.traffic = traffic or TrafficModel()
        self.planner_traffic = planner_traffic or self.traffic  # model the planner uses for ETAs
        self.rng = np.random.default_rng(seed)
        self.hlat = hospitals["lat"].to_numpy()
        self.hlon = hospitals["lon"].to_numpy()
        self.hh_km = pairwise_km(hospitals)
        self.total = np.stack(
            [hospitals["beds"].to_numpy(), hospitals["icu_beds"].to_numpy()]
        ).astype(int)
        occ0 = self.rng.uniform(*cfg.occ_range, size=(2, self.H))
        self.target = occ0
        self.occ = np.floor(occ0 * self.total).astype(int)
        self.published = self.total - self.occ
        self.inflight = np.zeros((2, self.H), dtype=int)
        self.t = 0
        self.rid = 0
        self.pending: list[Request] = []
        self.events: list = []  # heap of (t_arrive, rid, hospital)
        self.reqs: dict[int, Request] = {}
        self.rejections = np.zeros(self.H, dtype=int)
        self.arrivals = np.zeros(self.H, dtype=int)  # first-choice dispatches
        self.max_inflight_ratio = np.zeros(self.H)
        self._los = np.array(cfg.los_min)[:, None]
        self.drift = np.ones(self.H)  # per-hospital slowdown of trips that depart after drift_start
        if cfg.drift_n:
            drng = np.random.default_rng(seed + 777)  # separate stream: the drift never changes the base episode
            hit = np.arange(self.H) if cfg.drift_n < 0 else drng.choice(self.H, cfg.drift_n, replace=False)
            self.drift[hit] = cfg.drift_factor
        self.obs: list = []  # (hospital, realised / planned travel time) of finished trips, drained by the policy
        self._obs_heap: list = []  # (time revealed, hospital, ratio)
        self.cap = np.ones((self.H, 4), dtype=bool)  # column 0 = no specialty needed
        if cfg.clinical:
            cols = [hospitals[c].to_numpy(bool) for c in ("cap_cardiac", "cap_stroke", "cap_trauma")]
            self.cap = np.stack([np.ones(self.H, dtype=bool)] + cols, axis=1)

    def _assign_clinical(self, r: Request):
        c = self.cfg
        r.severity = int(self.rng.choice(3, p=c.sev_probs))
        r.needs_icu = bool(self.rng.random() < c.icu_by_severity[r.severity])
        if self.rng.random() < c.spec_prob[r.severity]:
            r.spec = 1 + int(self.rng.choice(3, p=c.spec_mix))

    # ------------------------------------------------------------ time stepping
    def _minute_of_day(self) -> float:
        return self.sc.start_min + self.t

    def _background(self):
        dis = self.rng.binomial(self.occ, 1.0 / self._los)
        adm = self.rng.poisson(self.target * self.total / self._los)
        room = self.total - (self.occ - dis)
        self.occ = self.occ - dis + np.minimum(adm, room)

    def _process_arrivals(self):
        while self.events and self.events[0][0] <= self.t:
            t_arr, rid, h = heapq.heappop(self.events)
            self._arrive(self.reqs[rid], h, t_arr)

    def _arrive(self, req: Request, h: int, t_arr: float):
        p = req.pool
        self.inflight[p, h] -= 1
        if np.isnan(req.t_first_arrival):
            req.t_first_arrival = t_arr
        if self.occ[p, h] < self.total[p, h]:
            self.occ[p, h] += 1
            req.t_done = t_arr
            if self.cfg.clinical and req.spec and not self.cap[h, req.spec]:
                # Stabilised here, then transferred to the nearest capable hospital (bed there assumed).
                capable = np.flatnonzero(self.cap[:, req.spec])
                if len(capable):
                    j = int(capable[np.argmin(self.hh_km[h, capable])])
                    travel = float(self.traffic.actual_min(
                        self.hh_km[h, j], self.sc.start_min + t_arr, self.rng))
                    req.t_done = t_arr + self.cfg.transfer_delay_min + travel
                    req.transferred = True
            return
        self.rejections[h] += 1
        req.diversions += 1
        if req.diversions > self.cfg.max_diversions:
            req.boarded = True
            req.t_done = t_arr + self.cfg.boarding_penalty_min
            return
        # Crew calls around: nearest untried hospital with a published bed.
        cand = [j for j in range(self.H) if j not in req.tried]
        if self.cfg.clinical and req.spec:  # prefer hospitals that can treat the patient
            cand = [j for j in cand if self.cap[j, req.spec]] or cand
        with_beds = [j for j in cand if self.published[p, j] > 0]
        cand = with_beds or cand
        h2 = min(cand, key=lambda j: self.hh_km[h, j])
        self._send(req, h2, t_arr + self.cfg.handover_min, self.hh_km[h, h2])

    def _send(self, req: Request, h: int, t_depart: float, dist_km: float):
        mod = self.sc.start_min + t_depart
        travel = float(self.traffic.actual_min(dist_km, mod, self.rng))
        if self.cfg.drift_n and t_depart >= self.cfg.drift_start:
            travel *= self.drift[h]
        ratio = travel / max(float(self.planner_traffic.eta_min(dist_km, mod)), 1e-6)
        heapq.heappush(self._obs_heap, (t_depart + self.cfg.obs_frac * travel, h, ratio))
        req.tried.add(h)
        self.inflight[req.pool, h] += 1
        heapq.heappush(self.events, (t_depart + travel, req.rid, h))

    def next_batch(self) -> list[Request] | None:
        """Advance to the next dispatch window that has requests, or None."""
        while self.t < self.sc.duration:
            self.t += 1
            self._background()
            while self._obs_heap and self._obs_heap[0][0] <= self.t:
                _, oh, orat = heapq.heappop(self._obs_heap)
                self.obs.append((oh, orat))
            if self.t % self.cfg.report_interval == 0:
                self.published = self.total - self.occ
            self._process_arrivals()
            for _ in range(self.rng.poisson(self.sc.lam)):
                self.rid += 1
                r = Request(
                    self.rid,
                    self.t - self.rng.random(),
                    self.rng.uniform(BBOX["min_lat"], BBOX["max_lat"]),
                    self.rng.uniform(BBOX["min_lon"], BBOX["max_lon"]),
                    bool(self.rng.random() < self.cfg.icu_prob),
                )
                if self.cfg.clinical:
                    self._assign_clinical(r)
                self.reqs[r.rid] = r
                self.pending.append(r)
            if self.t % self.cfg.window_min == 0 and self.pending:
                batch, self.pending = self.pending, []
                return batch
        if self.pending:
            batch, self.pending = self.pending, []
            return batch
        return None

    # ----------------------------------------------------------------- policy IO
    def _dist(self, reqs: list[Request]) -> np.ndarray:
        lat = np.array([r.lat for r in reqs])[:, None]
        lon = np.array([r.lon for r in reqs])[:, None]
        return haversine_km(lat, lon, self.hlat[None, :], self.hlon[None, :])

    def make_view(self, reqs: list[Request]) -> View:
        mod = self._minute_of_day()
        # Planner without live traffic assumes night-time free-flow speed.
        plan_mod = mod if self.cfg.traffic_aware_planner else 3 * 60
        capable = severity = None
        if self.cfg.clinical:
            capable = np.stack([self.cap[:, r.spec] for r in reqs])
            severity = np.array([r.severity for r in reqs])
        return View(
            minute_of_day=mod,
            eta=self.planner_traffic.eta_min(self._dist(reqs), plan_mod),
            needs_icu=np.array([r.needs_icu for r in reqs]),
            free=self.published.copy(),
            inflight=self.inflight.copy(),
            total=self.total.copy(),
            capable=capable,
            severity=severity,
        )

    def dispatch(self, req: Request, h: int):
        req.t_dispatch = float(self.t)
        req.first_hospital = h
        self.arrivals[h] += 1
        d = float(self._dist([req])[0, h])
        self._send(req, h, float(self.t), d)
        ratio = self.inflight[req.pool, h] / max(self.published[req.pool, h], 1)
        self.max_inflight_ratio[h] = max(self.max_inflight_ratio[h], ratio)

    def finish(self, latencies_ms: list[float] | None = None) -> dict:
        while self.events:
            t_arr, rid, h = heapq.heappop(self.events)
            self.t = max(self.t, int(t_arr))
            self._arrive(self.reqs[rid], h, t_arr)
        done = [r for r in self.reqs.values() if not np.isnan(r.t_done)]
        ttb = np.array([r.t_done - r.t_call for r in done])
        div = np.array([r.diversions for r in done])
        icu = np.array([r.needs_icu for r in done])
        free0 = np.maximum(self.published.sum(axis=0), 1)
        share = self.arrivals / np.maximum(self.total[0], 1)
        jain = (share.sum() ** 2) / (self.H * (share**2).sum()) if share.sum() else 1.0
        extra = {}
        if self.cfg.clinical and len(done):
            sev = np.array([r.severity for r in done])
            spec = np.array([r.spec for r in done]) > 0
            moved = np.array([r.transferred for r in done])
            crit = sev == 2
            surv = np.exp(-np.array(self.cfg.hazard_per_hour)[sev] * ttb / 60.0)
            door = np.array([r.t_done - r.t_first_arrival for r in done])
            extra = dict(
                surv_mean=float(surv.mean()),
                surv_crit=float(surv[crit].mean()) if crit.any() else float("nan"),
                door_wait_mean=float(np.nanmean(door)),
                door_wait_p95=float(np.nanpercentile(door, 95)),
                crit_ttdc_mean=float(ttb[crit].mean()) if crit.any() else float("nan"),
                crit_ttdc_p95=float(np.percentile(ttb[crit], 95)) if crit.any() else float("nan"),
                spec_ttdc_mean=float(ttb[spec].mean()) if spec.any() else float("nan"),
                transfer_rate=float(moved[spec].mean()) if spec.any() else 0.0,
                n_spec=int(spec.sum()),
            )
        return dict(
            **extra,
            n=len(done),
            ttb_mean=float(ttb.mean()) if len(ttb) else float("nan"),
            ttb_median=float(np.median(ttb)) if len(ttb) else float("nan"),
            ttb_p95=float(np.percentile(ttb, 95)) if len(ttb) else float("nan"),
            diversion_rate=float((div > 0).mean()) if len(div) else 0.0,
            mean_diversions=float(div.mean()) if len(div) else 0.0,
            icu_diversion_rate=float((div[icu] > 0).mean()) if icu.any() else 0.0,
            boarded=int(sum(r.boarded for r in done)),
            rejections=int(self.rejections.sum()),
            max_rejections_one_hospital=int(self.rejections.max()),
            jain_load_balance=float(jain),
            decision_ms=float(np.mean(latencies_ms)) if latencies_ms else float("nan"),
        )


def run_episode(hospitals, scenario: Scenario, seed: int, policy, cfg=SimConfig(), traffic=None,
                planner_traffic=None) -> dict:
    """Run one full episode with a batch policy exposing `assign(view, rng)`."""
    sim = Sim(hospitals, scenario, seed, cfg, traffic, planner_traffic)
    prng = np.random.default_rng(seed + 10_000)
    lat_ms: list[float] = []
    while (batch := sim.next_batch()) is not None:
        if hasattr(policy, "observe"):  # online learners see what finished trips actually took
            policy.observe(sim.obs, sim)
            sim.obs.clear()
        view = sim.make_view(batch)
        t0 = time.perf_counter()
        assignment = policy.assign(view, prng)
        lat_ms.append((time.perf_counter() - t0) * 1000 / len(batch))
        for r, h in zip(batch, assignment):
            sim.dispatch(r, int(h))
    return sim.finish(lat_ms)
