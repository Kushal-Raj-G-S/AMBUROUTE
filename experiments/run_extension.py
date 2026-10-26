"""Extension studies beyond the paper (run after run_benchmark.py).

E1  Bed-data staleness up to 6 hours (Bengaluru's mandated bed-portal refresh interval).
E2  City-scale stress test with the 111 emergency-tagged OpenStreetMap hospitals.
    Bed counts for these hospitals are ASSUMED (OSM has almost none): drawn from a log-normal
    fitted to the 18 known hospitals. Geography is real; capacity is not.

    python experiments/run_extension.py
Outputs: data/osm_emergency_hospitals.csv, results/extension_staleness.csv, results/extension_scale.csv
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from amburoute.geo import ICU_FRACTION, haversine_km, load_hospitals
from amburoute.nash import NashPolicy
from amburoute.policies import CapacityHeuristic, Nearest, NearestWithBeds
from amburoute.sim import SCENARIOS, SimConfig, run_episode
from amburoute.traffic import load_calibrated

RES = ROOT / "results"
SEEDS = list(range(1000, 1030))  # 30 test seeds (extension, not the headline benchmark)
best = json.loads((RES / "best_baseline_params.json").read_text())
T = load_calibrated()


def policies():
    return [Nearest(), NearestWithBeds(), CapacityHeuristic(best["capacity_heuristic"]["w"]),
            NashPolicy(best["nash"]["beta"], best["nash"]["power"])]


# ---------------------------------------------------------------- E1: staleness up to 6 h
h18 = load_hospitals()
rows = []
for sc in ("peak", "surge"):
    for refresh in (5, 30, 60, 180, 360):
        cfg = SimConfig(report_interval=refresh)
        for p in policies():
            r = [run_episode(h18, SCENARIOS[sc], s, p, cfg, T) for s in SEEDS]
            rows.append(dict(study="staleness", scenario=sc, refresh_min=refresh, policy=p.name,
                             ttb_mean=np.mean([x["ttb_mean"] for x in r]),
                             rejections=np.mean([x["rejections"] for x in r])))
        print("E1", sc, refresh, flush=True)
pd.DataFrame(rows).round(3).to_csv(RES / "extension_staleness.csv", index=False)

# ---------------------------------------------------------------- E2: city scale (OSM geography)
raw = json.loads((ROOT / "data" / "osm_hospitals_raw.json").read_text(encoding="utf-8"))
recs = []
for e in raw["elements"]:
    t = e.get("tags", {})
    if t.get("emergency") != "yes" or not t.get("name"):
        continue
    lat = e.get("lat") or (e.get("center") or {}).get("lat")
    lon = e.get("lon") or (e.get("center") or {}).get("lon")
    if lat is None:
        continue
    recs.append((t["name"].strip(), float(lat), float(lon)))
df = pd.DataFrame(recs, columns=["name", "lat", "lon"])
keep = []  # drop same-name entries within 150 m (a hospital mapped as both a node and a way)
for i, r in df.iterrows():
    dup = any(r["name"].lower() == df.loc[j, "name"].lower()
              and float(haversine_km(r["lat"], r["lon"], df.loc[j, "lat"], df.loc[j, "lon"])) < 0.15 for j in keep)
    if not dup:
        keep.append(i)
df = df.loc[keep].reset_index(drop=True)

rng = np.random.default_rng(123)
mu, sd = np.log(h18["beds"]).mean(), np.log(h18["beds"]).std()
df["beds"] = np.clip(np.exp(rng.normal(mu, sd, len(df))), 30, 1500).round().astype(int)
df["icu_beds"] = np.maximum(4, (df["beds"] * ICU_FRACTION).round()).astype(int)
df["beds_imputed"] = True
df.insert(0, "id", df.index)
df.to_csv(ROOT / "data" / "osm_emergency_hospitals.csv", index=False)
ratio = df["beds"].sum() / h18["beds"].sum()
print(f"E2 hospitals: {len(df)}; total beds {df['beds'].sum()} ({ratio:.1f}x the 18-hospital set)", flush=True)

rows = []
for variant, scale in (("same calls", 1.0), ("same load per bed", ratio)):
    for scn in ("peak", "surge"):
        base = SCENARIOS[scn]
        sc = replace(base, lam=base.lam * scale)
        for p in policies():
            r = [run_episode(df, sc, s, p, SimConfig(), T) for s in SEEDS]
            rows.append(dict(study="scale", variant=variant, scenario=scn, hospitals=len(df), policy=p.name,
                             ttb_mean=np.mean([x["ttb_mean"] for x in r]),
                             ttb_p95=np.mean([x["ttb_p95"] for x in r]),
                             rejections=np.mean([x["rejections"] for x in r])))
        print("E2", variant, scn, flush=True)
out = pd.DataFrame(rows).round(3)
out.to_csv(RES / "extension_scale.csv", index=False)
pd.set_option("display.width", 200)
print(out.pivot_table(index=["variant", "scenario"], columns="policy", values=["ttb_mean", "rejections"]).round(2).to_string())
