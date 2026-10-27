"""Independent real-traffic check for Bengaluru, using an open community dataset.

Data: "Bangalore Traffic dataset", Mahesh Shantaram (2026), CC BY 4.0, https://github.com/thecont1/traffic-monitor-lizard
(Google Maps travel-time estimates for 13 routes of about 10 km, sampled about three times an hour since 2025-09-25).
It is NOT used to fit the travel-time model (that was fitted to Mapbox typical-traffic samples); it is used to
  1. validate the fitted model against an independent real source (hour-of-day profile, level of the travel times), and
  2. measure how large day-to-day slowdowns really are (ratio of a reading to the typical reading for the same route,
     weekday/weekend class and hour), to compare with the jam sizes assumed in the drift study.
Only aggregates are saved; the raw file is cached outside the repo.

    python experiments/analyse_gmaps_traffic.py [path_to_csv_dir]
Outputs: data/real_traffic_gmaps_profile.csv, results/gmaps_validation.csv, results/gmaps_variability.json
"""
import json
import sys
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from openlocationcode import openlocationcode as olc

from amburoute.geo import haversine_km
from amburoute.traffic import load_calibrated

BASE = "https://raw.githubusercontent.com/thecont1/traffic-monitor-lizard/main/data/"
cache = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / ".cache_gmaps"
cache.mkdir(exist_ok=True)
for f in ("csv-traffic-bangalore.csv", "csv-routes-bangalore.csv"):
    if not (cache / f).exists():
        urlretrieve(BASE + f, cache / f)
tr = pd.read_csv(cache / "csv-traffic-bangalore.csv", usecols=["date", "time", "route_code", "duration", "distance"])
rt = pd.read_csv(cache / "csv-routes-bangalore.csv")

tr = tr.dropna(subset=["duration", "distance"])
tr = tr[(tr.duration > 0) & (tr.duration < 240) & (tr.distance > 1)]
ts = pd.to_datetime(tr.date + " " + tr.time)
tr["hour"], tr["dow"] = ts.dt.hour, ts.dt.dayofweek
tr["daytype"] = np.where(tr.dow < 5, "weekday", "weekend")
print(f"{len(tr):,} readings, {tr.date.min()} to {tr.date.max()}, {tr.route_code.nunique()} routes, {tr.date.nunique()} days")


def centre(code):  # Plus Codes in the file are short; recover them next to central Bengaluru
    a, b = code.split("|")
    out = []
    for c in (a, b):
        full = olc.recoverNearest(c, 12.97, 77.59)
        area = olc.decode(full)
        out.append((area.latitudeCenter, area.longitudeCenter))
    return out


geo = {c: centre(c) for c in tr.route_code.unique()}
straight = {c: float(haversine_km(g[0][0], g[0][1], g[1][0], g[1][1])) for c, g in geo.items()}
label = dict(zip(rt.route_code, rt.label_short))

# typical reading per route, day type and hour, and the hourly profile
typ = tr.groupby(["route_code", "daytype", "hour"]).duration.median().rename("typical").reset_index()
prof = typ.merge(tr.groupby("route_code").distance.median().rename("road_km").reset_index(), on="route_code")
prof["straight_km"] = prof.route_code.map(straight)
prof["route"] = prof.route_code.map(label)
prof.round(2).drop(columns="route_code").to_csv(ROOT / "data" / "real_traffic_gmaps_profile.csv", index=False)

# 1) validation against the fitted travel-time model (car minutes for the straight-line distance, hour of day)
T = load_calibrated()
wk = prof[prof.daytype == "weekday"].copy()
wk["model"] = [float(T.car_min(s, h * 60 + 30)) for s, h in zip(wk.straight_km, wk.hour)]
wk["ratio"] = wk.model / wk.typical
rows = []
for r, g in wk.groupby("route"):
    rows.append(dict(route=r, road_km=g.road_km.iloc[0], straight_km=g.straight_km.iloc[0], gmaps_min_avg=g.typical.mean(),
                     model_min_avg=g.model.mean(), mean_ratio=g.ratio.mean(),
                     mape_pct=100 * np.mean(np.abs(g.model - g.typical) / g.typical),
                     hourly_corr=np.corrcoef(g.model, g.typical)[0, 1]))
val = pd.DataFrame(rows).round(3)
val.to_csv(ROOT / "results" / "gmaps_validation.csv", index=False)
peak = wk[wk.hour.isin([8, 9, 17, 18, 19])].ratio.mean()
night = wk[wk.hour.isin([1, 2, 3, 4])].ratio.mean()
allr = wk.groupby("hour").apply(lambda g: g.model.sum() / g.typical.sum(), include_groups=False)
print(val.to_string(index=False))
print("model / Google ratio by hour (routes pooled):", {int(h): round(float(v), 2) for h, v in allr.items()})

# 2) how big are real day-to-day slowdowns against the typical reading for the same route, day type and hour
m = tr.merge(typ, on=["route_code", "daytype", "hour"])
m["slow"] = m.duration / m.typical
q = {f"p{p}": round(float(np.percentile(m.slow, p)), 3) for p in (5, 25, 50, 75, 90, 95, 99, 99.9)}
peakq = {f"p{p}": round(float(np.percentile(m[m.hour.isin([8, 9, 17, 18, 19]) & (m.daytype == "weekday")].slow, p)), 3) for p in (50, 90, 95, 99)}
lognorm_sigma = float(np.std(np.log(m.slow)))
out = dict(readings=int(len(tr)), days=int(tr.date.nunique()), first=tr.date.min(), last=tr.date.max(), routes=int(tr.route_code.nunique()),
           slowdown_ratio_quantiles=q, slowdown_ratio_quantiles_weekday_peak_hours=peakq, log_sigma_of_ratio=round(lognorm_sigma, 3),
           share_of_readings_at_least_1_5x_typical=round(float((m.slow >= 1.5).mean()), 4),
           share_of_readings_at_least_2x_typical=round(float((m.slow >= 2.0).mean()), 4),
           model_over_google_peak=round(float(peak), 3), model_over_google_night=round(float(night), 3),
           model_mape_pct_mean=round(float(val.mape_pct.mean()), 1), hourly_corr_mean=round(float(val.hourly_corr.mean()), 3))
(ROOT / "results" / "gmaps_variability.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
