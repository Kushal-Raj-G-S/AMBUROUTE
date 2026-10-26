"""Collect real typical-traffic travel times from the Mapbox Directions API
(driving-traffic profile, `depart_at` + `duration_typical`) for calibrating and
validating the simulator's traffic model.

Pairs: 180 random city points -> random hospital, and 180 hospital -> hospital.
Times (local, a future Wednesday): 11 departure times across the day.
~3,960 requests, well inside the free tier. Resumable: re-run to continue.

    python experiments/collect_mapbox_traffic.py
"""
import csv
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx
import numpy as np

import amburoute  # noqa: F401  (loads .env)
from amburoute.geo import BBOX, haversine_km, load_hospitals

RES = ROOT / "results"
DATE = "2026-10-28"  # a Wednesday
TIMES = ["00:00", "03:00", "06:00", "08:30", "10:30", "13:00",
         "15:30", "17:30", "18:30", "20:30", "22:30"]
N_EACH = 180
FIELDS = ["pair_id", "kind", "olat", "olon", "hlat", "hlon", "depart", "haversine_km",
          "road_km", "duration_typical_s", "duration_s"]


def main():
    shard, nshards = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
    OUT = RES / ("mapbox_calibration.csv" if nshards == 1 else f"mapbox_calibration_part{shard}.csv")
    token = os.environ["MAPBOX_TOKEN"]
    h = load_hospitals()
    rng = np.random.default_rng(42)
    pairs = []
    for i in range(N_EACH):
        hi = int(rng.integers(len(h)))
        pairs.append((f"r{i}", "random_to_hospital",
                      rng.uniform(BBOX["min_lat"], BBOX["max_lat"]),
                      rng.uniform(BBOX["min_lon"], BBOX["max_lon"]), hi))
    for i in range(N_EACH):
        a, b = rng.choice(len(h), 2, replace=False)
        pairs.append((f"h{i}", "hospital_to_hospital", float(h.lat[a]), float(h.lon[a]), int(b)))

    pairs = pairs[shard::nshards]
    done = set()
    for part in RES.glob("mapbox_calibration*.csv"):  # resume across all shards
        with open(part) as fh:
            done |= {(r["pair_id"], r["depart"]) for r in csv.DictReader(fh)}
    new_file = not OUT.exists()
    f = open(OUT, "a", newline="")
    w = csv.DictWriter(f, FIELDS)
    if new_file:
        w.writeheader()

    client = httpx.Client(timeout=15)
    n = 0
    for pid, kind, olat, olon, hi in pairs:
        hlat, hlon = float(h.lat[hi]), float(h.lon[hi])
        hav = float(haversine_km(olat, olon, hlat, hlon))
        for t in TIMES:
            depart = f"{DATE}T{t}"
            if (pid, depart) in done:
                continue
            url = ("https://api.mapbox.com/directions/v5/mapbox/driving-traffic/"
                   f"{olon},{olat};{hlon},{hlat}")
            for attempt in range(4):
                r = client.get(url, params=dict(access_token=token, overview="false",
                                                depart_at=depart))
                if r.status_code == 429:
                    time.sleep(10 * (attempt + 1))
                    continue
                break
            if r.status_code != 200 or not r.json().get("routes"):
                continue  # e.g. unroutable point; skipped, not imputed
            rt = r.json()["routes"][0]
            if "duration_typical" not in rt:
                continue
            w.writerow(dict(pair_id=pid, kind=kind, olat=olat, olon=olon, hlat=hlat,
                            hlon=hlon, depart=depart, haversine_km=hav,
                            road_km=rt["distance"] / 1000,
                            duration_typical_s=rt["duration_typical"],
                            duration_s=rt["duration"]))
            n += 1
            if n % 100 == 0:
                f.flush()
                print(n, "rows", flush=True)
            time.sleep(0.25)  # ~240 req/min, under the 300/min limit
    f.close()
    print("done", n, "new rows")


if __name__ == "__main__":
    main()
