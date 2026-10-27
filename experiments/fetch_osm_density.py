"""Building counts per grid cell over the simulated area, from OpenStreetMap (ODbL), as a proxy for where people
(and so emergency calls) are. Buildings are not population: this is a coarse proxy and is described as one.

    python experiments/fetch_osm_density.py
Writes data/bengaluru_building_density.csv. Map data (c) OpenStreetMap contributors, ODbL.
"""
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from amburoute.geo import BBOX

N = 10
lats = np.linspace(BBOX["min_lat"], BBOX["max_lat"], N + 1)
lons = np.linspace(BBOX["min_lon"], BBOX["max_lon"], N + 1)
rows = []
for i in range(N):
    for j in range(N):
        q = f'[out:json][timeout:120];way["building"]({lats[i]},{lons[j]},{lats[i+1]},{lons[j+1]});out count;'
        for attempt in range(4):
            try:
                r = httpx.post("https://overpass-api.de/api/interpreter", data={"data": q}, timeout=150,
                               headers={"User-Agent": "AmbuRoute-research/0.1 (academic prototype)"})
                r.raise_for_status()
                n = int(r.json()["elements"][0]["tags"]["ways"])
                break
            except Exception as e:  # rate limit or timeout: back off and retry
                print("retry", i, j, type(e).__name__, flush=True)
                time.sleep(15 * (attempt + 1))
        else:
            raise SystemExit(f"cell {i},{j} failed")
        rows.append(dict(row=i, col=j, lat0=lats[i], lat1=lats[i + 1], lon0=lons[j], lon1=lons[j + 1], buildings=n))
        time.sleep(2)
    print("row", i, "done", flush=True)
df = pd.DataFrame(rows)
df.to_csv(ROOT / "data" / "bengaluru_building_density.csv", index=False)
print(df.buildings.describe())
