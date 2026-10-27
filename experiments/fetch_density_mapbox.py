"""Where are the buildings? Count building footprints per map tile from Mapbox vector tiles (zoom 15), binned into
the same 10x10 grid used by the simulator. A coarse proxy for where people (and emergency calls) are; buildings are
not population. Only the aggregated counts are saved (no tile data), in line with Mapbox's terms.

    python experiments/fetch_density_mapbox.py        (needs MAPBOX_TOKEN in .env or the environment)
Writes data/bengaluru_building_density.csv
"""
import math
import os
import sys
from pathlib import Path

import httpx
import mapbox_vector_tile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from amburoute.geo import BBOX

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass
TOKEN = os.environ["MAPBOX_TOKEN"]
Z, N = 15, 10


def tile_xy(lat, lon):
    n = 2 ** Z
    return int((lon + 180) / 360 * n), int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)


def tile_centre(x, y):
    n = 2 ** Z
    lon = (x + 0.5) / n * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 0.5) / n))))
    return lat, lon


x0, y1 = tile_xy(BBOX["min_lat"], BBOX["min_lon"])
x1, y0 = tile_xy(BBOX["max_lat"], BBOX["max_lon"])
lats = np.linspace(BBOX["min_lat"], BBOX["max_lat"], N + 1)
lons = np.linspace(BBOX["min_lon"], BBOX["max_lon"], N + 1)
grid = np.zeros((N, N), dtype=int)
bad = 0
with httpx.Client(timeout=60) as c:
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            lat, lon = tile_centre(x, y)
            i, j = np.searchsorted(lats, lat) - 1, np.searchsorted(lons, lon) - 1
            if not (0 <= i < N and 0 <= j < N):
                continue
            r = c.get(f"https://api.mapbox.com/v4/mapbox.mapbox-streets-v8/{Z}/{x}/{y}.mvt", params={"access_token": TOKEN})
            if r.status_code != 200:
                bad += 1
                continue
            layer = mapbox_vector_tile.decode(r.content).get("building", {})
            grid[i, j] += len(layer.get("features", []))
    print("tiles", (x1 - x0 + 1) * (y1 - y0 + 1), "failed", bad)
rows = [dict(row=i, col=j, lat0=lats[i], lat1=lats[i + 1], lon0=lons[j], lon1=lons[j + 1], buildings=int(grid[i, j]))
        for i in range(N) for j in range(N)]
df = pd.DataFrame(rows)
df.to_csv(ROOT / "data" / "bengaluru_building_density.csv", index=False)
print(df.buildings.describe())
print(grid)
