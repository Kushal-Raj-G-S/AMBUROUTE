"""Fetch Bengaluru hospitals from OpenStreetMap (ODbL licence) via the Overpass API.

    python experiments/fetch_osm_hospitals.py

Writes data/osm_hospitals_raw.json (full response) and data/osm_hospitals_summary.txt.
Map data (c) OpenStreetMap contributors, ODbL: https://www.openstreetmap.org/copyright
"""
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
BBOX = "12.80,77.45,13.15,77.80"  # south, west, north, east
QUERY = f"""
[out:json][timeout:60];
(
  node["amenity"="hospital"]({BBOX});
  way["amenity"="hospital"]({BBOX});
  relation["amenity"="hospital"]({BBOX});
);
out center tags;
"""

r = httpx.post("https://overpass-api.de/api/interpreter", data={"data": QUERY}, timeout=90,
               headers={"User-Agent": "AmbuRoute-research/0.1 (academic prototype)"})
r.raise_for_status()
data = r.json()
(ROOT / "data").mkdir(exist_ok=True)
(ROOT / "data" / "osm_hospitals_raw.json").write_text(json.dumps(data), encoding="utf-8")

els = data["elements"]
named = [e for e in els if e.get("tags", {}).get("name")]
with_beds = [e for e in els if "beds" in e.get("tags", {}) or "capacity:beds" in e.get("tags", {})]
emerg = [e for e in els if e.get("tags", {}).get("emergency") == "yes"]
lines = [f"elements: {len(els)}", f"with a name: {len(named)}", f"with a beds tag: {len(with_beds)}",
         f"tagged emergency=yes: {len(emerg)}"]
(ROOT / "data" / "osm_hospitals_summary.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
for e in with_beds[:15]:
    t = e["tags"]
    print(" ", t.get("name"), "| beds:", t.get("beds") or t.get("capacity:beds"))
sys.exit(0)
