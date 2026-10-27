"""Download real hospital capacity from the US federal facility-level dataset and keep only an anonymous profile table.

Source: "COVID-19 Reported Patient Impact and Hospital Capacity by Facility" (HHS, healthdata.gov dataset anag-cw7u),
weekly, last updated for the week of 2024-04-21. U.S. federal government data. Six weeks spread over 2023-2024 are used so
the profile is not one season. Hospital names and locations are dropped: only size, occupancy and ICU figures are kept,
because the simulator needs the joint distribution of (beds, occupancy, ICU beds, ICU occupancy), not named hospitals.

    python experiments/fetch_hhs_capacity.py [2023-24 | 2022]
Writes data/real_capacity_profile.csv (2023-24) or data/real_capacity_profile_<label>.csv and prints a summary.
"""
import sys
from io import StringIO
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PERIODS = {  # label -> weeks (one per two to three months, so the profile is not one season)
    "2023-24": ["2023-01-22", "2023-04-23", "2023-07-23", "2023-10-22", "2024-01-21", "2024-04-21"],
    "2022": ["2022-01-23", "2022-03-27", "2022-05-29", "2022-07-24", "2022-09-25", "2022-11-27"],
}
LABEL = sys.argv[1] if len(sys.argv) > 1 else "2023-24"
WEEKS = PERIODS[LABEL]
OUT = "real_capacity_profile.csv" if LABEL == "2023-24" else f"real_capacity_profile_{LABEL}.csv"
BASE = "https://healthdata.gov/resource/anag-cw7u.csv"
COLS = ("collection_week,inpatient_beds_7_day_avg,inpatient_beds_used_7_day_avg,total_icu_beds_7_day_avg,icu_beds_used_7_day_avg,"
        "hospital_subtype")

frames = []
for wk in WEEKS:
    where = quote(f"collection_week='{wk}T00:00:00.000'")
    url = f"{BASE}?$select={quote(COLS)}&$where={where}&$limit=8000"
    d = pd.read_csv(StringIO(urlopen(url, timeout=180).read().decode()))
    print(wk, len(d), "rows", flush=True)
    frames.append(d)
d = pd.concat(frames, ignore_index=True)
for c in ("inpatient_beds_7_day_avg", "inpatient_beds_used_7_day_avg", "total_icu_beds_7_day_avg", "icu_beds_used_7_day_avg"):
    d[c] = pd.to_numeric(d[c], errors="coerce").where(lambda s: s >= 0)
d = d[(d.inpatient_beds_7_day_avg >= 25) & (d.total_icu_beds_7_day_avg >= 4)].dropna(
    subset=["inpatient_beds_used_7_day_avg", "icu_beds_used_7_day_avg"])
out = pd.DataFrame({
    "week": d.collection_week.str[:10],
    "beds": d.inpatient_beds_7_day_avg.round(1),
    "occ": (d.inpatient_beds_used_7_day_avg / d.inpatient_beds_7_day_avg).clip(0, 1).round(4),
    "icu_share": (d.total_icu_beds_7_day_avg / d.inpatient_beds_7_day_avg).clip(0.01, 0.6).round(4),
    "icu_occ": (d.icu_beds_used_7_day_avg / d.total_icu_beds_7_day_avg).clip(0, 1).round(4),
})
out.to_csv(ROOT / "data" / OUT, index=False)
q = lambda s: [round(float(np.percentile(s, p)), 3) for p in (5, 25, 50, 75, 95)]
print("kept", len(out), "hospital-weeks")
print("beds p5..p95", q(out.beds), "| occ", q(out.occ), "| icu_share", q(out.icu_share), "| icu_occ", q(out.icu_occ))
big = out[out.beds >= 500]
print("hospitals with >= 500 beds:", len(big), "| their occ", q(big.occ), "| icu_occ", q(big.icu_occ))
sys.exit(0)
