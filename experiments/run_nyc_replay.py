"""Real arrival times instead of a Poisson process. Replays the 17:00-19:00 call times of ten NYC weekday evenings
(2023-03-06 to 2023-03-17, NYC EMS Incident Dispatch Data), thinned at random so the mean rate matches the paper's
peak scenario (1.2 calls per minute). Thinning pulls a bursty process toward Poisson, so the burstiness measured here
is a lower bound. The hospitals, traffic and policies are the simulator's own (Bengaluru, tuned policies, not re-tuned).

    python experiments/run_nyc_replay.py
Outputs: data/real_nyc_ems_evening_calls.csv (call times only), results/nyc_replay.csv, results/nyc_replay_tests.csv
"""
import json
import zlib
import sys
from io import StringIO
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from amburoute.geo import load_hospitals
from amburoute.nash import NashPolicy
from amburoute.policies import CapacityHeuristic, Nearest
from amburoute.sim import SCENARIOS, SimConfig, run_episode
from amburoute.traffic import load_calibrated

RES = ROOT / "results"
f = ROOT / "data" / "real_nyc_ems_evening_calls.csv"
if not f.exists():
    where = "incident_datetime between '2023-03-06T00:00:00' and '2023-03-17T23:59:59' AND date_extract_hh(incident_datetime) in (17,18)"
    url = ("https://data.cityofnewyork.us/resource/76xm-jjuj.csv?$select=incident_datetime&$limit=100000&$where=" + quote(where, safe="(),'"))
    pd.read_csv(StringIO(urlopen(url, timeout=180).read().decode())).to_csv(f, index=False)
calls = pd.read_csv(f, parse_dates=["incident_datetime"])
calls = calls[calls.incident_datetime.dt.day_of_week < 5]
calls["day"] = calls.incident_datetime.dt.date
calls["minute"] = (calls.incident_datetime.dt.hour - 17) * 60 + calls.incident_datetime.dt.minute + calls.incident_datetime.dt.second / 60
days = sorted(calls.day.unique())
print("weekdays", len(days), "calls per evening (2 h):", calls.groupby("day").size().to_numpy())

# burstiness of the real process: variance / mean of 2-minute counts (Poisson = 1)
disp = [np.var(c) / np.mean(c) for c in (np.histogram(calls[calls.day == d].minute, bins=np.arange(0, 121, 2))[0] for d in days)]
print("dispersion index of 2-min counts per evening (Poisson = 1): mean %.2f" % np.mean(disp))

best = json.loads((RES / "best_baseline_params.json").read_text())
H, T = load_hospitals(), load_calibrated()
pols = {"nearest": Nearest(), "heuristic": CapacityHeuristic(best["capacity_heuristic"]["w"]),
        "nash": NashPolicy(best["nash"]["beta"], best["nash"]["power"])}
sc = SCENARIOS["peak"]
rows = []
for d in days:
    m = calls[calls.day == d].minute.to_numpy()
    q = 1.2 * 120 / len(m)
    for k in range(5):
        thin = np.random.default_rng(zlib.crc32(f"{d}-{k}".encode())).random(len(m)) < q
        arr = m[thin]
        for name, p in pols.items():
            real = run_episode(H, sc, 1000 + k, p, SimConfig(), T, arrivals=arr)
            pois = run_episode(H, sc, 1000 + k, p, SimConfig(), T)
            rows.append(dict(day=str(d), rep=k, policy=name, n_calls=len(arr), real_ttb=real["ttb_mean"], poisson_ttb=pois["ttb_mean"],
                             real_rej=real["rejections"], poisson_rej=pois["rejections"]))
raw = pd.DataFrame(rows)
summ = raw.groupby("policy")[["n_calls", "real_ttb", "poisson_ttb", "real_rej", "poisson_rej"]].mean().round(3)
summ.to_csv(RES / "nyc_replay.csv")
print(summ.to_string())
t = []
g = lambda pol, col: raw[raw.policy == pol].sort_values(["day", "rep"])[col].to_numpy()
for a, b in (("heuristic", "nearest"), ("nash", "nearest"), ("nash", "heuristic")):
    d = g(a, "real_ttb") - g(b, "real_ttb")
    t.append(dict(a=a, b=b, rel_pct=100 * d.mean() / g(b, "real_ttb").mean(), p_raw=wilcoxon(d).pvalue))
t = pd.DataFrame(t)
t["p_bonf"] = np.minimum(1, t.p_raw * len(t))
t.round(4).to_csv(RES / "nyc_replay_tests.csv", index=False)
print(t.round(4).to_string(index=False))
