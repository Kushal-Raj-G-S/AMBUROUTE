"""Fit the simulator's travel-time model to Mapbox typical-traffic data and
validate it on held-out origin-destination pairs.

Pairs (not rows) are split 70/30 so held-out pairs are never seen in fitting.
Two model forms are compared by 5-fold cross-validation on TRAIN pairs only;
the winner is then scored once on the held-out pairs.

    python experiments/calibrate_traffic.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from amburoute.traffic import CalibratedTraffic, TrafficModel

RES = ROOT / "results"
df = pd.concat([pd.read_csv(p) for p in sorted(RES.glob("mapbox_calibration*.csv"))])
df = df.drop_duplicates(["pair_id", "depart"])
df = df[df.haversine_km > 0.5].copy()
df["hour"] = df.depart.str[-5:-3].astype(int) + df.depart.str[-2:].astype(int) / 60
df["car_min"] = df.duration_typical_s / 60

rng = np.random.default_rng(0)
pairs = np.array(sorted(df.pair_id.unique()))
test_pairs = set(rng.choice(pairs, size=int(0.3 * len(pairs)), replace=False))
df["split"] = np.where(df.pair_id.isin(test_pairs), "test", "train")
tr, te = df[df.split == "train"].copy(), df[df.split == "test"].copy()
hours = np.sort(df.hour.unique())


def fit(d, form):
    h = d.hour.to_numpy()
    km = d.haversine_km.to_numpy()
    y = d.car_min.to_numpy()
    if form == "linear":
        X = np.zeros((len(d), len(hours) + 1))
        for j, hr in enumerate(hours):
            X[:, j] = np.where(h == hr, km, 0.0)
        X[:, -1] = 1.0
        w, *_ = np.linalg.lstsq(X, y, rcond=None)
        return dict(form="linear", hours=hours.tolist(), coef=w[:-1].tolist(),
                    intercept_min=float(w[-1]), sigma=0.0)
    X = np.zeros((len(d), len(hours) + 1))
    for j, hr in enumerate(hours):
        X[:, j] = (h == hr).astype(float)
    X[:, -1] = np.log(km)
    w, *_ = np.linalg.lstsq(X, np.log(y), rcond=None)
    return dict(form="power", hours=hours.tolist(), coef=w[:-1].tolist(),
                power=float(w[-1]), sigma=0.0)


def predict(params, d):
    m = CalibratedTraffic(params, ambulance_boost=1.0)
    return np.array([float(m.eta_min(r.haversine_km, r.hour * 60)) for r in d.itertuples()])


def mape(p, y):
    return float(np.mean(np.abs(p - y) / y))


# ---- model selection: 5-fold CV over TRAIN pairs
tp = np.array(sorted(tr.pair_id.unique()))
folds = np.array_split(rng.permutation(tp), 5)
cv = {}
for form in ("linear", "power"):
    errs = []
    for f in folds:
        a, b = tr[~tr.pair_id.isin(f)], tr[tr.pair_id.isin(f)]
        errs.append(mape(predict(fit(a, form), b), b.car_min.to_numpy()))
    cv[form] = float(np.mean(errs))
best = min(cv, key=cv.get)
params = fit(tr, best)
params["sigma"] = float(np.std(np.log(te.car_min.to_numpy() / predict(params, te))))
(RES / "traffic_calibration.json").write_text(json.dumps(params, indent=1))

# ---- report
rows = []
old = TrafficModel(ambulance_boost=1.0)  # synthetic model, car-equivalent
cands = [("synthetic (before)", None)] + [(f"calibrated {f}", fit(tr, f)) for f in ("linear", "power")]
for name, prm in cands:
    for split, d in (("train", tr), ("held-out", te)):
        p = (np.array([float(old.eta_min(r.haversine_km, r.hour * 60)) for r in d.itertuples()])
             if prm is None else predict(prm, d))
        y = d.car_min.to_numpy()
        rows.append(dict(model=name, split=split, n=len(d), MAPE_pct=100 * mape(p, y),
                         median_APE_pct=100 * float(np.median(np.abs(p - y) / y)),
                         bias_pct=100 * float(np.mean((p - y) / y)),
                         R2=1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()))
tab = pd.DataFrame(rows).round(3)
tab.to_csv(RES / "traffic_calibration_validation.csv", index=False)
print(tab.to_string(index=False))
print("CV MAPE (train pairs):", {k: round(100 * v, 2) for k, v in cv.items()}, "-> chosen:", best)
print("rows", len(df), "pairs", len(pairs), "train/test pairs", len(tp), len(test_pairs),
      "held-out log-sigma", round(params["sigma"], 3), "circuity", round((df.road_km / df.haversine_km).median(), 3))

fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ps_old = np.array([float(old.eta_min(r.haversine_km, r.hour * 60)) for r in te.itertuples()])
ax[0].scatter(te.car_min, ps_old, s=6, alpha=.35, label="synthetic (before)")
ax[0].scatter(te.car_min, predict(params, te), s=6, alpha=.35, label=f"calibrated {best}")
m = float(te.car_min.max())
ax[0].plot([0, m], [0, m], "k--", lw=1)
ax[0].set_xlabel("Mapbox typical travel time (min)"); ax[0].set_ylabel("Model estimate (min)")
ax[0].set_title("Held-out pairs"); ax[0].legend(fontsize=8)
mm = CalibratedTraffic(params, ambulance_boost=1.0)
hh = np.linspace(0, 23.5, 48)
ax[1].plot(hh, [float(mm.car_min(5.0, x * 60)) for x in hh], label="fitted, 5 km trip")
obs = te[(te.haversine_km > 4.5) & (te.haversine_km < 5.5)].groupby("hour").car_min.mean()
ax[1].scatter(obs.index, obs.values, c="C1", s=20, label="Mapbox (4.5-5.5 km, held-out)")
ax[1].set_xlabel("Departure hour"); ax[1].set_ylabel("Travel time (min)"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(RES / "fig_traffic_calibration.png", dpi=200)
