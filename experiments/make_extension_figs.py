"""Figures for the extension studies (docs/images/staleness.png, docs/images/scale.png)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)
LAB = {"nearest": "Nearest", "nearest_beds": "Nearest + beds", "capacity_heuristic": "Capacity heuristic", "nash": "Nash"}
COL = {"nearest": "#999999", "nearest_beds": "#C44E52", "capacity_heuristic": "#4C72B0", "nash": "#2f9e6b"}

s = pd.read_csv(ROOT / "results" / "extension_staleness.csv")
s = s[s.refresh_min <= 180]
xt = [5, 30, 60, 180]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for ax, sc in zip(axes, ("peak", "surge")):
    for p in LAB:
        d = s[(s.scenario == sc) & (s.policy == p)].sort_values("refresh_min")
        ax.plot(range(len(d)), d.ttb_mean, marker="o", label=LAB[p], color=COL[p])
    ax.set_xticks(range(len(xt)), ["5 min", "30 min", "60 min", "frozen"])
    ax.set_title("Evening peak" if sc == "peak" else "Surge")
    ax.set_xlabel("Bed-count refresh interval")
axes[0].set_ylabel("Mean time to bed (min)")
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "staleness.png", dpi=200)
plt.close(fig)

c = pd.read_csv(ROOT / "results" / "extension_scale.csv")
c = c[c.variant == "same load per bed"]
fig, ax = plt.subplots(figsize=(5.2, 3.4))
w = 0.2
for i, p in enumerate(LAB):
    vals = [c[(c.scenario == sc) & (c.policy == p)].ttb_mean.iloc[0] for sc in ("peak", "surge")]
    ax.bar([j + (i - 1.5) * w for j in range(2)], vals, w, label=LAB[p], color=COL[p])
ax.set_xticks([0, 1], ["Evening peak", "Surge"])
ax.set_ylabel("Mean time to bed (min)")
ax.set_title("110 OpenStreetMap hospitals, same load per bed", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "scale.png", dpi=200)
print("saved figures")
