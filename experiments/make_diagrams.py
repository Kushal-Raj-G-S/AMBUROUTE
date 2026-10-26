"""Redraw the system architecture diagram at single-column size so the text stays readable."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper"
if not OUT.exists() and (ROOT.parent / "paper").exists():
    OUT = ROOT.parent / "paper"
OUT.mkdir(exist_ok=True)
FS = 6.8


def box(ax, x, y, w, h, text, fc="#e8f0fa", ec="#34567a"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=FS, linespacing=1.15)


def arrow(ax, x1, y1, x2, y2, label=None, dx=0.12):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#34567a", lw=0.9, shrinkA=0, shrinkB=0))
    if label:
        ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2, label, fontsize=5.8, color="#555", va="center")


fig, ax = plt.subplots(figsize=(3.5, 3.15))
ax.set_xlim(0, 10)
ax.set_ylim(0.6, 8.2)
ax.axis("off")

# row 1: console and external services
box(ax, 0.2, 6.6, 5.6, 1.4, "Dispatcher console\n(Mapbox GL JS)", fc="#eaf6ee")
box(ax, 6.7, 6.6, 3.1, 1.4, "Mapbox / OSRM\n(external)", fc="#eeeeee", ec="#777777")
# row 2: service
box(ax, 0.2, 4.0, 3.0, 1.6, "REST API\n/route /assign\n/hospitals")
box(ax, 3.5, 4.0, 3.0, 1.6, "Selection\nPPO | Nash |\nheuristic")
box(ax, 6.8, 4.0, 3.0, 1.6, "Routing adapter\nMapbox, OSRM,\nbuilt-in model")
# row 3: state and offline
box(ax, 0.2, 1.2, 3.0, 1.6, "Hospital state\nbeds, ICU,\nin flight", fc="#fff6e0")
box(ax, 3.5, 1.2, 3.0, 1.6, "Calibrated\ntraffic model", fc="#fff6e0")
box(ax, 6.8, 1.2, 3.0, 1.6, "Simulator,\nPPO training", fc="#fff6e0")

arrow(ax, 2.2, 6.6, 1.7, 5.6)
arrow(ax, 3.2, 4.8, 3.5, 4.8)
arrow(ax, 6.5, 4.8, 6.8, 4.8)
arrow(ax, 8.3, 5.6, 8.25, 6.6)
arrow(ax, 4.0, 4.0, 2.2, 2.8)
arrow(ax, 5.0, 4.0, 5.0, 2.8, "ETA")
arrow(ax, 7.6, 2.8, 6.0, 4.0, "policy", dx=0.15)
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig(OUT / "fig_architecture.png", dpi=300)
print("saved architecture diagram to", OUT)
