"""Turn results/*.csv into LaTeX tables and copy figures into paper/.
Every number in the paper's tables comes from here, never typed by hand."""
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PAPER = ROOT / "paper"
if not PAPER.exists() and (ROOT.parent / "paper").exists():
    PAPER = ROOT.parent / "paper"  # layout used while the paper was drafted
PAPER.mkdir(exist_ok=True)

import pandas as pd

NAME = {"nearest": "Nearest", "nearest_beds": "Nearest + beds", "prototype_heuristic": "Prototype rule",
        "capacity_heuristic": "Capacity heuristic", "nash": "Nash equilibrium", "ppo": "PPO"}
SC = {"offpeak": "Off-peak (0.5 calls/min)", "peak": "Evening peak (1.2 calls/min)", "surge": "Surge (2.5 calls/min)"}
ORDER = list(NAME)


def tex(s):
    return str(s).replace("_", r"\_")


def write(name, text):
    text = re.sub(r"(?<=[\s\[(])-(?=\d)", "$-$", text)  # typographic minus
    (PAPER / name).write_text(text, encoding="utf-8")


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ---- traffic calibration
v = pd.read_csv(RES / "traffic_calibration_validation.csv")
v = v[v.model != "calibrated linear"]
rows = []
for r in v.itertuples():
    nm = "Synthetic model (initial)" if r.model.startswith("synthetic") else "Calibrated power-law"
    rows.append(f"{nm} & {r.split} & {r.n} & {r.MAPE_pct:.1f} & {r.median_APE_pct:.1f} & {r.bias_pct:+.1f} & {r.R2:.2f} \\\\")
write("tab_traffic.tex", "\\begin{tabular}{llrrrrr}\n\\toprule\nModel & Split & $n$ & MAPE\\,\\% & Med.\\,APE\\,\\% & Bias\\,\\% & $R^2$ \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ---- main results
s = pd.read_csv(RES / "final_summary.csv").set_index(["scenario", "policy"])
lines = []
for sc in SC:
    lines.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{SC[sc]}}}}} \\\\")
    for k in ORDER:
        if (sc, k) not in s.index:
            continue
        r = s.loc[(sc, k)]
        lines.append(f"{NAME[k]} & {r.ttb_mean:.2f} [{r.ttb_ci_lo:.2f}, {r.ttb_ci_hi:.2f}] & {r.ttb_p95:.1f} & "
                     f"{r.rejections:.1f} & {r.diversion_rate * 100:.1f} & {r.jain_load_balance:.2f} \\\\")
    lines.append("\\midrule")
lines = lines[:-1]
write("tab_main.tex", "\\begin{tabular}{lccccc}\n\\toprule\nPolicy & Mean time to bed [95\\% CI] & p95 & Rejected & Diverted\\,\\% & Jain \\\\\n\\midrule\n"
      + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

# ---- paired tests (time to bed)
t = pd.read_csv(RES / "final_tests.csv")
t = t[t.metric == "ttb_mean"]
lines = []
for sc in SC:
    lines.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{SC[sc].split(' (')[0]}}}}} \\\\")
    for r in t[t.scenario == sc].itertuples():
        lines.append(f"{NAME[r.a]} vs {NAME[r.b]} & {r.mean_diff:+.2f} [{r.diff_ci_lo:+.2f}, {r.diff_ci_hi:+.2f}] & "
                     f"{r.rel_change_pct:+.1f} & {fmt_p(r.p_holm)} \\\\")
    lines.append("\\midrule")
write("tab_tests.tex", "\\begin{tabular}{lccc}\n\\toprule\nComparison (A vs B) & $\\Delta$ min [95\\% CI] & $\\Delta$\\,\\% & $p$ (Holm) \\\\\n\\midrule\n"
      + "\n".join(lines[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")

# ---- ablations
a = pd.read_csv(RES / "ablations.csv")
pols = [p for p in ("nearest", "nearest_beds", "capacity_heuristic", "nash") if p in a.policy.unique()]
STUDY = {"freshness": "Bed-data refresh interval", "traffic": "Routing / traffic availability",
         "ambulance_speedup": "Ambulance speed-up over cars", "eta_noise": "Travel-time noise ($\\sigma$ scale)"}
VL = {"report_every_1min": "1 min", "report_every_5min": "5 min (default)", "report_every_15min": "15 min",
      "report_every_30min": "30 min", "traffic_aware": "Traffic-aware ETAs", "traffic_outage": "Free-flow ETAs (outage)",
      "boost_1.0": "1.00 (cars)", "boost_1.25": "1.25 (default)", "boost_1.5": "1.50",
      "sigma_x0.5": "0.5x", "sigma_x1.0": "1x (default)", "sigma_x2.0": "2x"}
lines = []
for sc in ("peak", "surge"):
    for st in STUDY:
        d = a[(a.study == st) & (a.scenario == sc)]
        if d.empty:
            continue
        lines.append(f"\\multicolumn{{{1 + len(pols)}}}{{l}}{{\\textit{{{sc.capitalize()}: {STUDY[st]}}}}} \\\\")
        for var in d.variant.unique():
            vals = []
            for p in pols:
                x = d[(d.variant == var) & (d.policy == p)].ttb_mean
                vals.append(f"{x.iloc[0]:.2f}" if len(x) else "--")
            lines.append(f"\\quad {VL.get(var, tex(var))} & " + " & ".join(vals) + " \\\\")
write("tab_ablation.tex", "\\begin{tabular}{l" + "c" * len(pols) + "}\n\\toprule\nSetting & " + " & ".join(NAME[p] for p in pols)
      + " \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

# ---- beta sweep + convergence
b = pd.read_csv(RES / "final_beta_sweep.csv")
lines = []
for sc in ("peak", "surge"):
    for r in b[b.scenario == sc].itertuples():
        lines.append(f"{sc.capitalize()} & {r.beta:g} & {r.ttb_mean:.2f} & {r.ttb_p95:.1f} & {r.rejections:.1f} & {r.jain_load_balance:.2f} \\\\")
write("tab_beta.tex", "\\begin{tabular}{lrrrrr}\n\\toprule\nScenario & $\\beta$ & Mean & p95 & Rejected & Jain \\\\\n\\midrule\n"
      + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
c = pd.read_csv(RES / "final_nash_convergence.csv")
lines = [f"{r.scenario.capitalize()} & {int(r.batches)} & {r.converged_pct:.1f} & {r.mean_rounds:.2f} & {int(r.max_rounds)} \\\\" for r in c.itertuples()]
write("tab_conv.tex", "\\begin{tabular}{lrrrr}\n\\toprule\nScenario & Batches & Converged\\,\\% & Mean rounds & Max rounds \\\\\n\\midrule\n"
      + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

# ---- figures
for f in ("fig_traffic_calibration.png", "fig_final_time_to_bed.png", "fig_final_rejections.png",
          "fig_final_beta_sweep.png", "fig_final_ppo_training.png"):
    if (RES / f).exists():
        shutil.copy(RES / f, PAPER / f)
# ---- staleness paired tests
st = pd.read_csv(RES / "staleness_tests.csv")
lines = []
for r in st.itertuples():
    lines.append(f"{r.scenario.capitalize()} & {r.refresh_min} & {tex(r.test)} & {r.mean_diff:+.2f} & {fmt_p(r.p_holm)} \\\\")
write("tab_stale.tex", "\\begin{tabular}{lrlrr}\n\\toprule\nScenario & Refresh (min) & Comparison & $\\Delta$ min & $p$ (Bonf.) \\\\\n\\midrule\n"
      + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

# ================= compact tables for the 6-page version =================
# traffic: held-out rows only
vt = pd.read_csv(RES / "traffic_calibration_validation.csv")
vt = vt[(vt.model != "calibrated linear") & (vt.split == "held-out")]
lines = []
for r in vt.itertuples():
    nm = "Initial synthetic model" if r.model.startswith("synthetic") else "Calibrated power law"
    lines.append(f"{nm} & {r.MAPE_pct:.1f} & {r.median_APE_pct:.1f} & {r.bias_pct:+.1f} & {r.R2:.2f} \\\\")
write("tab_traffic_small.tex", "\\begin{tabular}{lrrrr}\n\\toprule\nModel (held-out routes) & MAPE\\,\\% & Med.\\,APE\\,\\% & Bias\\,\\% & $R^2$ \\\\\n\\midrule\n"
      + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

# paired tests: peak and surge, key pairs
tt = pd.read_csv(RES / "final_tests.csv")
tt = tt[tt.metric == "ttb_mean"]
KEY = [("nash", "nearest"), ("capacity_heuristic", "nearest"), ("nash", "capacity_heuristic"),
       ("nash", "prototype_heuristic"), ("ppo", "capacity_heuristic")]
lines = []
for sc, lab in (("peak", "Evening peak"), ("surge", "Surge")):
    lines.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{lab}}}}} \\\\")
    for a, b in KEY:
        r = tt[(tt.scenario == sc) & (tt.a == a) & (tt.b == b)].iloc[0]
        lines.append(f"{NAME[a]} $-$ {NAME[b]} & {r.mean_diff:+.2f} [{r.diff_ci_lo:+.2f}, {r.diff_ci_hi:+.2f}] & "
                     f"{r.rel_change_pct:+.1f} & {fmt_p(r.p_holm)} \\\\")
write("tab_tests_small.tex", "\\begin{tabular}{lccc}\n\\toprule\nA $-$ B & Mean diff.\\ (min) [95\\% CI] & \\% & $p$ (Holm) \\\\\n\\midrule\n"
      + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

# beta sweep: compact
bb = pd.read_csv(RES / "final_beta_sweep.csv")
lines = []
for beta in sorted(bb.beta.unique()):
    p = bb[(bb.scenario == "peak") & (bb.beta == beta)].iloc[0]
    s_ = bb[(bb.scenario == "surge") & (bb.beta == beta)].iloc[0]
    lines.append(f"{beta:g} & {p.ttb_mean:.2f} & {p.rejections:.1f} & {s_.ttb_mean:.2f} & {s_.rejections:.1f} \\\\")
write("tab_beta_small.tex", "\\begin{tabular}{rrrrr}\n\\toprule\n & \\multicolumn{2}{c}{Peak} & \\multicolumn{2}{c}{Surge} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n$\\beta$ & Time & Rej. & Time & Rej. \\\\\n\\midrule\n"
      + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

# robustness: selected variants, three policies
ab = pd.read_csv(RES / "ablations.csv")
VARS = [("report_every_5min", "Default"), ("report_every_1min", "Bed refresh 1 min"),
        ("report_every_30min", "Bed refresh 30 min"), ("traffic_outage", "Free-flow ETAs"),
        ("boost_1.0", "Speed-up 1.0 (cars)"), ("boost_1.5", "Speed-up 1.5"), ("sigma_x2.0", "Noise $2\\times$")]
PL = ["nearest", "capacity_heuristic", "nash"]
lines = []
for var, lab in VARS:
    cells = []
    for sc in ("peak", "surge"):
        for p in PL:
            x = ab[(ab.variant == var) & (ab.scenario == sc) & (ab.policy == p)].ttb_mean
            cells.append(f"{x.iloc[0]:.2f}")
    lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
write("tab_robust_small.tex", "\\begin{tabular}{lcccccc}\n\\toprule\n & \\multicolumn{3}{c}{Peak} & \\multicolumn{3}{c}{Surge} \\\\\n"
      "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
      "Setting & Near. & Heur. & Nash & Near. & Heur. & Nash \\\\\n\\midrule\n"
      + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

print("wrote tables + figures to", PAPER)
