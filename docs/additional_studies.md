# Additional studies: modules, objectives, online adaptation, learned costs, real-data checks

*All runs use the paper's simulator, travel-time model and tuned parameters. Test seeds 1000-1049 (30 for the clinical and multi-objective studies); weights and scales are chosen on validation seeds 500-529 only. Tests are paired Wilcoxon with Bonferroni correction. Every table is produced by the script named in its heading; raw numbers are in `results/`.*

## 1. What does each module contribute? (`experiments/run_module_ablation.py`)

Full system = calibrated travel times + in-flight load + congestion game. Each row removes one module; nothing is re-tuned.

| Variant | Peak: min to bed | vs full | Peak rejections | Surge: min to bed | vs full | Surge rejections |
|---|---:|---:|---:|---:|---:|---:|
| Full system | 22.64 | | 2.8 | 22.91 | | 5.6 |
| No congestion game (heuristic instead) | 22.73 | +0.4 % (n.s.) | 3.5 | 22.97 | +0.3 % (n.s.) | 6.2 |
| No traffic calibration | 22.84 | +0.9 % (n.s.) | 3.8 | 23.20 | +1.3 % (p = 0.004) | 8.3 |
| **No in-flight load** | **23.69** | **+4.6 % (p < 0.001)** | 11.3 | **25.17** | **+9.9 % (p < 0.001)** | 24.2 |
| None of the three (nearest hospital) | 25.78 | +13.9 % | 30.1 | 27.75 | +21.1 % | 46.8 |

Counting ambulances already on the road is the module that matters. Calibration helps a little, mostly under surge. The game adds almost nothing in minutes and a little in rejections.

**PPO warm start** (`experiments/run_ppo_warmstart.py`, 1M steps, three seeds per variant, seed chosen on validation):

| PPO variant | Peak | Surge | Peak rejections | Surge rejections |
|---|---:|---:|---:|---:|
| With behaviour-cloning warm start (chosen seed) | 22.94 | 23.50 | 5.5 | 11.1 |
| Without warm start (chosen seed) | 24.36 | 25.76 | 18.4 | 29.7 |
| Tuned heuristic | 22.73 | 22.97 | 3.5 | 6.2 |

Without imitation the learner is 6 % (peak) and 10 % (surge) slower than with it, and its three seeds spread widely (peak 24.4 to 25.8). Warm-started PPO is within 0.9 % of the heuristic at peak (not significant after correction) and 2.3 % slower in the surge.

## 2. Several objectives at once (`experiments/run_multiobjective.py`)

Objectives: mean time to definitive care, expected survival, wait after reaching a hospital, load balance across hospitals (Jain). **The survival curve is assumed** (`exp(-hazard * t)`, hazards 0.05 / 0.25 / 0.90 per hour for stable / urgent / critical). 60 weightings (door-wait weight, severity scaling, crowding penalty) were scored on validation seeds and the non-dominated set found.

Result on 30 test seeds (peak): the goals mostly point the same way. The balanced operating point picked on validation is exactly the clinical heuristic of `clinical_extension.md`; the "fastest" point saves 0.15 min of mean time by accepting 4x more waiting at the door (0.35 vs 0.08 min) and 4x more rejections. Expected survival moves only between 0.913 and 0.915 because the assumed curve is gentle. **There is no sharp trade-off here**, and the survival numbers should not be read as clinical.

![Pareto](images/multiobjective.png)

## 3. Online adaptation to local traffic jams (`experiments/run_online.py`)

Drift: from minute 30, trips to four randomly chosen hospitals take twice as long as the fitted model says. `OnlineETA` learns a per-hospital correction from en-route progress reports (assumed to be available 35 % into each trip) and applies only relative corrections.

| Mean minutes to bed (heuristic base) | Fixed model | Online | Oracle (knows the jams) |
|---|---:|---:|---:|
| Local jams, peak | 26.21 | **24.58 (-6.2 %)** | 23.42 (-10.6 %) |
| Local jams, surge | 25.44 | **24.79 (-2.6 %)** | 23.46 (-7.8 %) |
| No drift, peak | 22.73 | 22.86 (+0.6 %, n.s.) | 22.73 |
| Citywide slowdown x1.5, peak | 30.88 | 31.39 (+1.6 %, p = 0.002) | 30.97 |

The learner recovers about 60 % of the possible gain at peak and about a third in the surge, and costs up to 2 % when the whole city slows evenly (its corrections are then noise). **A first version that learned only from finished trips gave no gain**: observations arrive too late and too rarely, because few ambulances go to any one hospital. The same holds for the equilibrium base policy.

## 4. Learned congestion cost with a graph network (`experiments/run_gnn.py`)

A two-layer graph network over a hospital nearest-neighbour graph predicts the minutes a patient will lose at the door at each hospital, trained on 77,407 exploratory dispatches (training seeds only), replacing the hand-set `w / (1 + beds)` term. The cost scale was chosen on validation seeds (6.0 for the graph network, 3.0 without edges; 6.0 is the top of the search grid).

| Mean minutes to bed | Off-peak | Peak | Surge |
|---|---:|---:|---:|
| Nearest | 20.79 | 25.78 | 27.75 |
| Tuned heuristic | 19.66 | 22.73 | 22.97 |
| Tuned Nash | 19.67 | 22.64 | 22.91 |
| **Graph network** | 19.67 | 22.59 | 22.90 |
| Same network without edges | 19.68 | 22.59 | 22.88 |

The graph network matches the tuned heuristic and Nash assignment (all differences are small and none is significant after correction), **without any hand-set congestion weight**. The edges add nothing measurable on this 18-hospital map. It is about 20x slower per decision (0.24 ms vs 0.01 ms), which is immaterial. The honest reading: a learned cost is a viable replacement for the hand-crafted one, but not an improvement.

## 5. Checks against real data (`experiments/real_data_nyc_ems.py`, `run_clinical_realmix.py`)

No call-level EMS or hospital-bed data for Bengaluru is public (a CAG audit report covers Karnataka's 108 service only in aggregate). We used the public NYC "EMS Incident Dispatch Data" (NYC Open Data, 2023; aggregates in `data/real_nyc_ems_*`) to check two assumptions:

- **Demand is flat in the afternoon and evening** (17-19h vs 12-14h ratio 1.01). The paper's "evening peak" (1.2 vs 0.5 calls per minute) is therefore a **stress scenario**, not a measured demand spike; traffic and bed occupancy, not call volume, make the evening hard.
- **Real dispatch priority mix** is 22 % highest priority (levels 1-2), 35 % (3-4), 43 % (5+), against our assumed 15 / 30 / 55 split (priority codes are not clinical triage, so this is a proxy). Re-running the clinical study with this mix and unchanged policies: clinical Nash reaches critical patients 23 % faster than nearest-capable at peak and 21 % faster in the surge, and 28 % / 26 % faster than the specialty-blind Nash (all p < 0.001; Nash is again not distinguishable from the clinical heuristic). The conclusion survives a more severe mix.

## Limits

The traffic-drift, survival and GPS-progress assumptions are ours; beds, ICU capacity, calls and hospital capabilities remain simulated. NYC is a different city and system. Several comparisons are not significant and are reported as such.
