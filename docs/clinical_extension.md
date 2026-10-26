# Severity and specialty: a clinical extension

*Exploratory study, 30 held-out seeds (1000-1029), policies tuned on separate validation seeds (500-529). **Every clinical input below is an assumption**, not measured data. Reproduce: `python experiments/tune_clinical.py`, then `python experiments/run_clinical.py`.*

The paper's own literature review lists two open gaps: **triage- or survival-based objectives** and **specialty capability** (for example cardiac or stroke centres). This study adds both to the simulator, behind a flag that is off by default, so none of the paper's numbers change (24 re-run episodes matched the saved results exactly).

## 1. The model

| Element | Assumption |
|---|---|
| Severity | Stable 55 %, urgent 30 %, critical 15 % |
| Needs a specialty centre | Critical 85 %, urgent 25 %, stable 0 % (about 30 patients per episode); mix cardiac 40 %, stroke 30 %, trauma 30 % |
| ICU need | 5 % / 30 % / 80 % by severity (about 24 % overall, as in the paper) |
| Hospital capabilities | Assigned from size with a fixed seed: cardiac 5, stroke 8, trauma 7 of the 18 hospitals (`data/hospital_capabilities.csv`). **Not real data** |
| Wrong hospital | The patient is stabilised, then transferred: 15 min overhead plus the drive to the nearest capable hospital (a bed there is assumed) |
| Metric | **Time to definitive care**: bed at a capable hospital, or completed transfer |
| Severity weights | 1, 1.5, 3 on travel time; critical patients are assigned first |

## 2. Policies compared

| Policy | Sees beds and ambulances en route | Sees specialty | Severity-aware |
|---|:-:|:-:|:-:|
| Nearest | no | no | no |
| Nearest capable | no | **yes** | no |
| Capacity heuristic / Nash (from the paper) | yes | no | no |
| **Clinical heuristic** | yes | yes | yes |
| **Clinical Nash** | yes | yes | yes |

**The equilibrium guarantee survives.** In the clinical game, each ambulance's strategy set is limited to capable hospitals and its travel-time term is multiplied by its severity weight `w_i`. Both are player-specific *additive* terms (the congestion term is unchanged), so with `Phi(a) = sum_i w_i * tau_i,h(i) + (load terms)` a unilateral move still changes `Phi` by exactly the mover's change in cost. Best-response dynamics therefore still end at a pure Nash equilibrium. The test suite checks this on random instances (exact potential change, and no profitable deviation at the endpoint).

## 3. Results (mean minutes to definitive care; lower is better)

![Clinical study](images/clinical.png)

| Scenario | Policy | Critical patients | Critical p95 | Specialty patients | Specialty patients transferred | Rejections / episode |
|---|---|---:|---:|---:|---:|---:|
| Peak | Nearest | 46.38 | 90.3 | 48.11 | 51 % | 24.2 |
| Peak | Nearest capable | 33.85 | 70.7 | 33.39 | 0 % | 20.5 |
| Peak | Capacity heuristic (specialty-blind) | 41.36 | 76.5 | 44.62 | 65 % | 2.2 |
| Peak | Nash (specialty-blind) | 41.68 | 76.2 | 45.13 | 65 % | 1.7 |
| Peak | **Clinical heuristic** | **27.47** | **47.7** | **28.38** | 0 % | 0.6 |
| Peak | **Clinical Nash** | 28.41 | 49.0 | 28.96 | 0 % | 1.2 |
| Surge | Nearest | 48.94 | 98.4 | 50.90 | 47 % | 37.4 |
| Surge | Nearest capable | 34.81 | 84.0 | 34.29 | 0 % | 27.3 |
| Surge | Capacity heuristic (specialty-blind) | 41.52 | 78.5 | 45.74 | 64 % | 3.8 |
| Surge | Nash (specialty-blind) | 41.86 | 79.8 | 46.39 | 67 % | 3.5 |
| Surge | **Clinical heuristic** | **27.88** | **47.9** | **28.70** | 0 % | 1.7 |
| Surge | **Clinical Nash** | 28.19 | 51.5 | 28.95 | 0 % | 1.9 |

(Tuned on validation seeds: clinical heuristic weight 80; clinical Nash beta 40, gamma 1.)

## 4. What it shows

1. **Ignoring specialty is expensive, even for capacity-aware policies.** The paper's heuristic and Nash leave about **two-thirds of specialty patients needing a transfer**; critical patients wait about 41 min against about 28 min for the clinical versions: **-32 % to -34 %**, significant at p < 0.001 (Bonferroni over the 42 comparisons in `results/clinical_tests.csv`).
2. **Clinical policies beat the strongest simple baseline, "nearest capable hospital".** Critical patients: **-16 % at peak, -19 % in the surge** (Nash vs nearest capable, p < 0.001), with **about 94 % fewer rejections** (20.5 → 1.2 at peak; 27.3 → 1.9 in the surge). Against plain nearest-hospital dispatch the gain is **-39 % (peak) and -42 % (surge)**.
3. **Severity weighting is a small, real gain.** Taking it away costs the heuristic about **1.2 min (4 %)** for critical patients at peak (corrected p = 0.006) and about 1.1 min in the surge (corrected p = 0.056, not significant); for Nash the differences are suggestive only. Most of the benefit comes from specialty awareness and capacity awareness, not from the weights.
4. **Equilibrium again matches the simple heuristic.** Clinical Nash is within about 1 min of the clinical heuristic (0.9 min slower at peak, 0.3 min in the surge, neither significant). Same lesson as the paper: ship the simple, explainable version first.

## 5. Limits

All severity, specialty and capability inputs are assumed, and results depend on them (for example, how scarce each specialty is). A transferred patient is assumed to find a bed. Stabilisation care, pre-alerting the receiving hospital and clinical outcomes are not modelled; "time to definitive care" is a proxy, not survival. One city, 18 hospitals, 30 seeds. The point is the direction and size of the effect, which a real pilot with true capability data would need to confirm.
