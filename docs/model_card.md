# Model card: AmbuRoute dispatch policies

## Policies

| Policy | What it does | Learned? |
|---|---|---|
| Nearest | closest hospital by estimated travel time | no |
| Nearest with beds | closest hospital with a published free bed | no |
| Capacity heuristic | travel time plus a penalty that grows as free beds minus ambulances already on the way shrinks | no (one tuned weight) |
| Equilibrium (Nash) | assigns simultaneous dispatches so that no ambulance gains by switching alone; a congestion game with a potential function, so best-response always stops | no (two tuned parameters) |
| PPO | reinforcement-learning policy over 12 candidate hospitals, warm-started by imitating the heuristic | yes |
| Graph network cost | a small graph network over hospitals that predicts the minutes lost at the door, replacing the hand-set penalty | yes |
| Clinical heuristic / clinical equilibrium | the above restricted to hospitals that can treat the patient's specialty need, with severity-weighted travel time | no |
| Online ETA correction | per-hospital travel-time correction learned from trip progress reports | yes (online) |

The service defaults to the capacity heuristic (or the clinical heuristic when severity or specialty is given), because it is simple, explainable and as good as the alternatives in the experiments.

## Data

- **Hospitals:** 18 Bengaluru hospitals (name, location, bed totals; the table's original source is still uncited) and 110 OpenStreetMap emergency hospitals for a scale study (beds assumed).
- **Travel times:** 3,861 Mapbox typical-traffic samples, a power-law model by hour; 11.3 % error on 105 held-out routes (25.8 % before calibration).
- **Calls, bed occupancy, ICU capacity:** simulated.
- **Capabilities:** public sources, lower bound (see `safety.md`).
- **Real-data checks:** NYC EMS dispatch log (hourly shape, severity mix, arrival times) and building density from Mapbox tiles for call locations.

## Evaluation

Disjoint seed sets: training 100000 and up, validation 500-529 (all tuning and model selection), test 1000-1049 (scored once). Paired Wilcoxon tests with multiple-comparison correction; bootstrap intervals. Reproduce with the scripts listed in the README.

Headline (mean minutes from call to bed, 50 test seeds): evening peak 25.78 (nearest), 22.73 (heuristic), 22.64 (equilibrium), 22.94 (PPO); surge 27.75, 22.97, 22.91, 23.50. Counting ambulances already on the way is worth 5 to 10 % on its own. PPO and the graph-network cost match the heuristic but do not beat it. Details: [`additional_studies.md`](additional_studies.md).

## Limits and risks

See [`safety.md`](safety.md). In short: simulated beds, one city, proxy outcomes, assumptions listed per study, and no field validation. PPO training is not bit-for-bit repeatable; shipped weights are in `results/models/`.
