# The journey: from the inherited draft to a defensible project

Written 2026-10-28 from the audit in `PLAN_OF_ACTION.md` (project root), `research/PROJECT_LOG.md` and the current repository. Three stages: **A** the original draft and prototype that were inherited, **B** the first paper we rebuilt (2026-10-25 to 26), **C** the project today.

## 1. Side by side

| | **A. Original draft (inherited)** | **B. First rebuilt paper** | **C. Today** |
|---|---|---|---|
| **Code behind the claims** | None. The public repository had one commit from 2025-09-05: a 178-line Flask prototype with a fixed 60/40 scoring formula, static bed counts, a test aimed at Manhattan that would fail, and committed credentials. No PPO, no equilibrium solver, no FastAPI, no experiments | Simulator, six policies, equilibrium solver, PPO environment, FastAPI service, 7 tests | 29 tests, lint clean, CI (lint, tests, dependency audit, secrets scan, Docker build), non-root container smoke-tested |
| **Headline claim** | "DRL is 367 % better": 3 min against 14 min, which is arithmetically a 79 % reduction, and "3.67x faster" is also wrong (14/3 is 4.67x) | Capacity awareness cuts time to a bed 12 % at the evening peak and 17 % in a surge | The same, now stated as conditional: 4-6 % where hospitals have slack, 10-12 % where they are as full as 2026 English trusts |
| **Reinforcement learning** | "DRL improves" with about 15 runs, a literal `nan` in the table, DRL slower and equal success rate | PPO trained and tested on held-out seeds; **it matches the tuned heuristic but does not beat it**, and the paper says so | Reported again with warm-start ablation (without imitation it is 6-10 % worse) and a graph-network alternative that also only matches |
| **Equilibrium (the abstract's headline)** | Never evaluated; "high convergence rates" with no data | Proved convergent (potential game) and converged in all 4,239 test batches | Proof extended to severity-weighted, capability-restricted games and unit-tested |
| **Traffic** | Described, not measured | Fitted to 3,861 real Mapbox samples, 11.3 % error on 105 unseen routes | Also checked against 113,886 independent real Google Maps readings (shape correlation 0.94, rush-hour level 0.98); too slow at night, documented |
| **Hospital data** | Static beds in two CSVs; ICU a boolean | 18 hospitals, simulated beds and occupancy, stated as such | Occupancy and ICU shares **measured** from 14,202 + 14,955 US hospital-weeks (2022, 2023-24) and 124 English trusts (2026); specialty capabilities from public sources (lower bound) |
| **Comparison with the real world** | A table of FDNY operational figures against our simulated output, uncited | Removed | Real data used only as inputs and checks, each labelled (NYC call timing and severity mix, building density) |
| **Statistics** | "Statistically significant", no test | Disjoint training, validation and test seeds; Wilcoxon with corrections; bootstrap intervals | Same discipline across every extension (corrected p-values in every table) |
| **Scores such as robustness 1.00 / 0.90 / 0.95** | No metric defined, no runs | Removed | Replaced by measured ablations: counting ambulances already on the road is worth 5-10 %; the other modules add little |
| **Claims of pilots and cloud rollout** | "15-20 % in cloud pilots", GCP deployment: none existed | Removed | A shadow-mode pilot is *designed* (`docs/pilot_design.md`), not claimed |
| **Citations** | Several pointed at the wrong paper; unused entries; duplicate | 16 original references kept; 8 added from IEEE, Springer or MDPI | Same; evidence for market claims logged with checks at source |
| **Reproducibility** | Not possible | One command per result; seeds disjoint; models shipped | Manifested results, generated docs tables, handoff document, full-history backup |
| **Honesty about limits** | Absent | Provenance table of real versus simulated | Limits stated per study; corrected our own claims twice when real data disagreed |

## 2. What changed in the conclusions (the most important point)
The original draft said a learned policy was several times better. The honest result is the opposite and smaller: **the gain comes from one simple idea, counting ambulances already on their way, and a simple rule captures almost all of it.** The size of that gain depends on how full hospitals are. That is a modest, true and defensible finding, which is more valuable than a large, unsupported one.

## 3. What we went through
- **Day 1 audit:** found six serious problems in the numbers and tables, and that the system in the paper did not exist in code.
- **Rebuild:** simulator, solver, PPO, API and console, calibrated against real traffic collected through the Mapbox API (about 40 minutes of collection).
- **The deletion incident (2026-10-26):** a bulk delete wiped the working folder because Windows treats differently-cased folder names as the same. It was rebuilt from the session transcript. Everything deterministic reproduced exactly; PPO did not, and its numbers were updated.
- **Search summaries were wrong more than once** (a statistic for the wrong time window, an old article presented as new). We opened pages before using figures.
- **Failures that taught us something:** the first online-learning version gave no gain; the first PPO attempts lost to the heuristic; the OpenStreetMap download failed (replaced by map tiles); search found no live bed data anywhere.
- **Reviews of our own work:** the realistic-traffic check showed our jam assumption was extreme, so we downgraded a claim; the measured-capacity study showed our headline gain describes a full system.
- **Venue rules read late:** COMPSIF is double-blind, so the repository link with the username cannot appear in the submission version.

## 4. Achievements
- A working, tested, containerised system with a dispatcher console, where there was a prototype.
- **13 documented studies**: module ablation, multi-objective trade-offs, online adaptation, a learned graph-network cost, stress sweeps, call locations, real call timing, specialty capability with public sources, three measured capacity profiles, and an independent real-traffic check.
- **Real data in every layer that matters:** traffic (two independent sources), hospital capacity (three periods, two countries), call timing and severity (NYC), call locations (map tiles).
- **Every claim traceable:** to a script, a result file and a test.
- **Negative results kept**, and our own earlier claims corrected when evidence disagreed.
- Documentation for another person to take over: README, model card, safety, deployment, performance, a handoff and this record.

## 5. What is still not there
No real Bengaluru bed counts, ambulance trips or outcomes; no field pilot; the paper itself is not yet rewritten (COMPSIF deadline 14 Nov 2026, 6 pages, double-blind); the hospital table's original source is unrecorded. See `HANDOFF.md` section 8.
