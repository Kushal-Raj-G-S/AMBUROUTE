# PROJECT_LOG: AmbuRoute (read this first in any new session)

Last updated: 2026-10-26. Written so that nothing depends on conversation memory.

## 1. What this is
An IEEE conference paper (COMPSIF, **hard limit 6 pages**) plus code for ambulance hospital selection and routing, by a BMSIT AIML student under Dr. Archana Bhat (mam). The original paper had unreproducible numbers and an unsupported "DRL improves" claim; the work was rebuilt from scratch with real experiments. The user is now also aiming to raise it to a startup-grade body of work (research, pilot design, product polish) without founding a company. They do not mind the repo being public.

## 2. Where everything lives
Project root: `E:\BMSIT\Research paper\COMPSIF_AMBULANCE-Archana mam\`

| Path | Role |
|---|---|
| `conference_101719.tex`, `ref.bib`, original PDFs/images | **mam's originals. Never modify.** |
| `ambuRoute/` | Working copy with `.venv`, raw Mapbox CSVs (`results/mapbox_calibration_part*.csv`, not published), trained models, all results |
| `repo_stage/` | **Git checkout of the GitHub repo; push from here.** Has `docs/`, CI, Dockerfile, README |
| `paper/` | `main_template.tex` + `build_paper.py` build `main.tex`; tables come from `experiments/make_paper_assets.py`; `check_tex.py` checks structure |
| `recovery/` | Rebuild artefacts from the 2026-10-26 incident (replay scripts, console screenshots in `shots/`, backup of original paper tables) |
| `research/` | This folder: log, source ledger, proof screenshots |

- GitHub: https://github.com/Kushal-Raj-G-S/AMBUROUTE (public, MIT, CI passing). `gh` active account `Kushal-Raj-G-S`.
- Overleaf project: https://www.overleaf.com/project/6ade5b17d48d9344683cb865 ("Compsif amburoute"). Overleaf ignores simulated paste: the **user must press Ctrl+A, Ctrl+V** after the text is copied.
- Mapbox public token is in `ambuRoute/.env` (git-ignored). Never write it into a document.
- No LaTeX compiler on this machine; the user compiles on Overleaf and sends the PDF in Downloads (`Compsif_amburoute.pdf`); check pages with `pymupdf`.

## 3. Rules the user set
1. **No authors** in the paper (empty author block; decided later with mam). Do not add names.
2. **Keep all 16 of mam's references and her Table I.**
3. Any reference I add must be from **IEEE, Springer or MDPI** (exception kept: Schulman PPO, arXiv).
4. **6 pages max.** Prose must read natural; no tool names or watermarks in the paper. Do not hide AI use from the venue; check COMPSIF's disclosure policy.
5. Every result must come from real runs. Report negative results honestly (PPO matches but does not beat the tuned heuristic).
6. Be thrifty with tokens; no chatter while waiting.
7. User does not mind public repo or mam's opinion; wants the work "outstanding". Honesty about limits is still required.

## 4. Headline results (test: 50 held-out seeds, seeds 1000-1049; validation 500-529; training >=100000)
Mean minutes from call to bed / rejections per episode:

| Scenario | Nearest | Capacity heuristic | Nash | PPO |
|---|---|---|---|---|
| Peak (1.2 calls/min) | 25.78 / 30.1 | 22.73 / 3.5 | 22.64 / 2.8 | 22.94 / 5.5 |
| Surge (2.5 calls/min) | 27.75 / 46.8 | 22.97 / 6.2 | 22.91 / 5.6 | 23.50 / 11.1 |

- Capacity awareness: -12 % (peak), -17 % (surge) vs nearest; rejections down 87-93 %. Nash = heuristic on speed (differences not significant), 10-20 % fewer rejections, better load balance (Jain 0.51 vs 0.48 at peak). PPO slower than heuristic by 0.9-2.3 % at peak/surge.
- Nash converged in all 4,239 batches (2,747 peak + 1,492 surge), mean 1.005/1.023 rounds, max 2/3. Proof via Rosenthal potential function.
- Tuned on validation seeds: heuristic w=20; Nash beta=10, gamma=2.
- PPO: config B = k=12 candidates, herding coef 0.3, behaviour-cloning warm start (150 episodes), lr 1e-4, clip 0.1, ent 0.005, 1M steps, calibrated traffic. **Selected seed B_s1** (validation: s1 21.818, s2 21.826, s0 22.070; tuned heuristic 21.513). PPO vs old prototype rule: peak -0.70 (36/50 seeds, Bonferroni p 0.001), surge -1.73 (42/50), off-peak ns.
- Traffic calibration: 351 pairs, 3,861 samples (3,960 queried), 105 held-out pairs; error 25.8 % -> 11.3 %, R2 0.52 -> 0.90; power-law form beat linear in CV (12.3 vs 17.0 %); residual sigma 0.14. Ambulance speed-up 1.25 over car times is an **assumption**.
- Statistics: Wilcoxon signed-rank with Holm correction; bootstrap 95 % CIs (5,000 resamples).
- Extensions (30 seeds, exploratory, in repo `docs/`): staleness up to "frozen" (Nash still ~10 % faster than nearest; its edge over heuristic grows with staleness); city scale with 110 OpenStreetMap emergency hospitals, **beds assumed** (67,166, 6.4x): same load per bed -> time to bed -21 % (peak), -24 % (surge).
- **Clinical extension (severity + specialty)**, `docs/clinical_extension.md`, code in `amburoute/clinical.py` + `clinical=True` flag in `SimConfig` (off by default; regression check: 24 re-run episodes identical to saved paper results). Assumed: severity mix 55/30/15, specialty need (critical 85 %, urgent 25 %), capabilities by hospital size (cardiac 5, stroke 8, trauma 7 of 18; file `data/hospital_capabilities.csv`), transfer overhead 15 min, severity weights 1/1.5/3. Tuned on validation seeds: clinical heuristic w=80; clinical Nash beta=40, gamma=1. Results (30 test seeds, critical patients, minutes to definitive care): peak nearest 46.38, nearest-capable 33.85, specialty-blind heuristic 41.36, clinical heuristic 27.47, clinical Nash 28.41; surge 48.94 / 34.81 / 41.52 / 27.88 / 28.19. Clinical vs blind: -32 to -34 %; clinical Nash vs nearest-capable: -16 % (peak), -19 % (surge); Nash ~ heuristic again (not significant); severity weights worth ~1 min (peak heuristic corrected p=0.006). Exact-potential property proved and unit-tested for the weighted, restricted game. 12 tests now.
- Pilot sizing (simulated variance): beating nearest-hospital needs 3-15 matched days; Nash vs heuristic needs 458-5,675 days, so ship the simple heuristic first.

## 5. What is real vs simulated
Real: hospital names/coordinates/bed totals for 18 hospitals (3 missing counts imputed, 2 non-emergency removed; **source of this table still uncited**), traffic (Mapbox typical-traffic fit). Simulated: ICU beds (8 % of beds), occupancy, calls (Poisson, uniform), ETA noise. Results compare policies; they do not predict real response times.

## 6. Incidents and lessons
- **2026-10-26: I ran `rm -rf AMBUROUTE`; on Windows that is the same folder as `ambuRoute` and it wiped the working folder.** Recovery: replayed Write/Edit calls from the session transcript (`recovery/replay.py`), re-collected 3,960 Mapbox samples, retrained PPO. Deterministic results reproduced exactly (tuning identical to every digit; Nash, heuristic, baselines, convergence identical). Calibration reproduced to ~3 decimals (held-out MAPE 11.284 vs 11.285). **PPO is not bit-reproducible**: retrained seeds differ, selected seed changed B_s0 -> B_s1, paper PPO numbers updated. One ablation cell changed 23.07 -> 23.11 (speed-up 1.5, surge, nearest). Lesson saved to persistent memory: never delete case-variant folder names.
- Shell pitfalls: `python - <<'EOF'` and heredocs through the Bash tool can mangle backslashes; use the Write/Edit tools or script files for anything with `\`.
- Browser pane screenshots are always <= 800 px wide; sharper captures are not possible. Reddit and some news sites are blocked; do not work around.
- The first search summaries contained errors that only opening the pages caught (72 % figure is for 30 minutes; Citizen Matters is from 2013; the 31-year-old patient story is April 2021). **Open the page before relying on a figure.**

## 7. Rebuild recipes
```
cd ambuRoute
.venv\Scripts\python -m pytest tests -q
.venv\Scripts\python experiments\collect_mapbox_traffic.py <shard> 6      # run 0..5 in parallel; needs MAPBOX_TOKEN; query date must still be in the future
.venv\Scripts\python experiments\calibrate_traffic.py
.venv\Scripts\python experiments\tune_baselines.py
.venv\Scripts\python -m amburoute.train_ppo --tag B --seed N --steps 1000000 --k 12 --herding 0.3 --bc 150 --lr 1e-4 --clip 0.1 --ent 0.005 --calibrated
.venv\Scripts\python experiments\validate_ppo.py ; run_benchmark.py ; run_ablations.py ; run_staleness_tests.py ; run_ppo_vs_prototype.py
.venv\Scripts\python experiments\make_paper_assets.py ; cd ..\paper ; ..\ambuRoute\.venv\Scripts\python build_paper.py ; ..\ambuRoute\.venv\Scripts\python check_tex.py
```
The Mapbox collection script's date is 2026-10-28; after that date it must be changed.

## 8. Paper state
6 pages, 24 references = mam's 16 + 8 that I added (Rosenthal, Baan-Kooman, Eidstø, Jánošíková, Wu: Springer; Tluli et al. IWCMC: IEEE; Murareţu: MDPI; Schulman PPO: arXiv, the one exception). Latest compiled PDF checked 2026-10-26: link to the repo is clickable, no broken citations. Availability sentence says raw Mapbox responses are not redistributed.

## 9. Open items
- Cite the hospital table's original source.
- mam should check my related-work comparisons (Fu 2025, Xu 2024, Bhatia 2024, Sutherland 2023) against the actual papers; I only had titles/abstracts.
- Check COMPSIF's AI-assistance disclosure policy.
- Decide the author list with mam.
- **mam's feedback (received 2026-10-27, pasted by the user):** "Paper is fine. You can add any one or more of the following points: (1) multi-objective optimisation (travel time + survival probability + waiting time + fairness); (2) patient severity (triage)-aware routing with differentiated priorities; (3) GNN to model hospital-ambulance interactions instead of hand-crafted congestion costs; (4) online learning that adapts to changing traffic and hospital conditions; (5) validation using real EMS or hospital datasets, even if limited; (6) ablation studies showing the contribution of each module (traffic calibration, in-flight load modelling, congestion game, PPO warm start)." She also said the text "has 22% AI. reduce to 0%". Status: the paper content was accepted; the six points are optional additions; the AI-detector score concerns the writing. Already done in the repo but not in the paper: (2) via the clinical extension, (6) partly via staleness, outage, speed-up, beta and PPO-attempt results. Not done: (1) beyond severity weights, (3), (4), (5) beyond real traffic and OpenStreetMap geography. **Position on the 22 % AI score:** do not engineer text to evade detectors; the honest route is the student rewriting prose in their own words with understanding and being open with mam and the venue about AI-assisted drafting. Ask mam for the sentence-level report from her checker.
- Verify the remaining ○ rows in `sources.md`; finish competitor search (US EMS tools such as Pulsara not covered).
- DONE 2026-10-26: severity/specialty extension; `docs/briefing_for_mam.md` (one-page summary for mam, with honest limits and four questions for her guidance).
- Candidate next builds: journal-length paper combining the clinical, staleness and city-scale studies with full significance testing; real capability data for hospitals (public sources) to replace the assumed capabilities; verify remaining ○ sources; a short viva Q&A cheat sheet.
- GitHub repo holds a README, `docs/market_research.md`, `docs/extension_results.md`, `docs/pilot_design.md`; keep them consistent with `research/sources.md`.

## 10. Session 2026-10-27: building mam's add-ons (plan: build, test, new proof, THEN a new paper; paper later)
User decision: build mam's points first, then write a newest paper and handle the 22 % AI score by rewriting prose in own words (no detector-evasion). Code is in repo_stage (not yet committed/pushed as of this note); logs in research/*.log.
- Sim hooks added (defaults unchanged; 12 tests pass; module-ablation full-system row reproduces paper 22.641/22.913): planner_traffic, SimConfig.hazard_per_hour, drift_n/drift_factor/drift_start/obs_frac, Request.t_first_arrival, use_inflight flags on CapacityHeuristic/NashPolicy.
- Module ablation (experiments/run_module_ablation.py, results/module_ablation.csv; 50 test seeds, nothing re-tuned): removing in-flight load costs +4.6 % (peak) / +9.9 % (surge), p<0.001, rejections 2.8->11.3 / 5.6->24.2; removing traffic calibration +0.9 % peak (n.s. after correction) / +1.3 % surge (p=0.004); removing the congestion game (heuristic instead) +0.4 % / +0.3 % (n.s. on time; more rejections 3.5 vs 2.8, 6.2 vs 5.6); nearest hospital +13.9 % / +21.1 %. PPO warm-start ablation (run_ppo_warmstart.py): without behaviour cloning PPO is 24.36 (peak) / 25.76 (surge) vs 22.94 / 23.50 with it (+6 % / +10 %, p<0.001), seeds spread 24.4-25.8; warm-started B_s1 vs heuristic +0.9 % peak (n.s. after correction), +2.3 % surge (p<0.001).
- Multi-objective (experiments/run_multiobjective.py, results/mo_*.csv, docs/images/multiobjective.png): time, assumed survival exp(-hazard*t), door wait, load fairness (Jain). Finding: objectives mostly AGREE; front is narrow; balanced pick = paper's clinical heuristic (w=80, s=1, lam=0). Survival differences tiny (0.913-0.915) because the hazard curve is assumed. Honest conclusion: no strong trade-off found.
- Online learning (experiments/run_online.py, results/online_*.csv, docs/images/online.png): drift = 4 random hospitals 2x slower from minute 30. Fixed 26.21, online 24.58 (-6.2 %), oracle 23.42 (-10.6 %) (peak, heuristic); surge -2.6 % vs oracle -7.8 %. Costs: no drift +0.6 % (n.s.), citywide slowdown x1.5 +1.6-2.1 % worse. First version (arrival-time feedback only) FAILED (no gain) and learner state leaked across episodes until fixed; final learner uses assumed GPS progress at 35 % of trip (obs_frac) and relative per-hospital corrections.
- REAL DATA (point 5): NYC EMS Incident Dispatch Data (NYC Open Data 76xm-jjuj, 2023, aggregates in data/real_nyc_ems_*). Findings: hourly call share is FLAT in the afternoon/evening (17-19h vs 12-14h ratio 1.01), so our 'peak' scenario (1.2 vs 0.5 calls/min) is a stress test, NOT a measured evening demand spike - paper wording must say so. Severity levels 1-2 / 3-4 / 5+ = 22.2 % / 35.0 % / 42.7 % (assumed mix was 15/30/55). Rerunning the clinical study with the real mix: clinical Nash vs nearest-capable -23 % (peak) / -21 % (surge), vs specialty-blind Nash -28 % / -26 % (all p<0.001); clinical Nash ~ heuristic again. Median call-to-hospital in NYC 42.7 min (includes scene time; not comparable to our time-to-bed). NYC priority codes are dispatch priority, not clinical triage. SF ambulance patient offload data and Karnataka 108 call-level data: none found openly (CAG audit report is the Karnataka source).
- GNN (point 3, DONE): amburoute/gnn.py + experiments/run_gnn.py (learned door-delay cost on a hospital kNN graph, 77,407 exploratory training dispatches, ablation without edges). Cost scale picked on validation (6.0 at the top of the grid; first grid topped at 2.0 and was widened, log kept in gnn_first_grid.log). Test: peak 22.59 vs heuristic 22.73 vs Nash 22.64, surge 22.90 / 22.97 / 22.91; none significant; edges add nothing measurable. Verdict: learned cost matches hand-crafted, does not beat it.
- Docs written: docs/additional_studies.md (all five studies), README updated. 17 tests pass (tests/test_extensions.py includes a bit-exact check against saved paper results). Local commits only (b8a427b, cb6b06f, then docs); NOT pushed.
- Still to do: (a) push when user says so (check screenshots for tokens first; repo is public); (b) make_paper_assets for new tables; (c) draft new 6-page paper, plainer voice, say 'peak' is a stress scenario; (d) user rewrites prose in own words; (e) still open: cite hospital-table source, mam verify related work, COMPSIF AI-disclosure policy, authors.

## 11. Session 2026-10-27 (later): cheap items
- DONE: stress tests (run_stress.py), day-in-the-life profile (run_day_profile.py); results in docs/additional_studies.md sections 6-7; commit ae6a3bf (local, not pushed).
- BLOCKED: spatial-demand rerun. sim.py supports SimConfig(spatial_demand=True) and experiments/run_spatial_demand.py exists, but data/bengaluru_building_density.csv could not be fetched: public Overpass servers (overpass-api.de, kumi.systems, private.coffee) timed out or returned 502/504 on 2026-10-27 evening; fetch_osm_density.py aborted at cell (1,1). Retry later (script is not resumable; row 0 only). I wrongly told the user it was running; it had already failed.
- NOT started: real hospital capability/ICU sources (web research), API/console exposure of clinical + online features, NYC day replay beyond the hourly profile.

## 12. Session 2026-10-27 (evening): remaining project items finished
- Spatial demand: Overpass servers were down; replaced by Mapbox vector-tile building counts (experiments/fetch_density_mapbox.py, 462 tiles, 0 failed; only aggregate counts saved). Result: all times fall (calls near the centre); capacity-awareness gain shrinks to -10 % peak / -13 % surge (uniform: -12 / -17); Nash = heuristic; PPO trained on uniform calls still ~ heuristic (docs section 8).
- NYC arrival replay: 2-min count dispersion index 0.92 (Poisson fine); on replayed arrivals heuristic -14.2 % vs nearest, Nash ~ heuristic (section 9). sim.py now accepts explicit arrival times (run_episode(..., arrivals=)); random stream of earlier results unchanged (tests check bit-exact reproduction).
- API/console: /route and /assign take severity + specialty (clinical heuristic / clinical Nash), new POST /feedback learns per-hospital corrections (use_online_correction); console has severity, specialty and 'learn from trip times'; screenshot docs/images/console_clinical.jpg; 19 tests pass.
- Researched capabilities: data/hospital_capabilities_researched.csv (cardiac 4, stroke 3, trauma 4 of 18, evidence + URLs, from search-result summaries NOT individually opened; verify at source before citing; several campus matches ambiguous). Clinical policies still beat nearest-capable by ~21-22 %, but capable hospitals saturate (rejections 15-22) (section 10). The ASSUMED capability file (data/hospital_capabilities.csv) marked Jayadeva as non-cardiac: the paper's clinical numbers rest on that assumed file, so any new paper should present the researched version as the primary one or state this clearly.
- All local commits (latest 86bd2a8 + this log); NOT pushed. .env copied into repo_stage is git-ignored (never commit it).
- Remaining: new paper; user's rewrite; push decision; verify capability sources at the source; open items from section 9.

## 13. Block B done (2026-10-27 night)
Re-tuned clinical policies on researched capabilities + real NYC severity mix (tune_clinical_researched.py -> best_clinical_params_researched.json: heuristic w=320 [grid top, optimum may be higher], Nash beta=80 power=1). One-shot test (30 seeds): retuned heuristic 38.9 / 41.7 min (peak/surge) vs nearest capable 52.3 / 56.8, specialty-blind heuristic 46.7 / 48.1; -24..-27 % vs nearest capable (p_bonf<=0.004), -17 % / -13 % vs specialty-blind heuristic; Nash ~ heuristic (n.s.). Docs section 10 updated. Next from FINISH_CHECKLIST (root research/): A (verify capability sources), C-E (hardening, docs, release), F (paper). Not pushed.

## 14. Block A (partial): capability sources opened 2026-10-27
Verified at source: Jayadeva (primary angioplasty), NIMHANS (thrombolysis + 24h thrombectomy, DHN updated 15 May 2026), KC General (round-the-clock angioplasty, DHN updated 29 Nov 2024), St. Martha's (Angels: primary centre in stroke network). Verified self-reported: Manipal OAR (cath lab for emergency angioplasty), HOSMAT (24/7 trauma). Partly: Aster RV (biplane cath lab named, PCI not stated), Fortis (sponsored; no cath lab named). UNVERIFIED: Victoria trauma (DH 403; Wikipedia 404; myhospitalnow unreliable). Capability set unchanged, so no rerun; CSV has verification + checked_on columns. Still open in Block A: remaining unresearched rows (BMCRI, Columbia Asia, The Bangalore Hospital, Narayana, Bowring), hospital-table origin, sources.md circle rows, COMPSIF AI policy.

## 15. COMPSIF rules (read from https://bmsit.ac.in/compsif/ on 2026-10-27; use this URL, not searches)
- **Submission deadline 14 Nov 2026 AoE (firm)**; notification 1 Jan 2027; camera-ready 15 Jan 2027.
- Research paper max **6 pages including figures, tables and references**; IEEE template, double column, PDF. (Industry and WIP papers max 4.)
- **Double-blind review: remove names, affiliations, acknowledgements; self-citations in third person; papers posted on arXiv-like sites are desk-rejected.**
- **ACTION: the paper's Availability sentence links github.com/Kushal-Raj-G-S/AMBUROUTE, which names the author. For the submission use no link (or an anonymised mirror) and add the real link only in the camera-ready.** Also check the repo title/README do not appear in a way that deanonymises (the repo is public).
- Mandatory plagiarism screening per IEEE guidelines; **no generative-AI policy stated and no similarity percentage limit** on the page. Mam's 22 % AI figure is a Turnitin AI-writing indicator, not a stated venue limit. Still be open about AI-assisted drafting if asked.
- Author list is final at submission; at least one author presents in person; IEEE copyright form for accepted papers.

## 16. Block C (API hardening) done 2026-10-27
api.py: optional X-API-Key on write endpoints (constant-time compare), CORS allow-list, per-address rate limit, MAX_AMBULANCES, request ids + JSON access logs, STATE_PATH persistence (atomic write; bed state + learned corrections), stale bed flag (STALE_MINUTES) per hospital, explanation (top-3 by ETA, free beds, ambulances on the way, capable, stale, plain-language reason) + "recommendation only" notice in /route responses, /ready, POST /decision (dispatcher override logged to SHADOW_LOG, no patient data). tests/test_hardening.py: 7 tests; 26 tests pass. Not yet done in Block C: Docker rebuild + smoke test, load test, lint/CI, pinned deps.

## 17. Block C continued (2026-10-27)
Dockerfile hardened (non-root, healthcheck, /data volume), docker-compose.yml, requirements-dev.txt (pytest, ruff, pip-audit, mapbox-vector-tile, dotenv), ruff.toml (lint clean, 6 auto-fixes), CI now lint + tests + pip-audit (report-only) + gitleaks + docker build/start. **Docker image NOT built locally (Docker Desktop daemon not running); CI will test it on first push.** Load test: /assign 225 rps, p50 168 ms, p95 605 ms at 50 concurrent; /route bound by Mapbox (14 rps burst, p50 3 s at 50 concurrent) - docs/performance.md. 26 tests pass.

## 18. Console redesign (2026-10-27)
amburoute/web/index.html fully redesigned: floating glass panel over a calm Mapbox light/dark style (light-v11 / dark-v11), light + dark themes (auto from OS, toggle saved in localStorage), teal-to-blue brand gradient with indigo info and green/amber/coral status colours, segmented controls (severity colour-coded) built from the original hidden selects so every id and API call is unchanged, hero ETA card with capacity bars, "why this hospital" callout, compared-alternatives table, stale badges, empty-state guide, loading spinners, route casing + hospital halos, pulsing incident marker, comparison tab as bar cards, mobile bottom-sheet layout. Routes and hospital layers re-added after a theme (style) switch. Capability badge now reads 'public sources' (API uses data/hospital_capabilities_researched.csv). Screenshots in docs/images/console_{single,multi,compare}.jpg. 26 tests pass.

## 19. Block E mostly done (2026-10-27)
Added docs/deployment.md, docs/safety.md, docs/model_card.md, CITATION.cff (authors: 'AmbuRoute contributors'), README links. Fresh-clone test passed: clone of the local repo has no .env, 26 tests pass, ruff clean. STILL OPEN: Docker image never built locally (daemon was off; CI builds it on first push); original source of the 18-hospital table (ask the student/original repo history); five unresearched hospital rows; unchecked market-report sources (circle rows in sources.md); docs tables not yet script-generated; push + v1.0.0 release (user's go-ahead, check screenshots for tokens first); the new paper (parked by the user on 2026-10-27; COMPSIF deadline 14 Nov 2026, double-blind: no repo link with the username in the submission).

## 20. Console v2 + Docker verified (2026-10-27)
UI rebuilt again as a night-shift dispatch console after the first redesign read as generic (design guidance applied: subject-specific identity, avoid default gradients/Inter/rounded-card look): top status strip (live clock, traffic-model LED, free beds/ICU/reporting count, live cursor coordinates), docked rail instead of floating glass, Big Shoulders Display + IBM Plex Sans/Mono, hi-vis lime as the single bold accent, triage-tag colours for priority, Battenburg-checker mark, results printed as a dispatch ticket (priority band, huge ETA with count-up, ECG trace, bed-cell gauges, 'why', decision trace bars), animated ambulance unit driving each route, radar standby state, hospital labels and stale hollow rings on the map, day/night themes. All ids and API calls unchanged. Docker: image builds (1.93 GB), non-root, healthy, /ready ok, 401 without key, state persisted across restart, shadow log written (smoke-tested with a named volume; note Git Bash rewrites /data paths unless MSYS_NO_PATHCONV=1).
- Typography softened on request (2026-10-27): Fraunces (soft serif) for headings and the drive-time numeral, Figtree (rounded humanist sans) for text, DM Mono for data; sentence case instead of capitals; friendly rounded buttons with soft shadow; README screenshots retaken.

## 21. Release (2026-10-28)
Last five hospital rows researched (nothing counted: campus ambiguous or weak sources), docs/results_tables.md generated by experiments/make_docs_tables.py. Token scan of the whole git history: clean. Pushed main (22 commits, aa53be3 + this log) and tag v1.0.0 to github.com/Kushal-Raj-G-S/AMBUROUTE. The GitHub *release page* was NOT created (blocked by the permission check); release notes are ready in the session scratchpad and can be pasted via 'gh release create v1.0.0 --notes-file ...' or the web UI. CI (lint, tests, audit, gitleaks, docker build) started on push; check the Actions tab. Still open: original source of the 18-hospital table (not recorded; ask the original author), the unchecked market-report sources (marked circle in sources.md), the paper (parked; COMPSIF deadline 14 Nov 2026, double-blind), Victoria trauma entry unverified.

## 22. Real hospital capacity data search (2026-10-28)
No free, open, LIVE per-hospital bed feed found anywhere (commercial: Bamboo Health, OpenBeds; DC ArcGIS bed-utilization layer errored on query; Ireland's real-time dataset 'does not exist'; ICUBAM is manual entry; SF diversions monthly). Real facility-level HISTORICAL data exists: HHS 'COVID-19 Reported Patient Impact and Hospital Capacity by Facility' (healthdata.gov dataset anag-cw7u, weekly, to 2024-04, US federal data, exact coordinates, inpatient beds, beds used, ICU beds, ICU used; verified by query) and Germany's DIVI ICU register archived on Zenodo (daily, hospital-level ICU, 2020-2023, not yet opened). Measured on 3 weeks (2023-04, 2023-10, 2024-04), 4,916 US hospitals: general-bed occupancy p5/p25/p50/p75/p95 = 0.25/0.51/0.71/0.86/0.98 (assumed in simulator: 0.88-0.99 for every hospital); ICU occupancy median 0.73; ICU beds = 11.7 % of inpatient beds at the median (simulator: 8 %); occupancy moves ~7 points week to week; only 18 % of hospital-weeks had < 10 % free beds. => the simulator's headline regime is the tight tail of reality; benefits should be reported as conditional on capacity pressure (consistent with the scale study). Candidate next steps: rerun with real occupancy mix; 'real metro' run with HHS hospitals + Mapbox traffic; importer to seed the service from a real snapshot (historical replay, not live).

## 23. Measured-capacity version built (2026-10-28)
Chose the Bengaluru version, with a sampler that works for any hospital table (Karnataka/other cities would need only their table + traffic). Files: experiments/fetch_hhs_capacity.py (6 weeks 2023-2024 of HHS facility data -> data/real_capacity_profile.csv, 14,202 anonymous hospital-weeks), amburoute/capacity.py (size-matched draw), SimConfig.real_capacity, experiments/run_real_capacity.py, docs/additional_studies.md section 11, results/real_capacity_*.csv. Result: re-tuned parameters equal the paper's (w=20, beta=10, power=2). Real-capacity free beds avg 20 % vs 6.5 % assumed. Heuristic vs nearest: -2.7 / -3.9 / -5.6 % (off-peak/peak/surge) vs -5.5 / -11.8 / -17.2 % under assumed capacities; turn-aways -93..-96 % still. Nash ~ heuristic (<0.4 %); PPO (not retrained) 0.3-0.8 % slower. Benefit rises with pressure in the surge (7.3 % tight vs 3.2 % loose), not clean at peak. IMPORTANT for any paper/claims: the old headline 12 %/17 % is the tight-capacity regime; report both. US-derived profiles, not Bengaluru measurements. 28 tests pass. Committed locally; not pushed.
