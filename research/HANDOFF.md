# AmbuRoute handoff: read this if the AI assistant, the conversation or the subscription is gone

Written 2026-10-28. Everything needed to understand, run, extend and defend the project is in this repository. Nothing depends on the chat history. Keep this file and `PROJECT_LOG.md` updated whenever something changes.

## 1. What this project is
A traffic- and capacity-aware system for choosing which hospital an ambulance should go to, for a Bengaluru conference paper (COMPSIF, BMS Institute of Technology, guide Dr. Archana Bhat). Parts:
- a simulator of Bengaluru (18 hospitals, travel times fitted to real Mapbox traffic data),
- dispatch policies: nearest hospital, a capacity-aware heuristic, an equilibrium (congestion-game) assignment, PPO, a graph-network cost, clinical (severity and specialty) versions, an online travel-time correction,
- a FastAPI service and a dispatcher console (`amburoute/web/index.html`),
- a reproducible evaluation with held-out seeds, plus extension studies and real-data checks.

## 2. Where everything is
Project root: `E:\BMSIT\Research paper\COMPSIF_AMBULANCE-Archana mam\`

| Folder | What it is |
|---|---|
| `repo_stage/` | **The git repository. Work and push from here.** GitHub: https://github.com/Kushal-Raj-G-S/AMBUROUTE (public, MIT) |
| `repo_stage/research/` | project memory: `PROJECT_LOG.md` (chronological record, newest last), `sources.md` (market evidence ledger), `screenshots/` (proofs), run logs, this file |
| `research/` (project root) | a second copy of the same folder, plus `FINISH_CHECKLIST.md` (local only). If they differ, the repository copy is newer |
| `ambuRoute/` | the Python virtual environment (`.venv`) used for every run, plus an old working copy. Recreate it from `requirements.txt` and `requirements-dev.txt` if lost |
| `paper/` | LaTeX template and build scripts for the 6-page paper (parked; see section 8) |
| `recovery/` | files from the 2026-10-26 deletion incident (scripts that rebuilt the repository) |
| `_cleanup_removed/`, `_backup/` | files moved out during cleanup; safe to delete |
| `conference_101719.tex`, `ref.bib`, original PDFs and images | **the guide's original paper files. Never modify.** |

Git state when this was written: GitHub `main` at `c03720d` plus tag `v1.0.0`; **several newer commits exist only on this computer until pushed** (check with `git log origin/main..HEAD`). The GitHub release page for v1.0.0 was not created; notes are in `research/RELEASE_NOTES_v1.0.0.md` (project-root copy).

## 3. Rules the owner set (keep following them)
1. The paper has **no author names** until decided with the guide; the venue is **double-blind**, so no GitHub link containing the username in the submission version.
2. Keep all 16 of the guide's original references and her Table I. Any reference added must be from IEEE, Springer or MDPI (one kept exception: Schulman PPO, arXiv).
3. **6 pages including references**, IEEE template, PDF.
4. Every number in the paper comes from a real run; negative results are reported (PPO matches but does not beat the tuned heuristic).
5. Never delete folders by case-variant names (Windows is case-insensitive; this once wiped the working folder). Never commit `.env`.
6. Be honest about what is simulated and what is real. Do not engineer text to evade AI-detection tools; rewrite flagged prose in your own words and disclose AI-assisted drafting if asked.

## 4. How to run it
```bash
cd repo_stage
python -m venv ../ambuRoute/.venv        # if missing
../ambuRoute/.venv/Scripts/pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
../ambuRoute/.venv/Scripts/pip install -r requirements.txt -r requirements-dev.txt
../ambuRoute/.venv/Scripts/python -m pytest tests -q                 # 29 tests
../ambuRoute/.venv/Scripts/python -m ruff check amburoute tests experiments
# service + console (needs a free Mapbox public token in .env: MAPBOX_TOKEN=pk...)
../ambuRoute/.venv/Scripts/python -m uvicorn amburoute.api:app --port 8000     # open http://127.0.0.1:8000
docker build -t ambu-route . && docker compose up                    # container route (see docs/deployment.md)
```
Regenerate the documented tables: `python experiments/make_docs_tables.py`. Experiment scripts and what each produces are listed in the README and at the top of each script. The most important ones: `run_benchmark.py` (paper headline), `run_module_ablation.py`, `run_real_capacity.py [2023-24|2022|england-2026]`, `run_stress.py`, `run_online_realistic.py`, `analyse_gmaps_traffic.py`.

**Secrets and accounts:** the only secret is the Mapbox public token in `repo_stage/.env` and `ambuRoute/.env` (both git-ignored; a new free token works). GitHub access is through the `gh` CLI logged in as `Kushal-Raj-G-S`. The Overleaf project for the paper: https://www.overleaf.com/project/6ade5b17d48d9344683cb865. There is no LaTeX compiler on this machine; compile on Overleaf.

## 5. Read in this order to understand it
1. `README.md` (what it is, headline results, quick start)
2. `research/PROJECT_LOG.md` (everything done, in order, with the reasons, incidents and rebuild recipes)
3. `docs/additional_studies.md` (every extension study with tables and caveats, sections 1-13) and `docs/results_tables.md` (generated)
4. `docs/safety.md`, `docs/model_card.md`, `docs/deployment.md`, `docs/performance.md`, `docs/pilot_design.md`, `docs/market_research.md`
5. `research/sources.md` and `research/screenshots/` (evidence for market claims)
6. `research/JOURNEY.md` (the original draft versus the first rebuilt paper versus today, and what we went through)

## 6. Key results (all reproducible; details and caveats in the docs)
- **Paper headline** (simulated, tight capacity, 50 held-out seeds): minutes from call to bed, evening peak 25.78 (nearest), 22.73 (heuristic), 22.64 (equilibrium), 22.94 (PPO); surge 27.75, 22.97, 22.91, 23.50. Turned-away ambulances fall 87-93 %.
- **What matters:** counting ambulances already on the road (worth 5-10 % alone). Equilibrium, heuristic, PPO and the graph-network cost are statistically equivalent; ship the simple heuristic first.
- **Travel-time model:** fitted to 3,861 Mapbox samples (11.3 % error on 105 held-out routes); independently checked on 113,886 real Google Maps readings (hour shape correlation 0.94, rush-hour level 0.98; too slow at night and on long expressway routes).
- **Real jam sizes** are small (95 % of readings within 1.2x typical), so online ETA correction helps only for rare 1.5x+ jams; the earlier 6 % gain is a 2x stress test.
- **Benefit depends on hospital fullness** (measured capacity, three profiles): about 4-6 % faster in looser US hospitals (about 20 % of beds free), 10-12 % in 2026 English trusts (about 6 % free), with 92-97 % fewer turned-away ambulances in all. The paper's assumed 88-99 % occupancy matches full public systems, not hospitals in general.
- **Clinical extension** (severity and specialty, capabilities from public sources, a lower bound): critical patients reach definitive care about 24-27 % faster than "nearest capable hospital"; capable hospitals saturate when few exist.
- **Real data checks:** NYC EMS log (call timing close to Poisson, real severity mix), building density for call locations (benefit shrinks to 10-13 %).

## 6a. Data inventory (real versus simulated)
| Item | Status |
|---|---|
| Bengaluru map, hospital coordinates and bed totals (18 hospitals) | real; **original source of the table not recorded** (ask the original author) |
| Travel times | fitted to real Mapbox typical traffic (token needed to refetch; raw responses not published) |
| Traffic validation | 113,886 Google Maps readings, CC BY 4.0, Mahesh Shantaram, https://github.com/thecont1/traffic-monitor-lizard |
| Hospital capacity profiles | US federal facility data (2022 and 2023-24, healthdata.gov `anag-cw7u`, latest week 2024-04-21, series ended) and NHS England beds situation reports 2026-27 (licence not stated on the page; likely Open Government Licence, unconfirmed) |
| Call timing and severity mix | NYC EMS dispatch log (NYC Open Data), aggregates only |
| Call locations | building counts from Mapbox vector tiles (aggregates only) |
| Specialty capabilities | public sources, `data/hospital_capabilities_researched.csv` (one entry, Victoria trauma, unverified) |
| Beds, ICU occupancy and calls in the simulator | simulated (or drawn from the measured profiles); ambulance speed-up over cars (1.25) assumed |

## 7. Honest limits
Results compare policies; they do not predict real response times. No Bengaluru bed, call or ambulance-trip data was available. Capacity profiles are US and England, applied to Bengaluru hospitals by size. Time to a bed is a proxy for outcome. PPO training is not bit-for-bit repeatable (shipped weights in `results/models/`). Docker image was built and smoke-tested once locally; the repository's CI builds it on push. No field validation.

## 8. Open items and next steps
- **Paper (parked):** COMPSIF submission deadline is **14 Nov 2026 (firm)**; 6 pages; double-blind. A new draft should report the benefit as conditional on capacity pressure (use sections 11-13), say "peak" is a stress scenario, and be rewritten in the author's own words. The AI-writing indicator reported by the guide's checker was 22 %; the venue states no AI policy and no similarity limit (https://bmsit.ac.in/compsif/).
- Push the unpushed commits and create the v1.0.0 release page.
- Cite the original source of the hospital table; ask the guide to verify the related-work comparisons (Fu 2025, Xu 2024, Bhatia 2024, Sutherland 2023).
- Verify Victoria's trauma entry; the unchecked market-report sources (marked in `sources.md`).
- Production path (not done): real bed reporting from hospitals, a shadow-mode pilot, regulatory and privacy review, a database with several instances, real accounts and audit trail, hospital and ambulance apps.

## 9. Lessons (so they are not repeated)
- On 2026-10-26 an `rm -rf` of a case-variant folder name wiped the working folder; it was rebuilt from the session transcript. Everything deterministic reproduced exactly; PPO did not (retrained seeds differ).
- Search-engine summaries contained wrong figures twice; open the page before using a number.
- In Git Bash on Windows, heredocs and `python -` can mangle backslashes, and a path such as `/data/x` is rewritten unless `MSYS_NO_PATHCONV=1`.
- Some sites (NHS England) return an empty challenge to scripts; read them in a real browser.
