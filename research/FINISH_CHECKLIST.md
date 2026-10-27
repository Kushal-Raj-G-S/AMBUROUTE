# Finish checklist: production-level project, then the paper

Written 2026-10-27 so that one session can finish everything after credits refresh. Read `research/PROJECT_LOG.md` first (sections 10-12 describe the current state). Work top to bottom; each block ends with a check that must pass before moving on. Tick items as they are done and commit after every block (local first, push only on the user's word).

**Credit-saving rules for the session:** run scripts in the background and read only summary lines; generate docs and paper tables from the CSVs by script instead of retyping numbers; no screenshots except the ones listed; no web research beyond block A; no subagents; one build of the paper, one Overleaf round trip.

**Decisions needed from the user before starting (answer in one message):**
- [ ] Push to the public GitHub repo? (all commits so far are local; repo is public; `.env` is git-ignored)
- [ ] Author list for the paper (currently blank by rule)
- [ ] Is a database/auth layer wanted now, or is "production-ready prototype" (documented, tested, containerised, with stated limits) enough?
- [ ] Which hospitals' capability sources can the user or mam confirm by phone/website?

---

## A. Data truth (about 25 tool calls)
- [ ] Open each source URL in `data/hospital_capabilities_researched.csv` and mark verified / not verified in the CSV (columns `verified`, `checked_on`); fix wrong entries. Campus ambiguity: Narayana, Manipal, Fortis, Sagar, Baptist.
- [ ] Research the remaining unresearched rows (BMCRI, Columbia Asia, The Bangalore Hospital, Narayana, Bowring) once, mark "none found" if nothing.
- [ ] Find and cite the original source of the 18-hospital table (name, coordinates, beds). Record it in `data/` README and the paper.
- [ ] Verify the remaining open (circle) rows in `research/sources.md` by opening the page; fix any figure that does not match.
- [ ] Check COMPSIF's AI-assistance disclosure policy; record the exact wording and URL in the log.
- **Check:** every number that will appear in the paper has a verified source or is labelled assumed.

## B. Re-tune and rerun on researched inputs (compute only, about 6 tool calls)
- [ ] Re-tune the clinical heuristic weight and the clinical Nash beta/gamma on **validation** seeds using the researched capabilities and the real NYC severity mix (adapt `experiments/tune_clinical.py`).
- [ ] Rerun `run_clinical_researched.py` with the re-tuned parameters (test seeds, once). Keep the not-re-tuned run as a robustness row.
- [ ] Rerun `run_stress.py`, `run_online.py` clinical combination with researched capabilities (optional, only if time).
- [ ] Regenerate all result CSVs; compute a manifest `results/MANIFEST.csv` (file, sha256, script, date).
- **Check:** all 19 tests pass; headline paper numbers unchanged (22.64 / 22.91 etc.).

## C. Production hardening of the service (about 30 tool calls)
Security and safety
- [ ] API-key header for write endpoints (`/hospitals/*/beds`, `/feedback`, simulate-updates off by default); constant-time compare; key from env.
- [ ] Request limits: body size, max ambulances per `/assign`, rate limiting (slowapi or a simple token bucket), CORS allow-list from env.
- [ ] Input validation tests (bad lat/lon, bad specialty, negative beds, huge lists) and error-format tests.
- [ ] Never log tokens or patient coordinates beyond need; add a privacy note (coordinates are not stored by default).
- [ ] Dependency pinning (`requirements.txt` with versions, `requirements-dev.txt`), `pip-audit` in CI, secret scan (gitleaks) in CI.
Reliability
- [ ] Persist hospital bed state and learned corrections (SQLite file, path from env) with a startup load and atomic writes; test restart persistence.
- [ ] Routing fallback chain test: Mapbox failure -> OSRM -> built-in model, each exercised with a mocked failure; timeouts and retries set.
- [ ] Staleness guard: bed reports carry a timestamp; reports older than N minutes are flagged in the response and down-weighted or ignored (documented rule; ties to the paper's staleness study).
- [ ] Explainability in the response: top 3 hospitals with ETA, free beds, in-flight count, capability, and why the first won.
- [ ] Human-in-the-loop: every response states "recommendation, dispatcher decides"; add an override endpoint that logs the dispatcher's final choice (for shadow-mode evaluation).
- [ ] Shadow-mode logger (append-only JSONL: request, recommendation, actual choice, outcome fields blank) matching `docs/pilot_design.md`.
Operations
- [ ] Structured logging (JSON), request ids, `/health` (liveness) and `/ready` (model, traffic file, db), `/metrics` (Prometheus text, optional).
- [ ] Dockerfile: non-root user, pinned base image, healthcheck, `.dockerignore` check; `docker-compose.yml` with a volume for the database; build and run once locally and call `/health`.
- [ ] Config via env only (`.env.example` complete): token, API key, db path, CORS, rate limits, model path.
- [ ] Load test (locust or a short script): p95 latency of `/route` and `/assign` at 50 and 200 requests/s on this machine; record in `docs/performance.md`.
- **Check:** `pytest` (expect 30+ tests), `ruff check`, `docker build` and a container smoke test, load-test numbers recorded.

## D. Console polish (about 8 tool calls, 2 screenshots)
- [ ] Show the top-3 comparison and the reason text from C.
- [ ] Show staleness badge per hospital; show the "assumed capabilities" label (already present).
- [ ] Console works with the API key (stored in session memory only).
- [ ] Final screenshots: dispatch (clinical) and compare tab; update README images.

## E. Repository, docs, release (about 10 tool calls)
- [ ] README refresh: results table (paper + additional studies), what is real vs assumed, quick start, Docker, API table, limits, citation block; `CITATION.cff`.
- [ ] `docs/deployment.md` (Docker, env, backup, upgrade), `docs/safety.md` (intended use, not a medical device, failure modes, human override), `docs/model_card.md` (policies, data, limits).
- [ ] `docs/additional_studies.md` regenerated tables from CSVs by a script (`experiments/make_docs_tables.py`) so numbers cannot drift.
- [ ] CI: tests, lint, audit, secret scan, docker build; badge.
- [ ] Final `research/PROJECT_LOG.md`, `research/sources.md` updated; copy `research/` into the repo (done once; keep in sync).
- [ ] After the user says push: check images and `research/screenshots` for tokens, push, tag `v1.0.0`, create a GitHub release.
- **Check:** fresh clone in a temp folder installs, passes tests and serves `/health`.

---

## F. The paper (new, 6 pages, IEEE, no authors until decided) (about 25 tool calls)
Inputs are ready: `docs/additional_studies.md`, `results/*.csv`, `paper/` build scripts. Existing paper rules still apply: keep mam's 16 references and Table I; added references only from IEEE, Springer or MDPI (one kept exception: Schulman PPO); every number from a run; negative results reported.
- [ ] Decide the story (recommended): the original herding result + congestion game + calibrated traffic, then an "extensions" section with: module ablation, severity/specialty with researched capabilities, online adaptation, realism checks (spatial demand, NYC arrivals/severity). Mention GNN and multi-objective in two sentences each as negative/neutral results.
- [ ] Page budget table first: Abstract 0.2, Intro 0.7, Related work 0.6 (mam's, keep), Method 1.2, Setup and real-vs-assumed table 0.6, Results 1.7, Extensions 0.7, Limits 0.3, Conclusion 0.2, References 0.8. Cut figures before text.
- [ ] State explicitly: "peak" is a stress scenario, not a measured evening spike; capabilities are researched lower bounds; survival curve assumed.
- [ ] Extend `make_paper_assets.py` for the new small tables (ablation, researched-capability study, realism checks) and figures (module ablation bars, day profile or stress curve).
- [ ] Update `paper/main_template.tex` and `build_paper.py` in one pass; `check_tex.py`; user compiles on Overleaf; check the PDF page count and citations with pymupdf; at most two fix rounds.
- [ ] Plain-prose pass: short sentences, concrete, no filler. **The user then rewrites the flagged and key passages in their own words** (the AI-detector score is not engineered around). Include an honest AI-assistance statement if COMPSIF requires one (from A).
- [ ] mam to verify the related-work comparisons (Fu 2025, Xu 2024, Bhatia 2024, Sutherland 2023) against the actual papers; update if she corrects them.
- [ ] Optional after acceptance: journal-length version, viva Q&A sheet.
- **Check:** 6 pages, no broken citations, every number traceable to a CSV in `results/`, availability link works.

---

## Definition of done
All checks above pass; repo fresh-clone test passes; manifest and log updated; local commits pushed only with the user's approval; user has the new PDF and the viva notes.

## Rough effort
Blocks A-E about 75-85 tool calls plus compute; block F about 25. One focused session if credits allow; otherwise stop after any block (each ends in a green state and a local commit).
