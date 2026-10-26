# Pilot design: AmbuRoute in shadow mode

*A design proposal, not a plan that has been agreed with any hospital or operator. Numbers marked "simulated" come from our simulator and must be re-estimated from real data.*

## 1. Purpose and intended use

**Intended use (proposed):** logistics decision support for a human dispatcher. The tool recommends a destination hospital and route with a stated reason; **the dispatcher and crew decide**. It does not triage patients or give clinical advice. Stating the intended use this narrowly matters for regulation (see `market_research.md`, section 5).

**Primary question:** does recommending a hospital that accounts for beds *and* ambulances already en route reduce the time from call to admission and the number of patients turned away, compared with current practice?

## 2. Phased design

| Phase | What happens | Patient impact | Exit criterion |
|---|---|---|---|
| 0. Data audit | Agree a bed-reporting channel with 2-3 hospitals; collect anonymised past trip logs (call time, pickup area, hospital chosen, arrival, admission time) | None | At least 3 months of logs; bed reports arriving at least every 15 min |
| 1. **Shadow mode** | The system runs silently next to dispatch and logs what it *would* have recommended. Nothing is shown to crews | None | Enough matched days (section 4); no data or latency failures |
| 2. Advisory | Recommendations are shown to dispatchers, who may accept or override. Every override is logged with a reason | Minimal; human decides | Acceptance rate and override reasons reviewed by clinicians |
| 3. Evaluation | Compare advisory days with matched baseline days | Minimal | Pre-registered metrics (section 3) |

Shadow mode first is deliberate: it proves the data pipeline and measures prediction error without any risk.

## 3. Metrics (fix before starting)

**Primary:** median and 95th-percentile **time from call to admission to a bed** (minutes).
**Secondary:** share of patients **turned away** at the first hospital; ICU-patient diversions; ETA error (predicted vs actual travel time); dispatcher acceptance rate; decision latency; agreement between shadow recommendation and what crews actually did.
**Safety and fairness:** any patient outcome flagged by clinicians; outcomes broken down by pickup area, time of day and hospital type, to catch an equilibrium that overloads some neighbourhoods.

## 4. How many days? (simulated variability)

Paired design (same day, two policies), two-sided alpha 0.05, power 0.80. From `experiments/pilot_power.py`:

| Scenario | Comparison | Mean difference | Days needed |
|---|---|---:|---:|
| Off-peak | Nash vs nearest hospital | -1.12 min | 15 |
| Evening peak | Nash vs nearest hospital | -3.14 min | 3 |
| Surge | Nash vs nearest hospital | -4.83 min | 4 |
| Evening peak | Nash vs capacity heuristic | -0.09 min | 458 |
| Surge | Nash vs capacity heuristic | -0.06 min | 1,940 |

**Reading:** showing that capacity awareness beats "nearest hospital" would take **days to a few weeks** of matched data. Separating the equilibrium assignment from the simple heuristic is **not feasible in a pilot**; the two are effectively equivalent on speed. **Recommendation: ship the simple, explainable capacity heuristic first** and keep the equilibrium solver for multi-ambulance surges, where it reduces rejections.

## 5. Data requirements and a minimal bed-report spec

Hospitals do not need a national feed. A lightweight report is enough:

```json
POST /hospitals/{id}/beds
{ "free_beds": 12, "free_icu_beds": 2, "reported_at": "2026-11-02T14:05:00+05:30", "source": "ward-nurse-app" }
```

Targets: refresh at most every **5 minutes** (Bengaluru's mandated portal refresh was six hours, which our staleness study suggests costs time; see `extension` results), explicit `reported_at` so the system can discount old reports, and a **fail-safe**: if a hospital's report is older than a set limit, treat its beds as unknown and fall back to nearest-with-a-recent-report.

## 6. Safety, privacy, ethics

- **Human in the loop** at all times; one-click override; a visible "why this hospital" explanation.
- **Fail-safe defaults:** if traffic, bed data or the service itself is unavailable, show the nearest hospital with a recent bed report and say so.
- **Privacy:** log pickup *area* and times, not names or identifiers. India's Digital Personal Data Protection Act 2023 applies if personal data are processed; seek legal review before collecting any.
- **Consent and governance:** written agreement with each hospital and the operator; an ethics or institutional review before phase 2; a named clinical lead.
- **Fairness monitoring:** track outcomes by area; an equilibrium that balances hospital load must not systematically lengthen trips for one neighbourhood.
- **Audit trail:** every recommendation, input snapshot and override is logged for review.

## 7. Risks and open questions

| Risk | Mitigation |
|---|---|
| Hospitals report beds late or wrongly | Timestamps, staleness discounting, agreement checks against crew reports |
| Crews ignore recommendations | Study override reasons; improve explanations |
| Traffic model drift | Re-fit against fresh Mapbox samples; alert on rising ETA error |
| Regulatory classification unclear | Define intended use; seek regulatory advice before phase 2 |
| Buyer and incentives unclear | Interviews with 108 operators, private fleets and hospital groups (open question 3 in the market report) |
