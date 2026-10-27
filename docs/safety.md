# Intended use and safety

## What this is

A decision-support prototype that suggests which hospital an ambulance should go to, with the reasons, and a simulator that compares dispatch rules. **It is not a medical device, has not been clinically validated, and must not be the sole basis for a dispatch decision.** Every recommendation carries the notice "Recommendation only; the dispatcher decides", and `POST /decision` records when the dispatcher chose differently so that disagreement can be studied.

India's medical-device software guidance (CDSCO, final version July 2026, see [`market_research.md`](market_research.md)) treats decision-support software as potentially regulated depending on intended use. Any real deployment needs a regulatory review for that use.

## Known limits

- **Bed data in the simulator is simulated**, including ICU capacity and occupancy. The live service starts from simulated beds too, until hospitals report real counts.
- **Specialty capabilities** (cardiac, stroke, trauma) come from public sources and are a **lower bound**: a service with no findable source is counted as absent. One entry (Victoria's trauma centre) could not be verified at its source, and two (Aster RV, Fortis) are only partly verified. See `data/hospital_capabilities_researched.csv`.
- **Time to a bed or to definitive care is a proxy**, not survival. The survival curve in the multi-objective study is assumed.
- Travel times follow typical traffic fitted to Mapbox data (11.3 % error on held-out routes); the ambulance speed-up over cars (1.25) is an assumption.
- Results are for one city and 18 hospitals and compare policies; they do not predict real response times.
- Stale bed reports are flagged, not corrected: a hospital that stops reporting is still considered, with a visible warning.

## Failure behaviour

| Failure | What happens |
|---|---|
| Mapbox unavailable or over quota | routing falls back to OSRM, then to the built-in travel-time model; the response says which source was used |
| No hospital has the requested beds | the service returns 404 with a reason; it does not invent a recommendation |
| Learned model missing | the PPO endpoint falls back to the equilibrium solver |
| Bed reports stop | affected hospitals are flagged stale in every response and in `/ready` |
| Unauthorised write | rejected with 401 when `API_KEY` is set |

## Privacy

The service does not store incident locations or patient data. Requests carry coordinates only to compute a route; access logs record method, path, status, latency and a request id, not the body. The shadow log stores hospital ids and an optional short reason. Do not put patient identifiers in the reason field.

## Before any real use

1. A shadow-mode pilot ([`pilot_design.md`](pilot_design.md)) with real bed reports, comparing recommendations with dispatchers' actual choices.
2. Replacement of simulated beds with reported ones and review of the capability table with each hospital.
3. Clinical review of how severity and specialty are used, and a regulatory assessment.
4. Load and failure testing on the target infrastructure.
