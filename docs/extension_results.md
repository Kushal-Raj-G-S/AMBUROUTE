# Beyond the paper: two extension studies

*Exploratory: 30 held-out seeds (1000-1029), tuned policies unchanged, means only (no significance tests). Reproduce with `python experiments/run_extension.py`, then `python experiments/make_extension_figs.py`.*

## E1. What if the bed counts are very old?

Our paper tested bed data refreshed every 1-30 minutes. In Bengaluru's 2020 bed portals, hospitals were reportedly *required* to update only every **six hours** ([Udayavani](https://english.udayavani.com/article/all-pvt-hospitals-ordered-to-update-bed-availability-data/895519); not verified at the source), and reporters found the numbers lagging. We therefore extended the refresh interval.

> **Note on "frozen":** episodes last at most about three hours, so refresh intervals of 180 and 360 minutes behave identically: the published bed counts never refresh during an episode. We call this **frozen** rather than claiming six-hour-old data. At the start of each episode the published counts are fresh, so this is a mild version of reality.

Mean time to bed (minutes):

| Refresh | Peak: Nearest | Peak: Heuristic | Peak: Nash | Surge: Nearest | Surge: Heuristic | Surge: Nash |
|---|---:|---:|---:|---:|---:|---:|
| 5 min | 25.96 | 22.77 | 22.84 | 27.51 | 22.78 | 22.77 |
| 30 min | 26.39 | 23.36 | 23.04 | 27.90 | 24.00 | 23.57 |
| 60 min | 26.41 | 23.86 | 23.57 | 27.91 | 24.44 | 24.05 |
| Frozen | 27.60 | 25.00 | 24.72 | 28.99 | 24.99 | 24.64 |

Rejections per episode (peak): heuristic 3.9 → 25.6, Nash 3.7 → 21.6, nearest-hospital 32.3 → 47.7.

![Staleness study](images/staleness.png)

**What it shows**
1. **Stale data erodes the benefit but does not remove it.** With frozen beds the Nash assignment is still about **10 % faster** than nearest-hospital dispatch at peak (24.72 vs 27.60 min).
2. **The equilibrium step's edge grows as data ages.** At 5 minutes Nash and the heuristic are equal; from 30 minutes on Nash is 0.3-0.4 min faster, with 16-25 % fewer rejections at peak (exploratory; the paper's tests found the difference significant only at 30 minutes at peak).
3. **Product implication:** timely bed reports are worth real minutes, which supports building a simple hospital-side reporting channel (see `pilot_design.md`).

## E2. City scale with real OpenStreetMap geography

OpenStreetMap lists **1,137 hospital records** in the Bengaluru area; **110 distinct hospitals** are tagged as offering emergency services. Only **one** record has a bed count, so beds are **assumed**, drawn from a log-normal fitted to the 18 hospitals with known beds (total 67,166 beds, 6.4x the original set). **Geography is real; capacity is not.** Map data © OpenStreetMap contributors (ODbL).

Mean time to bed (minutes):

| Load | Scenario | Nearest | Nearest + beds | Heuristic | Nash |
|---|---|---:|---:|---:|---:|
| Same number of calls | Peak | 10.86 | 10.60 | 10.71 | 10.66 |
| Same number of calls | Surge | 10.95 | 10.72 | 10.80 | 10.73 |
| **Same load per bed** | Peak | 15.02 | 13.55 | 12.01 | **11.90** |
| **Same load per bed** | Surge | 16.37 | 15.34 | 12.71 | **12.47** |

95th percentile (same load per bed): peak 40.9 → 24.5 min, surge 46.6 → 25.1 min (nearest → Nash).

![City-scale study](images/scale.png)

**What it shows**
1. **When capacity is abundant, the policies barely differ** (same calls spread over 110 hospitals: under 2 % apart). Herding only hurts when beds are tight.
2. **When the load per bed matches the original scenarios, the gains are bigger than in the paper:** time to bed falls **21 % (peak) and 24 % (surge)** and the 95th percentile by about **40-46 %**, because crews have many near-by hospitals to pile onto.
3. **Product implication:** the value is concentrated in tight-capacity moments (peak hours, surges, ICU). That is also where public evidence says the pain is (97 % of Bengaluru's COVID ICU beds were occupied in the 2020 surge; see `market_research.md`).

**Limits:** assumed beds, uniform call locations, one traffic model, 30 seeds, no significance tests. These are directions for a journal-length extension, not headline claims.
