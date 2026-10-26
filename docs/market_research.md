# AmbuRoute: problem evidence, landscape and regulatory notes

*Secondary research from public sources, compiled 26 October 2026. No primary interviews were conducted. Several sources are dated (2014-2020); each is marked. Reddit could not be reached by our search tool, so forum voice is represented by news and journal reporting. Figures from vendors and market-research firms are self-reported and should be treated as rough.*

## 1. Is the problem real in Bengaluru? Evidence

| Finding | Source | Date / caveat |
|---|---|---|
| A government audit of Karnataka's 108 service found the 20-minute urban response target was met in only **72 %** of cases | [CAG audit report, Chapter II](https://cag.gov.in/webroot/uploads/download_audit_report/2020/5.Chapter%20II-05fd1cdd9ce5711.66448512.pdf); coverage: [The News Minute](https://www.thenewsminute.com/article/108-ambulances-failed-reach-50-patients-within-time-karnataka-cag-139331) | Audit period 2014-15 to 2018-19 |
| Bengaluru's 108 call centre took roughly **250-300 calls a day**; average urban arrival about **14.5 min** | [Citizen Matters](https://citizenmatters.in/plenty-ambulances-in-city-but-not-enough-for-needy/) | Article date not shown; treat as dated |
| Bengaluru Traffic Police data show accident victims who died **on the way to hospital** rising from 175 (2021) to **292** (2023) | [Deccan Herald opinion, "No green for the ambulance"](https://www.deccanherald.com/opinion/no-green-for-the-ambulance-3403011) | Opinion piece citing police data |
| A patient was **denied admission by 9 hospitals** and died in the ambulance; another died after being turned away from several hospitals while searching for an ICU bed | [The News Minute, 31-year-old patient](https://www.thenewsminute.com/article/31-year-old-bengaluru-covid-19-patient-dies-after-struggling-find-icu-bed-146937); [The News Minute, harrowing stories](https://thenewsminute.com/article/harrowing-stories-bengaluru-covid-19-patients-who-struggled-get-medical-help-128066) | **COVID surge, 2020**: extreme conditions, but exactly the failure mode AmbuRoute targets |
| Ambulance drivers in Hyderabad and Ahmedabad name **congested streets** as a major obstacle | [Journal of Emergency Medicine survey](https://wmpllc.org/ojs/index.php/jem/article/view/1316) | Two cities, not Bengaluru |
| A review notes **over 40 %** of transported patients are *not* taken to the nearest hospital, and capacity of the destination is one reason | [Review of EMS destination choice](https://revistas.javeriana.edu.co/index.php/gerepolsal/article/download/24409/21467) | International; no India-specific study of how crews choose was found |

**What this means for AmbuRoute:** the two delays we model (time on the road and being turned away at the door) are both documented in Bengaluru. No India-specific study of *how a crew picks a hospital* turned up, which is itself a research gap.

## 2. Bed information in India: what exists and how fresh it is

| Finding | Source | Caveat |
|---|---|---|
| Bengaluru ran a public **BBMP bed dashboard** covering 102 hospitals (ICU, ventilators) | [The News Minute](https://thenewsminute.com/article/bengaluru-hospital-bed-availability-can-now-be-tracked-through-bbmp-dashboard-128587) | 2020, COVID-era |
| A state order required hospitals to refresh bed data **every six hours** | [Udayavani](https://english.udayavani.com/article/all-pvt-hospitals-ordered-to-update-bed-availability-data/895519) | 2020 |
| Private-hospital bed data sat in a separate portal that **did not allocate beds** | [The News Minute](https://www.thenewsminute.com/article/dashboard-bed-status-private-bengaluru-hospitals-launch-may-3-148141) | 2020 |
| Portal figures **lagged reality**; some listed beds were not offering service | [Deccan Herald](https://www.deccanherald.com/india/karnataka/bengaluru/bengaluru-faces-icu-bed-shortage-after-slight-spike-in-covid-19-cases-887159.html) | 2020 |
| Delhi launched an **ICU bed dashboard** for 17 government hospitals in July 2024 | [PIB document](https://static.pib.gov.in/WriteReadData/specificdocs/documents/2026/jul/doc202676912801.pdf) (via search summary; verify) | Government hospitals only |
| The national **Health Facility Registry** is a directory of facilities; no evidence it exposes live bed counts | [IndianPharmaPost](https://www.indianpharmapost.com/policy/lite/nha-announces-successful-onboarding-of-1-lakh-healthcare-facilities-12640) | Confirm on ABDM sandbox docs |

**What this means:** the data AmbuRoute needs exists only in pieces, is often hours old and is not reliable. Our simulated 5-30 minute staleness is optimistic against a six-hour mandated refresh. A product would need a lightweight way for hospitals to report beds (our console already has a bed-update endpoint) rather than waiting for a national feed.

## 3. Landscape

**Indian ambulance operators (mostly fleet dispatch and booking, not hospital-aware routing in the sources found):**
- **StanPlus** (Hyderabad): in-house StanFleet platform; reported Series A of about USD 20 M in 2022 ([YourStory](https://yourstory.com/2022/01/funding-alert-ambulance-startup-stanplus-healthquad-kalaari-capital-healthx/amp)).
- **Medulance** (Delhi): aggregator with a stated 5,000+ ambulances in 22 cities ([BioSpectrum](https://biospectrumindia.com/features/85/19748/startupoftheday-45-medulance-healthcare.html)); self-reported.
- **Ziqitza Healthcare**: modernised dispatch with a vendor ([case study](https://www.casestudies.com/company/daffodil-software/case-study/ziqitza-healthcare-ltd-serves-3-million-patients-with-daffodil-software)); vendor-authored, figures not verified.
- **GVK EMRI** runs the state 108 service under a public-private partnership ([CAG](https://cag.gov.in/webroot/uploads/download_audit_report/2020/5.Chapter%20II-05fd1cdd9ce5711.66448512.pdf)). We did not find public detail on its dispatch software.

**Global computer-aided dispatch (CAD):** market estimates range from about USD 2.1 billion (2023, [Global Market Insights](https://gminsights.com/industry-analysis/computer-aided-dispatch-market/market-size)) to USD 3.0 billion (2025, [IMARC](https://www.imarcgroup.com/computer-aided-dispatch-market)). Named vendors include Hexagon, CentralSquare, Caliber and Avtec; Motorola Solutions acquired RapidDeploy in 2025. The reports describe a shift from on-premises to cloud CAD.

**Gap:** none of the sources we found describe hospital-aware destination selection that accounts for ambulances already en route. That is AmbuRoute's niche, but a thorough product search (US EMS tools, hospital-communication products) was not completed and should be done before any strong claim of novelty.

## 4. Traffic context

Bengaluru's adaptive traffic-control system was reported at **169 junctions** with an emergency-vehicle priority feature, operations beginning April 2025 ([Indian Infrastructure](https://indianinfrastructure.com/2026/04/06/smart-mobility-a-case-study-of-the-bengaluru-adaptive-traffic-control-system/); [The South First](https://thesouthfirst.com/news/bengaluru-163-traffic-junctions-to-be-upgraded-with-ai-based-adaptive-signal-system)). The Deccan Herald opinion piece above argues its effect on ambulances needs assessment. **Opportunity:** AmbuRoute chooses *where to go*; signal priority speeds *getting there*. They are complementary.

## 5. Regulatory notes (not legal advice)

- India's CDSCO issued **draft guidance on medical device software** dated 21 October 2025, covering software that provides clinical decision support or treatment recommendation; devices fall in risk classes A-D under the Medical Devices Rules 2017 ([Nishith Desai Associates](https://www.nishithdesai.com/research-and-articles/hotline/pharma-healthcare-update/from-code-to-compliance-decoding-indias-medical-device-software-guidance-15623); [India Briefing](https://india-briefing.com/news/cdsco-draft-guidance-medical-software-40691.html)). Sources disagree on whether it is still a draft; **check the CDSCO website**.
- A dispatcher-facing tool that *recommends a hospital* sits close to decision support. Whether it is regulated as a medical device depends on its intended use, so intended use should be defined early (for example, logistics and routing support with a human dispatcher making the decision).
- India's **Digital Personal Data Protection Act 2023** applies if patient data are processed. Our current system uses no patient data.

## 6. Open questions to resolve before any pilot

1. How do 108 crews and private operators actually choose a hospital today (phone calls, experience, control-room instruction)? A short set of interviews would settle this.
2. Which hospitals would report beds, how often, and through what channel?
3. Who is the buyer: a state 108 operator, a private fleet, or a hospital network?
4. Intended-use statement and regulatory classification (see section 5).
5. Safe-failure design: the system must degrade to "nearest hospital with a bed" if data are stale or missing.
