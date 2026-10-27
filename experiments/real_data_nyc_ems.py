"""Check the simulator's demand and severity assumptions against a real, public EMS dispatch log:
NYC "EMS Incident Dispatch Data" (NYC Open Data, dataset 76xm-jjuj). Only aggregates are saved.

    python experiments/real_data_nyc_ems.py
Outputs: data/real_nyc_ems_hourly.csv, data/real_nyc_ems_severity.csv, data/real_nyc_ems_times.json
"""
import json
import sys
from io import StringIO
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

BASE = "https://data.cityofnewyork.us/resource/76xm-jjuj.csv"
YEAR = "incident_datetime between '2023-01-01T00:00:00' and '2023-12-31T23:59:59'"


def soql(**q):
    url = BASE + "?" + "&".join(f"${k}={quote(str(v), safe=',()*')}" for k, v in q.items())
    with urlopen(url, timeout=180) as r:
        return pd.read_csv(StringIO(r.read().decode("utf-8")))


hourly = soql(select="date_extract_hh(incident_datetime) as hr, count(*) as n", where=YEAR, group="hr", order="hr")
hourly.to_csv(ROOT / "data" / "real_nyc_ems_hourly.csv", index=False)

sev = soql(select="initial_severity_level_code as sev, count(*) as n", where=YEAR, group="sev", order="sev")
sev.to_csv(ROOT / "data" / "real_nyc_ems_severity.csv", index=False)

rows = soql(select="incident_datetime,first_to_hosp_datetime,first_hosp_arrival_datetime,incident_close_datetime,"
                   "initial_severity_level_code,dispatch_response_seconds_qy,incident_travel_tm_seconds_qy",
            where=YEAR + " AND first_hosp_arrival_datetime IS NOT NULL AND first_to_hosp_datetime IS NOT NULL",
            limit=200000)
for c in ("incident_datetime", "first_to_hosp_datetime", "first_hosp_arrival_datetime", "incident_close_datetime"):
    rows[c] = pd.to_datetime(rows[c])
leg = (rows.first_hosp_arrival_datetime - rows.first_to_hosp_datetime).dt.total_seconds() / 60
at_hosp = (rows.incident_close_datetime - rows.first_hosp_arrival_datetime).dt.total_seconds() / 60
call_to_hosp = (rows.first_hosp_arrival_datetime - rows.incident_datetime).dt.total_seconds() / 60
ok = (leg > 0) & (leg < 120)
q = lambda x: {f"p{p}": round(float(np.percentile(x, p)), 2) for p in (5, 25, 50, 75, 95)}
out = dict(
    n_transports_sampled=int(len(rows)),
    transport_leg_min=q(leg[ok]),
    transport_leg_lognormal_sigma=round(float(np.log(leg[ok]).std()), 3),
    call_to_hospital_arrival_min=q(call_to_hosp[(call_to_hosp > 0) & (call_to_hosp < 300)]),
    arrival_to_incident_close_min=q(at_hosp[(at_hosp > 0) & (at_hosp < 300)]),
    dispatch_response_s=q(rows.dispatch_response_seconds_qy.dropna()),
)
(ROOT / "data" / "real_nyc_ems_times.json").write_text(json.dumps(out, indent=2))
h = hourly.set_index("hr").n
print(json.dumps(out, indent=2))
print("hourly share (%):", (100 * h / h.sum()).round(2).to_dict())
print("peak/off-peak ratio 17-19h vs 12-14h:", round(float(h.loc[17:18].mean() / h.loc[12:13].mean()), 2))
print("severity counts:", sev.set_index("sev").n.to_dict())
