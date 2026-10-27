import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

BODY = {"ambulance": {"lat": 12.97, "lon": 77.59}}


@pytest.fixture
def api(monkeypatch, tmp_path):
    """Reload the API module with a given environment; restore the default module afterwards."""
    import amburoute.api as mod

    def make(**env):
        for k, v in env.items():
            monkeypatch.setenv(k, str(v))
        return importlib.reload(mod), TestClient(mod.app)

    yield make
    monkeypatch.undo()
    importlib.reload(mod)


def test_write_endpoints_need_the_api_key_when_one_is_set(api):
    mod, c = api(API_KEY="s3cret")
    assert c.post("/hospitals/0/beds", json={"free_beds": 3}).status_code == 401
    assert c.post("/hospitals/0/beds", json={"free_beds": 3}, headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.post("/hospitals/0/beds", json={"free_beds": 3}, headers={"X-API-Key": "s3cret"}).status_code == 200
    assert c.post("/feedback", json={"hospital_id": 0, "planned_min": 5, "observed_min": 6}).status_code == 401
    assert c.post("/decision", json={"recommended_id": 0, "chosen_id": 1}).status_code == 401
    assert c.get("/hospitals").status_code == 200  # reads stay open
    assert c.post("/route", json=BODY).status_code == 200  # recommendations stay open


def test_no_key_configured_means_open_writes_for_local_use(api):
    mod, c = api(API_KEY="")
    assert c.post("/hospitals/0/beds", json={"free_beds": 3}).status_code == 200


def test_rate_limit_and_request_ids(api):
    mod, c = api(RATE_LIMIT_PER_MIN=3)
    codes = [c.get("/hospitals").status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429]
    assert c.get("/health").status_code == 200  # health checks are never limited
    r = c.get("/health", headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123"


def test_request_size_limits(api):
    mod, c = api(MAX_AMBULANCES=3)
    amb = {"lat": 12.97, "lon": 77.59}
    assert c.post("/assign", json={"ambulances": [amb] * 3}).status_code == 200
    assert c.post("/assign", json={"ambulances": [amb] * 4}).status_code == 422
    for bad in ({"ambulance": {"lat": 100, "lon": 0}}, {"ambulance": {"lat": 1, "lon": 200}}, {"ambulance": BODY["ambulance"], "severity": 5},
                {"ambulance": BODY["ambulance"], "specialty": "x"}, {"ambulance": BODY["ambulance"], "min_beds": -1}):
        assert c.post("/route", json=bad).status_code == 422
    assert c.post("/hospitals/0/beds", json={"free_beds": -1}).status_code == 422


def test_state_survives_a_restart(api, tmp_path):
    state = tmp_path / "state.json"
    mod, c = api(STATE_PATH=state)
    c.post("/hospitals/2/beds", json={"free_beds": 11, "free_icu_beds": 2})
    for _ in range(4):
        c.post("/feedback", json={"hospital_id": 5, "planned_min": 10, "observed_min": 20})
    factor = c.post("/feedback", json={"hospital_id": 5, "planned_min": 10, "observed_min": 20}).json()["correction_factors"][5]
    assert state.exists() and not state.with_suffix(".tmp").exists()
    mod2, c2 = api(STATE_PATH=state)  # a fresh process reading the same file
    h = c2.get("/hospitals").json()[2]
    assert h["available_beds"] == 11 and h["available_icu_beds"] == 2
    assert abs(mod2._online_factors()[5] - factor) < 1e-3  # the API rounds to 3 decimals


def test_stale_reports_are_flagged_and_explained(api):
    mod, c = api(STALE_MINUTES=15)
    assert not any(h["stale"] for h in c.get("/hospitals").json())
    mod._updated[:] -= 60 * 60  # every report is now an hour old
    assert all(h["stale"] for h in c.get("/hospitals").json())
    r = c.post("/route", json=BODY).json()
    assert r["meta"]["notice"].startswith("Recommendation only")
    assert len(r["meta"]["alternatives"]) >= 3 and sum(a["chosen"] for a in r["meta"]["alternatives"]) == 1
    assert all(a["stale"] for a in r["meta"]["alternatives"]) and r["meta"]["reason"]
    assert c.get("/ready").json()["stale_hospitals"] == len(c.get("/hospitals").json())


def test_ready_and_shadow_decision_log(api, tmp_path):
    log = tmp_path / "shadow.jsonl"
    mod, c = api(SHADOW_LOG=log)
    assert c.get("/ready").status_code == 200
    assert c.post("/decision", json={"recommended_id": 1, "chosen_id": 1}).json()["agreed"] is True
    assert c.post("/decision", json={"recommended_id": 1, "chosen_id": 2, "reason": "road closed"}).json()["agreed"] is False
    assert c.post("/decision", json={"recommended_id": 1, "chosen_id": 99}).status_code == 404
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert [x["agreed"] for x in lines] == [True, False] and lines[1]["reason"] == "road closed"
    assert not any(k in lines[0] for k in ("lat", "lon", "name"))  # no patient or location data
    mod2, c2 = api(SHADOW_LOG=tmp_path / "missing_dir" / "x.jsonl")
    assert c2.get("/ready").status_code == 503
