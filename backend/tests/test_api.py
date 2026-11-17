import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.pop("USE_SPARK_MODEL", None)

from backend.app.main import app  # noqa: E402

client = TestClient(app)


def post(**kw):
    body = {"miss_km": 0.34, "rel_vel": 14.2, "inc_div": 2.4, "bstar": 1.84}
    body.update(kw)
    return client.post("/api/predict", json=body)


def test_health():
    assert client.get("/api/health").json() == {"status": "ok"}


def test_predict_high():
    r = post()
    assert r.status_code == 200
    j = r.json()
    assert j["high_risk"] is True
    assert j["level"] == "HIGH CONJUNCTION RISK"
    assert j["chip"] == "CRITICAL EVENT"
    assert j["pc"] >= 1e-4
    assert j["miss_m"] == 340
    assert j["engine"] == "RULES"
    assert 0 <= j["score"] <= 1


def test_predict_high_with_minimum_modifiers():
    assert post(inc_div=0, bstar=0.1).json()["high_risk"] is True


def test_predict_low():
    j = post(miss_km=3).json()
    assert j["high_risk"] is False
    assert j["level"] == "LOW CONJUNCTION RISK"


def test_predict_low_even_with_max_modifiers():
    j = post(miss_km=3, rel_vel=16, inc_div=15, bstar=5).json()
    assert j["level"] == "LOW CONJUNCTION RISK"


def test_predict_moderate_band():
    # find a miss distance that lands between 2.5e-5 and 1e-4
    j = post(miss_km=1.5, inc_div=0, bstar=0.1).json()
    assert 2.5e-5 <= j["pc"] < 1e-4
    assert j["level"] == "MODERATE CONJUNCTION RISK"


def test_pc_monotonic_in_miss_distance():
    a = post(miss_km=0.2).json()["pc"]
    b = post(miss_km=1.0).json()["pc"]
    c = post(miss_km=5.0).json()["pc"]
    assert a > b > c


@pytest.mark.parametrize("field,value", [
    ("miss_km", 0.001), ("miss_km", 11), ("rel_vel", 0.1), ("rel_vel", 17),
    ("inc_div", -1), ("inc_div", 16), ("bstar", 0.01), ("bstar", 6),
])
def test_predict_validation(field, value):
    assert post(**{field: value}).status_code == 422


def test_stats_shape_and_demo_flag():
    j = client.get("/api/stats").json()
    for k in ("tracked", "debris", "high_risk", "pr_auc", "demo"):
        assert k in j


def test_objects_shape():
    objs = client.get("/api/objects").json()
    assert len(objs) >= 700
    o = objs[0]
    for k in ("norad", "name", "type", "alt", "inc", "ecc", "regime", "raan", "phase", "spd"):
        assert k in o
    assert {x["regime"] for x in objs} >= {"LEO", "SSO", "MEO", "GEO", "HEO"}
    assert {x["type"] for x in objs} <= {"PAYLOAD", "ROCKET BODY", "DEBRIS"}


def test_objects_deterministic():
    assert client.get("/api/objects").json() == client.get("/api/objects").json()


def test_metrics_shape():
    j = client.get("/api/metrics").json()
    assert "demo" in j and j["models"] and j["importance"]
    assert set(j["confusion"]) == {"tn", "fp", "fn", "tp"}
    assert sum(m["selected"] for m in j["models"]) == 1


def test_frontend_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "ORBITAL" in r.text
