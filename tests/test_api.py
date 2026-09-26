"""API tests that never call Claude (auth, validation, budget cap, /trends)."""
import datetime as dt

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.llmops import tracking

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_log(tmp_path, monkeypatch):
    monkeypatch.setattr(tracking, "LOG", tmp_path / "calls.jsonl")
    monkeypatch.delenv("SERVICE_API_KEY", raising=False)


def weekly_events(cluster, counts, start=dt.date(2026, 6, 1)):
    return [{"cluster": cluster, "date": str(start + dt.timedelta(days=7 * w, hours=0))}
            for w, n in enumerate(counts) for _ in range(n)]


def test_health_reports_spend_and_cap():
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["spent_usd"] == 0 and body["cap_usd"] > 0


def test_key_required_when_configured(monkeypatch):
    monkeypatch.setenv("SERVICE_API_KEY", "secret")
    payload = {"events": weekly_events("x", [5] * 6)}
    assert client.post("/trends", json=payload).status_code == 401
    assert client.post("/trends", json=payload, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/trends", json=payload, headers={"X-API-Key": "secret"}).status_code == 200


def test_trends_flags_a_spike_and_reports_week_start():
    events = weekly_events("shipping delays", [5, 6, 5, 4, 6, 5, 27, 5])
    body = client.post("/trends", json={"events": events}).json()
    assert body["n_weeks"] == 8
    assert [s["week"] for s in body["spikes"]] == [6]
    assert body["spikes"][0]["week_start"] == "2026-07-13"


def test_trends_needs_five_weeks():
    r = client.post("/trends", json={"events": weekly_events("x", [5, 5, 5])})
    assert r.status_code == 422


def test_analyze_validates_input():
    assert client.post("/analyze", json={"reviews": []}).status_code == 422
    too_many = [{"id": str(i), "text": "ok"} for i in range(21)]
    assert client.post("/analyze", json={"reviews": too_many}).status_code == 422


def test_spend_cap_blocks_calls(monkeypatch):
    monkeypatch.setenv("MAX_SPEND_USD", "0.01")
    tracking.LOG.write_text('{"cost_usd": 0.02}\n', encoding="utf-8")
    with pytest.raises(tracking.BudgetExceeded):
        tracking.TrackedClient(object(), "test", "v0").create(model="claude-haiku-4-5")
    r = client.post("/analyze", json={"reviews": [{"id": "1", "text": "hello"}]})
    assert r.status_code == 429
