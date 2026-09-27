"""Demo app tests: no Claude calls (the key is removed and the live guards are checked)."""
import pytest

pytest.importorskip("gradio")

import app  # noqa: E402


def test_saved_examples_load_and_render():
    labels = app.example_labels()
    assert len(labels) == len(app.EXAMPLES) >= 10
    review, analysis, reply, judge, _ = app.show_example(labels[0])
    assert review and reply
    assert "Sentiment" in analysis and "Status" in judge


def test_saved_examples_cover_six_languages_and_include_a_weak_one():
    assert {e["language"] for e in app.EXAMPLES} == set(app.LANG)
    assert any(not e["judge"]["acceptable"] for e in app.EXAMPLES)  # the demo shows a case the judge rejected
    assert all(e["status"] == "pending_human_review" for e in app.EXAMPLES if e["sentiment"] == "negative")


def test_spike_tab_finds_the_dashboard_alerts():
    msg, table, weekly = app.run_spikes(3.5, 8)
    assert len(table) == 4
    top = table.iloc[0]
    assert top["issue"].startswith("Delayed") and top["negative reviews"] == 27 and abs(top["z"] - 12.56) < 0.01
    assert set(weekly["week"]) <= {f"W{i:02d}" for i in range(1, 13)}


def test_spike_tab_stricter_threshold_finds_fewer():
    _, strict, _ = app.run_spikes(6.0, 15)
    assert len(strict) < 4


def test_live_tab_guards_do_not_call_claude(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DEMO_ACCESS_CODE", raising=False)
    assert "Paste a review" in app.analyse_live("  ", "", True, "", 0)[3]
    assert "live requests" in app.analyse_live("late parcel", "", True, "", app.MAX_CALLS)[3]
    assert "No ANTHROPIC_API_KEY" in app.analyse_live("late parcel", "", True, "", 0)[3]
    monkeypatch.setenv("DEMO_ACCESS_CODE", "secret")
    assert "access code" in app.analyse_live("late parcel", "", True, "wrong", 0)[3]
