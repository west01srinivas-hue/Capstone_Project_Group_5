"""FastAPI service: /analyze, /draft-response, /trends.

  uvicorn src.api.main:app --port 8000
Environment: ANTHROPIC_API_KEY, CLAUDE_MODEL (default claude-haiku-4-5), JUDGE_MODEL (default claude-sonnet-5),
SERVICE_API_KEY (if set, requests must send it in the X-API-Key header), MAX_SPEND_USD (default 1.0),
RESPONSE_VERSION (default v2).
"""
import datetime as dt
import json
import os

import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from src.analysis import trends as tr

load_dotenv()
os.environ.setdefault("RESPONSE_VERSION", "v2")

from src.llmops import tracking  # noqa: E402
from src.response import generate as gen  # noqa: E402
from src.response import judge as jd  # noqa: E402
from src.response.generate import ROOT, parse_json, text_of  # noqa: E402

app = FastAPI(title="Customer Review Insights & Response Generator", version="0.1.0")

TOPICS = ["shipping", "quality", "support", "features", "price", "ux", "other"]
GEN_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-sonnet-5")
CLASSIFY_PROMPT = "classify_v3"
_assets = None


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("SERVICE_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="missing or wrong X-API-Key header")


def anthropic_client():
    import anthropic

    return anthropic.Anthropic()


def assets() -> dict:
    global _assets
    if _assets is None:
        _assets = gen.load_assets()
    return _assets


def stars_to_label(stars: float) -> str:
    s = int(round(min(max(stars, 1), 5)))
    return "negative" if s <= 2 else "neutral" if s == 3 else "positive"


class ReviewIn(BaseModel):
    id: str
    text: str = Field(min_length=1, max_length=4000)
    customer_name: str | None = None


class AnalyzeRequest(BaseModel):
    reviews: list[ReviewIn] = Field(min_length=1, max_length=20)


class DraftRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    customer_name: str | None = None
    sentiment: str | None = Field(default=None, pattern="^(positive|neutral|negative)$")
    topic: str | None = None
    issue: str | None = None
    run_judge: bool = True


class TrendEvent(BaseModel):
    cluster: str
    date: dt.date


class TrendsRequest(BaseModel):
    events: list[TrendEvent] = Field(min_length=1)
    z_thresh: float = 3.5
    min_count: int = 8


def classify(client, reviews: list[dict]) -> list[dict]:
    """Sentiment (from the estimated star rating), topic, confidence for up to 10 reviews per call."""
    system = (ROOT / "prompts" / f"{CLASSIFY_PROMPT}.txt").read_text(encoding="utf-8")
    out = []
    for i in range(0, len(reviews), 10):
        chunk = reviews[i:i + 10]
        payload = [{"id": r["id"], "text": r["text"]} for r in chunk]
        resp = client.messages.create(model=GEN_MODEL, max_tokens=2000, system=system,
                                      messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
        by_id = {c["id"]: c for c in parse_json(text_of(resp))["classifications"]}
        for r in chunk:
            c = by_id.get(r["id"])
            if c is None:
                out.append({"id": r["id"], "error": "not classified"})
                continue
            topic = c["topic"] if c["topic"] in TOPICS else "other"
            out.append({"id": r["id"], "sentiment": stars_to_label(float(c["stars"])), "stars_estimate": c["stars"],
                        "topic": topic, "confidence": c["confidence"], "model_label": c["sentiment"]})
    return out


def issue_phrase(client, text: str) -> str | None:
    system = (ROOT / "prompts" / "issue_v1.txt").read_text(encoding="utf-8")
    resp = client.messages.create(model=GEN_MODEL, max_tokens=300, system=system,
                                  messages=[{"role": "user", "content": json.dumps([{"id": "1", "text": text[:700]}], ensure_ascii=False)}])
    items = parse_json(text_of(resp)).get("issues", [])
    return items[0]["issue"] if items else None


@app.get("/health")
def health():
    return {"status": "ok", "spent_usd": round(tracking.spent_usd(), 4), "cap_usd": tracking.cap_usd(),
            "model": GEN_MODEL, "judge_model": JUDGE_MODEL, "response_prompt": gen.SYSTEM_VERSION}


@app.post("/analyze", dependencies=[Depends(require_key)])
def analyze(req: AnalyzeRequest):
    client = tracking.TrackedClient(anthropic_client(), "analyze", CLASSIFY_PROMPT)
    try:
        results = classify(client, [r.model_dump() for r in req.reviews])
    except tracking.BudgetExceeded as e:
        raise HTTPException(status_code=429, detail=str(e))
    return {"results": results}


@app.post("/draft-response", dependencies=[Depends(require_key)])
def draft_response(req: DraftRequest):
    raw = anthropic_client()
    try:
        sentiment, topic, issue = req.sentiment, req.topic, req.issue
        if sentiment is None or topic is None:
            c = classify(tracking.TrackedClient(raw, "analyze", CLASSIFY_PROMPT), [{"id": "1", "text": req.text}])[0]
            if "error" in c:
                raise HTTPException(status_code=502, detail="could not classify the review")
            sentiment, topic = sentiment or c["sentiment"], topic or c["topic"]
        if sentiment == "negative" and not issue:
            issue = issue_phrase(tracking.TrackedClient(raw, "issue_phrase", "issue_v1"), req.text)
        review = {"review_id": "api", "review_text": req.text, "customer_name": req.customer_name,
                  "sentiment": sentiment, "topic": topic, "issue": issue}
        reply, _, _ = gen.draft(tracking.TrackedClient(raw, "draft_response", gen.SYSTEM_VERSION), GEN_MODEL, review, assets())
        result = {"reply": reply, "sentiment": sentiment, "topic": topic, "issue": issue, "prompt_version": gen.SYSTEM_VERSION,
                  "judge": None, "status": "pending_human_review"}
        if req.run_judge:
            scores, _, _ = jd.judge(tracking.TrackedClient(raw, "judge", jd.JUDGE_VERSION), JUDGE_MODEL, review, reply, jd.load_prompt())
            ok = jd.acceptable(scores)
            result["judge"] = {**scores, "acceptable": ok}
            if sentiment != "negative" and ok and not scores["needs_human_attention"]:
                result["status"] = "auto_approved"  # negatives always wait for a human
        return result
    except tracking.BudgetExceeded as e:
        raise HTTPException(status_code=429, detail=str(e))


@app.post("/trends", dependencies=[Depends(require_key)])
def trends(req: TrendsRequest):
    df = pd.DataFrame([{"cluster": e.cluster, "date": pd.Timestamp(e.date)} for e in req.events])
    week = ((df["date"] - df["date"].min()).dt.days // 7).astype(int)
    n_weeks = int(week.max()) + 1
    if n_weeks < 5:
        raise HTTPException(status_code=422, detail=f"need at least 5 weeks of data (4-week baseline + current); got {n_weeks}")
    counts = tr.weekly_counts(df, week, "cluster", n_weeks=n_weeks)
    flags = tr.detect_spikes(counts, z_thresh=req.z_thresh, min_count=req.min_count)
    start = df["date"].min()
    spikes = flags[flags.flag_z].copy()
    spikes["week_start"] = spikes["week"].map(lambda w: str((start + pd.Timedelta(days=7 * int(w))).date()))
    last = flags[flags.week == n_weeks - 1].sort_values("z", ascending=False)
    cols = ["group", "week", "week_start", "count", "prev_week", "wow_delta", "baseline_mean", "z", "flag_z", "flag_ewma"]
    last = last.assign(week_start=last["week"].map(lambda w: str((start + pd.Timedelta(days=7 * int(w))).date())))
    return {"n_weeks": n_weeks,
            "spikes": spikes[cols].to_dict(orient="records"),
            "latest_week": last[cols].to_dict(orient="records")}
