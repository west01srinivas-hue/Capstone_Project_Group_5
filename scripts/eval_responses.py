"""Stage 5: draft replies for a stratified set of reviews, score them with an LLM judge, build the human review queue.

  python scripts/eval_responses.py            # generate (Haiku) + judge (Sonnet), prints report; results are cached
  python scripts/eval_responses.py --n 2      # reviews per language and sentiment group (default 2 -> about 48 reviews)

Outputs: data/processed/response_eval_<version>.csv and data/processed/review_queue_<version>.csv (with empty
human_accept / human_comment columns for the team to fill; score them with scripts/score_human_review.py).
"""
import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
load_dotenv(ROOT / ".env")
import eval_sentiment as ev  # noqa: E402
from stack_sentiment import with_claude  # noqa: E402
from src.response import generate as gen  # noqa: E402
from src.response import judge as jd  # noqa: E402

PROC = ROOT / "data" / "processed"
GEN_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-sonnet-5")
PRICES = {"claude-haiku-4-5": (1.0, 5.0), "claude-sonnet-5": (2.0, 10.0)}  # USD per million tokens (in, out)
NAMES = {  # synthetic first names so personalisation can be tested (MARC has no reviewer names)
    "en": ["Emma", "Daniel", "Priya", "Marcus"], "de": ["Anna", "Lukas", "Sabine", "Jonas"],
    "es": ["Lucia", "Carlos", "Marta", "Diego"], "fr": ["Camille", "Louis", "Claire", "Hugo"],
    "ja": ["田中", "佐藤", "鈴木", "伊藤"], "zh": ["王女士", "李先生", "张女士", "陈先生"],
}


def cost(model: str, tin: int, tout: int) -> float:
    pin, pout = PRICES.get(model, (1.0, 5.0))
    return tin / 1e6 * pin + tout / 1e6 * pout


def load_cache(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            out[r["key"]] = r
    return out


def build_eval_set(n: int) -> pd.DataFrame:
    cache = ev.load_cache()
    df = with_claude(ev.load_sample("test"), cache)
    df["topic"] = df.review_id.map(lambda r: cache[f"classify_v3|{GEN_MODEL}|{r}"]["topic"])
    df["sentiment"] = df.pred
    issues = {}
    issue_file = PROC / "issue_phrases.jsonl"
    if issue_file.exists():
        for line in issue_file.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            issues[r["id"]] = r["issue"]
    df["issue"] = df.review_id.map(issues)
    parts = []
    for lang in ev.LANGS:
        for sent, k in (("negative", 2 * n), ("neutral", n), ("positive", n)):
            pool = df[(df.detected_language == lang) & (df.sentiment == sent)]
            parts.append(pool.sample(min(k, len(pool)), random_state=0))
    out = pd.concat(parts).reset_index(drop=True)
    out["customer_name"] = [NAMES[l][i % 4] for i, l in enumerate(out.detected_language)]
    return out


def run(fn, items, workers=4):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fn, it) for it in items]
        out = []
        for f in futs:
            try:
                out.append(f.result())
            except Exception as e:
                print("call failed:", str(e)[:120])
                out.append(None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    args = ap.parse_args()

    import anthropic

    client = anthropic.Anthropic()
    assets = gen.load_assets()
    judge_system = jd.load_prompt()
    df = build_eval_set(args.n)
    print(f"{len(df)} reviews: {df.sentiment.value_counts().to_dict()}; languages: {df.detected_language.value_counts().to_dict()}")

    gcache_path, jcache_path = PROC / "response_cache.jsonl", PROC / "judge_cache.jsonl"
    gcache, jcache = load_cache(gcache_path), load_cache(jcache_path)
    gkey = lambda r: f"{gen.SYSTEM_VERSION}|{GEN_MODEL}|{r['review_id']}|{r['customer_name']}"
    spent = {"gen": 0.0, "judge": 0.0}

    todo = [r for r in df.to_dict("records") if gkey(r) not in gcache]
    print(f"drafting {len(todo)} replies with {GEN_MODEL} ({len(df) - len(todo)} cached)")

    def do_gen(r):
        reply, a, b = gen.draft(client, GEN_MODEL, r, assets)
        return {"key": gkey(r), "reply": reply, "tin": a, "tout": b}

    with gcache_path.open("a", encoding="utf-8") as f:
        for rec in run(do_gen, todo):
            if rec:
                gcache[rec["key"]] = rec
                spent["gen"] += cost(GEN_MODEL, rec["tin"], rec["tout"])
                f.write(json.dumps(rec, ensure_ascii=False) + chr(10))
    df["reply"] = df.apply(lambda r: gcache.get(gkey(r), {}).get("reply"), axis=1)
    df = df[df.reply.notna()].reset_index(drop=True)

    jkey = lambda r: f"{jd.JUDGE_VERSION}n|{JUDGE_MODEL}|{r['review_id']}|{hashlib.md5(r['reply'].encode('utf-8')).hexdigest()[:10]}"
    jtodo = [r for r in df.to_dict("records") if jkey(r) not in jcache]
    print(f"judging {len(jtodo)} replies with {JUDGE_MODEL}")

    def do_judge(r):
        scores, a, b = jd.judge(client, JUDGE_MODEL, r, r["reply"], judge_system)
        return {"key": jkey(r), **scores, "tin": a, "tout": b}

    with jcache_path.open("a", encoding="utf-8") as f:
        for rec in run(do_judge, jtodo):
            if rec:
                jcache[rec["key"]] = rec
                spent["judge"] += cost(JUDGE_MODEL, rec["tin"], rec["tout"])
                f.write(json.dumps(rec, ensure_ascii=False) + chr(10))
    print(f"spent this run: drafting ${spent['gen']:.4f}, judging ${spent['judge']:.4f}")

    for col in jd.DIMENSIONS + jd.FLAGS + ["reason"]:
        df[col] = df.apply(lambda r: jcache.get(jkey(r), {}).get(col), axis=1)
    df = df[df.empathy.notna()].reset_index(drop=True)
    df["acceptable"] = df.apply(lambda r: jd.acceptable(r), axis=1)

    print(f"\n== judge scores (1-5), {len(df)} drafts ==")
    print(df[jd.DIMENSIONS].astype(float).mean().round(2).to_dict())
    print("guardrail flags:", {f: int(df[f].sum()) for f in jd.FLAGS})
    print(f"judge-acceptable (all scores >= 4, no promise / invented fact / wrong language): {df.acceptable.mean():.1%}")
    print("\nby sentiment:")
    print(df.groupby("sentiment")[jd.DIMENSIONS + ["acceptable"]].mean().round(2).to_string())
    print("\nby language:")
    print(df.groupby("detected_language")[jd.DIMENSIONS + ["acceptable"]].mean().round(2).to_string())
    bad = df[~df.acceptable].head(6)
    if len(bad):
        print("\nnot acceptable (first 6):")
        for r in bad.itertuples():
            print(f"[{r.detected_language} {r.sentiment}/{r.topic}] scores={[r.empathy, r.specificity, r.correctness, r.tone]} "
                  f"flags={[f for f in jd.FLAGS if getattr(r, f)]} reason={r.reason}")
            print("   review:", r.review_text[:110].replace(chr(10), " "))
            print("   reply :", r.reply[:160].replace(chr(10), " "))

    tag = gen.SYSTEM_VERSION
    df.to_csv(PROC / f"response_eval_{tag}.csv", index=False, encoding="utf-8")
    q = df.copy()
    q["status"] = q.apply(lambda r: "pending_human_review" if (r.sentiment == "negative" or r.needs_human_attention
                                                                or not r.acceptable) else "auto_approved", axis=1)
    q["human_accept"] = ""
    q["human_comment"] = ""
    q[["review_id", "detected_language", "sentiment", "topic", "review_text", "reply", "status"] + jd.DIMENSIONS
      + jd.FLAGS + ["acceptable", "human_accept", "human_comment"]].to_csv(
        PROC / f"review_queue_{tag}.csv", index=False, encoding="utf-8-sig")  # utf-8-sig so Excel shows Japanese/Chinese
    print("\nqueue:", q.status.value_counts().to_dict(), "->", PROC / f"review_queue_{tag}.csv")


if __name__ == "__main__":
    main()
