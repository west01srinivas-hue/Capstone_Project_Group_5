"""Evaluate sentiment classification on the MARC sample.

Ground truth comes from star ratings: 1-2 -> negative, 3 -> neutral, 4-5 -> positive.
The sample is split per language and star into a DEV part (tune prompts here) and a TEST part
(final reported numbers). Claude results are cached on disk, so a review is never paid for twice.

usage:
  python scripts/eval_sentiment.py baseline            # free TF-IDF + logistic regression
  python scripts/eval_sentiment.py claude --split dev   # costs money, prints estimated cost first
  python scripts/eval_sentiment.py claude --split test --yes
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
LANGS = ["en", "de", "es", "fr", "ja", "zh"]
PROMPT_NAME = os.getenv("PROMPT", "classify_v1")
BATCH = 10
PRICE_IN, PRICE_OUT = 1.00, 5.00  # Claude Haiku 4.5 list price per million tokens
CACHE = ROOT / "data" / "processed" / "claude_sentiment_cache.jsonl"


def stars_to_label(r: int) -> str:
    return "negative" if r <= 2 else "neutral" if r == 3 else "positive"


def load_sample(split: str) -> pd.DataFrame:
    if split == "train":  # separate sample from the MARC train split, used to fit the stacker
        df = pd.read_csv(ROOT / "data" / "samples" / "marc_train_sample.csv")
        df["label"] = df["rating"].map(stars_to_label)
        df["split"] = "train"
        return df
    df = pd.read_csv(ROOT / "data" / "samples" / "marc_sample.csv")
    df["label"] = df["rating"].map(stars_to_label)
    # deterministic dev/test split: within each language+star, the first 8 are dev, the rest test
    df["_pos"] = df.groupby(["detected_language", "rating"]).cumcount()
    df["split"] = (df["_pos"] < 8).map({True: "dev", False: "test"})
    return df[df["split"] == split].drop(columns=["_pos"]).reset_index(drop=True)


def report(df: pd.DataFrame, pred_col: str, title: str) -> None:
    rows = []
    for lang in LANGS + ["ALL"]:
        s = df if lang == "ALL" else df[df.detected_language == lang]
        if s.empty:
            continue
        binary = s[s.label != "neutral"]
        rows.append({
            "lang": lang, "n": len(s),
            "3-way acc": round(accuracy_score(s.label, s[pred_col]), 3),
            "3-way macroF1": round(f1_score(s.label, s[pred_col], average="macro"), 3),
            "binary acc": round(accuracy_score(binary.label, binary[pred_col].where(binary[pred_col] != "neutral", "x")), 3),
        })
    print(f"\n== {title} ==")
    print(pd.DataFrame(rows).to_string(index=False))


def train_baseline() -> dict:
    """Per-language TF-IDF (character n-grams, so CJK works) + logistic regression, trained on the
    30,000-review MARC validation split. Returns {lang: (vectorizer, classifier)}."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from datasets import load_dataset

    models = {}
    for lang in LANGS:
        d = load_dataset("json", data_files=f"hf://datasets/mteb/amazon_reviews_multi/{lang}/validation.jsonl",
                         split="train").to_pandas()
        y = (d["label"].astype(int) + 1).map(stars_to_label)
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), min_df=2, sublinear_tf=True, max_features=200000)
        models[lang] = (vec, LogisticRegression(max_iter=1000, C=3.0).fit(vec.fit_transform(d["text"]), y))
    return models


def baseline_predict(models: dict, df: pd.DataFrame):
    """Returns (labels, probabilities with columns negative/neutral/positive), aligned to df.index."""
    proba = pd.DataFrame(index=df.index, columns=["negative", "neutral", "positive"], dtype=float)
    labels = pd.Series(index=df.index, dtype=object)
    for lang, (vec, clf) in models.items():
        idx = df.index[df.detected_language == lang]
        if len(idx) == 0:
            continue
        x = vec.transform(df.loc[idx, "review_text"])
        proba.loc[idx, list(clf.classes_)] = clf.predict_proba(x)
        labels.loc[idx] = clf.predict(x)
    return labels, proba


def run_baseline(args) -> None:
    test = load_sample(args.split)
    test["pred"], _ = baseline_predict(train_baseline(), test)
    print(f"baseline trained on the MARC validation split; evaluating on '{args.split}' ({len(test)})")
    report(test, "pred", f"TF-IDF + logistic regression baseline ({args.split})")


def load_cache() -> dict:
    out = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            out[r["key"]] = r
    return out


def run_claude(args) -> None:
    import anthropic

    model = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5")
    system = (ROOT / "prompts" / f"{PROMPT_NAME}.txt").read_text(encoding="utf-8")
    df = load_sample(args.split)
    cache = load_cache()
    key = lambda rid: f"{PROMPT_NAME}|{model}|{rid}"
    todo = df[~df.review_id.map(lambda r: key(r) in cache)]
    n_calls = -(-len(todo) // BATCH)
    est = len(todo) * (130 * PRICE_IN + 45 * PRICE_OUT) / 1e6
    print(f"{len(df)} reviews in '{args.split}', {len(todo)} not cached -> {n_calls} calls, est. cost ${est:.3f}")
    if len(todo) and est > 0.5 and not args.yes:
        sys.exit("estimated cost above $0.50; re-run with --yes to confirm")

    client = anthropic.Anthropic()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    usage = {"in": 0, "out": 0}

    def do_batch(chunk: pd.DataFrame):
        payload = [{"id": r.review_id, "text": r.review_text} for r in chunk.itertuples()]
        resp = client.messages.create(model=model, max_tokens=2000, system=system,
                                      messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", resp.content[0].text.strip())
        items = json.loads(raw)["classifications"]
        return items, resp.usage.input_tokens, resp.usage.output_tokens

    chunks = [todo.iloc[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=4) as ex, CACHE.open("a", encoding="utf-8") as f:
        for chunk, fut in zip(chunks, [ex.submit(do_batch, c) for c in chunks]):
            try:
                items, tin, tout = fut.result()
            except Exception as e:  # a bad batch is skipped and retried on the next run
                print("batch failed:", str(e)[:120])
                continue
            usage["in"] += tin
            usage["out"] += tout
            for it in items:
                rec = {"key": key(it["id"]), **it}
                cache[rec["key"]] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    cost = usage["in"] / 1e6 * PRICE_IN + usage["out"] / 1e6 * PRICE_OUT
    print(f"done in {time.time() - t0:.0f}s, tokens in/out={usage['in']}/{usage['out']}, actual cost ${cost:.4f}")

    df["pred"] = df.review_id.map(lambda r: cache.get(key(r), {}).get("sentiment"))
    df["topic"] = df.review_id.map(lambda r: cache.get(key(r), {}).get("topic"))
    df["conf"] = df.review_id.map(lambda r: cache.get(key(r), {}).get("confidence"))
    df["stars_pred"] = df.review_id.map(lambda r: cache.get(key(r), {}).get("stars"))
    missing = df["pred"].isna().sum()
    if missing:
        print(f"WARNING: {missing} reviews have no prediction (failed batches); excluded from metrics")
        df = df.dropna(subset=["pred"])
    report(df, "pred", f"Claude {model} / {PROMPT_NAME} ({args.split})")
    if df["stars_pred"].notna().any():
        df["pred_from_stars"] = df["stars_pred"].astype(float).round().clip(1, 5).map(lambda s: stars_to_label(int(s)))
        report(df, "pred_from_stars", f"same run, label taken from the estimated star rating ({args.split})")
    print("\ntopic distribution:", df.topic.value_counts().to_dict())
    print("\nconfusion (rows = truth):")
    print(pd.crosstab(df.label, df.pred))
    df.to_csv(ROOT / "data" / "processed" / f"claude_sentiment_{args.split}.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["baseline", "claude"])
    ap.add_argument("--split", default="dev", choices=["dev", "test", "train"])
    ap.add_argument("--yes", action="store_true", help="confirm spending above $0.50")
    a = ap.parse_args()
    run_baseline(a) if a.mode == "baseline" else run_claude(a)
