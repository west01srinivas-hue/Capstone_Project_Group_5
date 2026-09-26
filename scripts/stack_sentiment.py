"""Stacker: combine Claude's sentiment/confidence/star estimate with the TF-IDF baseline's probabilities.

Training data = the 600-review MARC *train*-split sample (never used for the baseline, never in test).
Settings are chosen by 5-fold cross-validation on that sample only; the 960-review TEST set is scored once.
Run after: PROMPT=classify_v3 python scripts/eval_sentiment.py claude --split train / --split test
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_sentiment as ev  # noqa: E402

PROMPT = "classify_v3"
MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5")
CLASSES = ["negative", "neutral", "positive"]


def with_claude(df: pd.DataFrame, cache: dict) -> pd.DataFrame:
    rec = df.review_id.map(lambda r: cache.get(f"{PROMPT}|{MODEL}|{r}"))
    df = df[rec.notna()].copy()
    rec = rec[rec.notna()]
    df["pred"] = rec.map(lambda r: r["sentiment"]).values
    df["conf"] = rec.map(lambda r: float(r["confidence"])).values
    df["stars_pred"] = rec.map(lambda r: float(r["stars"])).values
    return df


def features(df: pd.DataFrame, proba: pd.DataFrame, use_baseline: bool) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    for s in CLASSES:
        x["claude_" + s] = (df["pred"] == s).astype(float)
    x["claude_conf"] = df["conf"]
    x["claude_stars"] = df["stars_pred"]
    if use_baseline:
        for s in CLASSES:
            x["base_" + s] = proba.loc[df.index, s]
    return x


def score(y_true, y_pred) -> dict:
    nn = y_true != "neutral"
    return {
        "3-way acc": round(accuracy_score(y_true, y_pred), 3),
        "macroF1": round(f1_score(y_true, y_pred, average="macro"), 3),
        "neutral recall": round(float(((y_pred == "neutral") & (y_true == "neutral")).sum() / (y_true == "neutral").sum()), 3),
        "binary acc": round(accuracy_score(y_true[nn], np.where(y_pred[nn] == "neutral", "x", y_pred[nn])), 3),
    }


def main() -> None:
    cache = ev.load_cache()
    train = with_claude(ev.load_sample("train"), cache)
    test = with_claude(ev.load_sample("test"), cache)
    print(f"stacker-train rows: {len(train)}, test rows: {len(test)}")

    base = ev.train_baseline()
    _, p_train = ev.baseline_predict(base, train)
    b_test_labels, p_test = ev.baseline_predict(base, test)

    rows = []
    def add(name, y, pred, split):
        rows.append({"model": name, "split": split, **score(y.values, np.asarray(pred))})

    # reference points
    from_stars = lambda d: d["stars_pred"].round().clip(1, 5).astype(int).map(ev.stars_to_label).values
    add("baseline TF-IDF", test.label, b_test_labels[test.index], "test")
    add("Claude v3 label", test.label, test.pred.values, "test")
    add("Claude v3 star estimate -> label", test.label, from_stars(test), "test")

    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    chosen = None
    for use_base in (False, True):
        for cw in (None, "balanced"):
            name = f"stacker: Claude features{' + baseline' if use_base else ''}{', balanced' if cw else ''}"
            pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, class_weight=cw))
            Xtr = features(train, p_train, use_base)
            cv_pred = cross_val_predict(pipe, Xtr, train.label, cv=skf)
            add(name, train.label, cv_pred, "cv (train)")
            pipe.fit(Xtr, train.label)
            pred_test = pipe.predict(features(test, p_test, use_base))
            add(name, test.label, pred_test, "test")
            cvf1 = f1_score(train.label, cv_pred, average="macro")
            if chosen is None or cvf1 > chosen[0]:
                chosen = (cvf1, name, pred_test)

    out = pd.DataFrame(rows)
    print("\n" + out.to_string(index=False))
    print(f"\nselected by cross-validated macro-F1 on the training sample: {chosen[1]}")
    print("\nselected model, test confusion (rows = truth):")
    print(pd.crosstab(test.label, pd.Series(chosen[2], index=test.index, name="pred")))
    out.to_csv(ev.ROOT / "data" / "processed" / "stacker_results.csv", index=False)


if __name__ == "__main__":
    main()
