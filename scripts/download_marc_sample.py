"""Download a stratified sample of the MARC multilingual reviews from the Hugging Face mirror
and save it in the project's unified review schema. No API keys required.

Notes on the mirror (mteb/amazon_reviews_multi):
- files are sorted by label, so we sample per star rating instead of taking the first N rows
- label is 0-4 (= 1-5 stars); title and body are already merged into `text`
- no product category or review date is included
"""
import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset

LANGS = ["en", "de", "es", "fr", "ja", "zh"]
OUT = Path(__file__).resolve().parents[1] / "data" / "samples"


def load_lang(lang: str, split: str) -> pd.DataFrame:
    ds = load_dataset(
        "json",
        data_files=f"hf://datasets/mteb/amazon_reviews_multi/{lang}/{split}.jsonl",
        split="train",
    )
    return ds.to_pandas()


def to_unified(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_id": "marc-" + df["id"].astype(str),
            "source": "marc",
            "product_category": None,
            "rating": df["label"].astype(int) + 1,
            "review_text": df["text"].str.strip(),
            "detected_language": lang,
            "translated_text": None,
            "review_date": None,
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-lang", type=int, default=200, help="rows per language (split evenly over 5 stars)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    per_star = max(1, args.per_lang // 5)
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for lang in LANGS:
        raw = load_lang(lang, args.split)
        sample = raw.groupby("label").sample(n=per_star, random_state=args.seed).sample(
            frac=1, random_state=args.seed
        )
        frames.append(to_unified(sample, lang))
        print(f"{lang}: sampled {len(sample)} of {len(raw)}")

    df = pd.concat(frames, ignore_index=True)
    path = OUT / "marc_sample.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"\nSaved {len(df)} rows -> {path}")
    print(df.pivot_table(index="detected_language", columns="rating", values="review_id", aggfunc="count"))
    print("\nmean review length (chars) by language:")
    print(df.assign(n=df["review_text"].str.len()).groupby("detected_language")["n"].mean().round(1))


if __name__ == "__main__":
    main()
