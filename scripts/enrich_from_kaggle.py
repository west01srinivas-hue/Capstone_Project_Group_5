"""Fill product_category / product_id in the sample CSVs from the Kaggle copy of MARC and verify the match.

The samples were first drawn from a Hugging Face copy of the same corpus. Kaggle's files carry the same
review ids, stars and text, plus product_category. Rows are joined by review id (unchanged, so cached Claude
results stay valid). Any mismatch in stars or text aborts before anything is written.

Set KAGGLE_MARC_DIR in .env to the folder holding Kaggle's test.csv / train.csv / validation.csv.
"""
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
KAGGLE_DIR = Path(os.environ["KAGGLE_MARC_DIR"])
COLS = ["review_id", "product_id", "product_category", "stars", "review_title", "review_body"]

JOBS = [("marc_sample.csv", "test.csv"), ("marc_train_sample.csv", "train.csv")]


def kaggle_text(k: pd.DataFrame) -> pd.Series:
    return (k.review_title.fillna("").str.strip() + "\n\n" + k.review_body.fillna("").str.strip()).str.strip()


def main() -> None:
    for sample_name, kaggle_name in JOBS:
        path = ROOT / "data" / "samples" / sample_name
        s = pd.read_csv(path)
        ids = s.review_id.str.replace("marc-", "", n=1, regex=False)
        k = pd.read_csv(KAGGLE_DIR / kaggle_name, usecols=COLS)
        k = k[k.review_id.isin(ids)].drop_duplicates("review_id").set_index("review_id")
        missing = set(ids) - set(k.index)
        if missing:
            sys.exit(f"{sample_name}: {len(missing)} ids not found in Kaggle {kaggle_name}")
        k = k.loc[ids.values]
        if not (k.stars.values == s.rating.values).all():
            sys.exit(f"{sample_name}: star ratings differ from Kaggle")
        if not (kaggle_text(k).values == s.review_text.str.strip().values).all():
            sys.exit(f"{sample_name}: review text differs from Kaggle")
        s["product_id"] = k.product_id.values
        s["product_category"] = k.product_category.values
        s["source"] = "kaggle-marc"
        s.to_csv(path, index=False, encoding="utf-8")
        print(f"{sample_name}: {len(s)} rows verified against Kaggle {kaggle_name}; "
              f"{s.product_category.nunique()} categories, top: {s.product_category.value_counts().head(4).to_dict()}")


if __name__ == "__main__":
    main()
