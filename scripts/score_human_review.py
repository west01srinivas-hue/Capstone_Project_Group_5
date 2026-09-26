"""Score the human review sheet.

Open data/processed/review_queue_response_v2.csv in Excel (or Google Sheets), fill the `human_accept` column
with Y (would send as is or with a trivial edit) or N (would not send), optionally add a `human_comment`, save as CSV,
then run:  python scripts/score_human_review.py [path]
Target from the proposal: at least 80% human acceptance.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "processed" / "review_queue_response_v2.csv"
df = pd.read_csv(path, encoding="utf-8-sig")
df["human_accept"] = df["human_accept"].astype(str).str.strip().str.upper().str[:1]
done = df[df.human_accept.isin(["Y", "N"])].copy()
if done.empty:
    sys.exit("no rows have human_accept = Y or N yet")
done["accepted"] = done.human_accept == "Y"
print(f"{len(done)} of {len(df)} drafts reviewed")
print(f"human acceptance: {done.accepted.mean():.1%}  (target 80%)")
print("\nby sentiment:\n" + done.groupby("sentiment").accepted.agg(["mean", "count"]).round(3).to_string())
print("\nby language:\n" + done.groupby("detected_language").accepted.agg(["mean", "count"]).round(3).to_string())
agree = (done.accepted == done.acceptable.astype(bool)).mean()
print(f"\nagreement with the LLM judge's acceptable/not-acceptable call: {agree:.1%}")
print("confusion (rows = human, columns = judge):")
print(pd.crosstab(done.accepted, done.acceptable.astype(bool)))
