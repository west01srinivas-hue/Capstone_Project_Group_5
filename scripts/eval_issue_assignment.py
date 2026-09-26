"""Check prompts/issue_cluster_v1.txt: does Claude put negative reviews into the same issue category the
embedding + k-means clustering found? (n8n Cloud cannot run embeddings, so the n8n workflow assigns categories with Claude.)

  python scripts/eval_issue_assignment.py [n]      # default 100 reviews, about $0.03
"""
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
import anthropic  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
df = pd.read_csv(ROOT / "data" / "processed" / "neg_clusters_issue.csv").sample(n, random_state=1).reset_index(drop=True)
system = (ROOT / "prompts" / "issue_cluster_v1.txt").read_text(encoding="utf-8")
tax = json.loads((ROOT / "prompts" / "issue_taxonomy_v1.json").read_text(encoding="utf-8"))
model = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5")
client = anthropic.Anthropic()

rows, tin, tout = [], 0, 0
for i in range(0, len(df), 10):
    chunk = df.iloc[i:i + 10]
    payload = [{"id": r.review_id, "text": r.review_text[:700]} for r in chunk.itertuples()]
    resp = client.messages.create(model=model, max_tokens=1500, system=system,
                                  messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
    tin, tout = tin + resp.usage.input_tokens, tout + resp.usage.output_tokens
    text = next(b.text for b in resp.content if b.type == "text")
    got = {a["id"]: a for a in json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip()))["assignments"]}
    for r in chunk.itertuples():
        a = got.get(r.review_id)
        rows.append({"review_id": r.review_id, "kmeans": r.cluster, "llm": a["category"] if a else None,
                     "issue": a["issue"] if a else None, "text": r.review_text})
out = pd.DataFrame(rows)
ok = out.dropna(subset=["llm"])
print(f"{len(ok)} of {n} assigned; exact agreement with the k-means cluster: {(ok.kmeans == ok.llm).mean():.1%}; "
      f"'none fits' (-1): {(ok.llm == -1).mean():.1%}")
print("spread of Claude's categories:", dict(Counter(ok.llm.astype(int)).most_common()))
print(f"cost ${tin / 1e6 + tout / 1e6 * 5:.4f}")
print("\nexamples where they differ:")
for r in ok[ok.kmeans != ok.llm].head(6).itertuples():
    print(f"  kmeans={tax[str(r.kmeans)][:34]!r:38} claude={tax.get(str(int(r.llm)), 'none')[:34]!r:38} issue={r.issue!r}")
    print(f"     {r.text[:110]!r}")
