"""Stage 4: issue clustering + trend detection on the reviews Claude classified as negative.

  python scripts/run_stage4.py issues             # ~ $0.10: Claude writes a short English issue phrase per review (cached)
  python scripts/run_stage4.py cluster [native|issue]   # free: embeddings + k-means / HDBSCAN, prints summaries
  python scripts/run_stage4.py label              # ~ $0.01: Claude names each cluster (cached)
  python scripts/run_stage4.py trends             # free: planted-spike test over many random timelines
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
import eval_sentiment as ev  # noqa: E402
from stack_sentiment import with_claude  # noqa: E402
from src.analysis import clustering as cl  # noqa: E402
from src.analysis import trends as tr  # noqa: E402

PROC = ROOT / "data" / "processed"
BASIS = sys.argv[2] if len(sys.argv) > 2 else "issue"  # what text is embedded: native review text or Claude's issue phrase
EMB = PROC / f"neg_embeddings_{BASIS}.npy"
CLUSTERS = PROC / f"neg_clusters_{BASIS}.csv"
LABELS = PROC / f"cluster_labels_{BASIS}.json"
ISSUES = PROC / "issue_phrases.jsonl"
TAXONOMY = ["shipping", "quality", "support", "features", "price", "ux", "other"]


def load_negatives() -> pd.DataFrame:
    cache = ev.load_cache()
    frames = [with_claude(ev.load_sample(s), cache) for s in ("test", "train")]
    df = pd.concat(frames, ignore_index=True)
    df["topic"] = df.review_id.map(lambda r: cache[f"classify_v3|{os.getenv('CLAUDE_MODEL', 'claude-haiku-4-5')}|{r}"]["topic"])
    df.loc[~df.topic.isin(TAXONOMY), "topic"] = "other"
    return df[df.pred == "negative"].reset_index(drop=True)


def do_issues() -> None:
    import anthropic
    from concurrent.futures import ThreadPoolExecutor

    df = load_negatives()
    have = {json.loads(l)["id"] for l in ISSUES.read_text(encoding="utf-8").splitlines()} if ISSUES.exists() else set()
    todo = df[~df.review_id.isin(have)]
    model = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5")
    system = (ROOT / "prompts" / "issue_v1.txt").read_text(encoding="utf-8")
    chunks = [todo.iloc[i:i + 10] for i in range(0, len(todo), 10)]
    print(f"{len(todo)} reviews without an issue phrase -> {len(chunks)} calls, est. cost ${len(todo) * 0.00017:.3f}")
    client = anthropic.Anthropic()

    def one(chunk):
        payload = [{"id": r.review_id, "text": r.review_text[:700]} for r in chunk.itertuples()]
        r = client.messages.create(model=model, max_tokens=1500, system=system,
                                   messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", r.content[0].text.strip())
        return json.loads(raw)["issues"], r.usage.input_tokens, r.usage.output_tokens

    tin = tout = 0
    PROC.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as ex, ISSUES.open("a", encoding="utf-8") as f:
        for fut in [ex.submit(one, c) for c in chunks]:
            try:
                items, a, b = fut.result()
            except Exception as e:
                print("batch failed:", str(e)[:120])
                continue
            tin, tout = tin + a, tout + b
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + chr(10))
    print(f"tokens in/out={tin}/{tout}, cost ${tin / 1e6 + tout / 1e6 * 5:.4f}")


def load_issue_phrases(df: pd.DataFrame) -> pd.Series:
    m = {}
    for l in ISSUES.read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        m[r["id"]] = r["issue"]
    return df.review_id.map(m)


def do_cluster() -> None:
    df = load_negatives()
    if BASIS == "issue":
        df["issue"] = load_issue_phrases(df)
        df = df[df.issue.notna()].reset_index(drop=True)
    embed_text = (df.issue if BASIS == "issue" else df.review_text).tolist()
    print(f"basis = {BASIS}")
    print(f"{len(df)} reviews classified negative (from {df.detected_language.nunique()} languages)")
    if EMB.exists() and len(np.load(EMB)) == len(df):
        emb = np.load(EMB)
    else:
        emb = cl.embed(embed_text)
        PROC.mkdir(parents=True, exist_ok=True)
        np.save(EMB, emb)

    km_labels, table = cl.kmeans_search(emb)
    print("\nk-means silhouette by k:\n" + table.to_string(index=False))
    df["cluster"] = km_labels
    hd = cl.hdbscan_labels(emb)
    df["hdbscan"] = hd
    n_hd = len(set(hd) - {-1})
    print(f"\nHDBSCAN: {n_hd} clusters, {(hd == -1).mean():.0%} of reviews left as noise")
    print("\nk-means clusters (dominant Claude topic and purity):")
    print(cl.summarize(df, "cluster").to_string(index=False))
    df.to_csv(CLUSTERS, index=False, encoding="utf-8")
    print("\nsaved", CLUSTERS)


def do_label() -> None:
    import anthropic

    df = pd.read_csv(CLUSTERS)
    payload = []
    for c, g in df.groupby("cluster"):
        sample = g.sample(min(8, len(g)), random_state=0)
        payload.append({"id": int(c), "size": len(g), "examples": [t[:180] for t in sample.review_text]})
    system = (
        "You name groups of customer complaints. Each group holds negative product reviews in several languages that "
        "were clustered by meaning. For every group give a short English label (3 to 6 words) that describes the shared "
        f"problem, and the closest canonical topic from: {', '.join(TAXONOMY)}. "
        'Respond with JSON only: {"clusters":[{"id":<int>,"label":"<label>","topic":"<topic>"}]}'
    )
    model = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5")
    r = anthropic.Anthropic().messages.create(model=model, max_tokens=2000, system=system,
                                              messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
    out = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", r.content[0].text.strip()))["clusters"]
    LABELS.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    cost = r.usage.input_tokens / 1e6 * 1 + r.usage.output_tokens / 1e6 * 5
    print(f"labelled {len(out)} clusters, tokens in/out={r.usage.input_tokens}/{r.usage.output_tokens}, cost ${cost:.4f}")
    lab = {o["id"]: o for o in out}
    summ = cl.summarize(df, "cluster")
    summ["label"] = summ.cluster.map(lambda c: lab[c]["label"])
    summ["llm_topic"] = summ.cluster.map(lambda c: lab[c]["topic"])
    print(summ[["cluster", "size", "label", "llm_topic", "dominant_topic", "topic_purity"]].to_string(index=False))


def do_trends(n_timelines: int = 200) -> None:
    df = pd.read_csv(CLUSTERS)
    sizes = df.cluster.value_counts()
    # plant spikes in business-style clusters (shipping delay, damaged packaging, price), later weeks so a
    # 4-week baseline exists; fall back to the largest clusters if the labels are not available
    labels = {o["id"]: o["label"] for o in json.loads(LABELS.read_text(encoding="utf-8"))} if LABELS.exists() else {}
    picks = []
    for kw in ("Delayed", "Damaged packaging", "Overpriced"):
        hit = [c for c, lab in labels.items() if kw.lower() in lab.lower()]
        if hit:
            picks.append(hit[0])
    if len(picks) < 3:
        picks = list(sizes.index[:3])
    print("spike clusters:", {int(c): labels.get(int(c), "?") for c in picks})
    spikes = [{"cluster": int(picks[0]), "week": 7, "factor": 5.0},
              {"cluster": int(picks[1]), "week": 9, "factor": 4.0},
              {"cluster": int(picks[2]), "week": 10, "factor": 3.0}]
    print("planted spikes:", spikes, "| cluster sizes:", sizes.to_dict())

    res = {"flag_z": [], "flag_ewma": []}
    quiet_fp = []
    for seed in range(n_timelines):
        week = tr.assign_timeline(df, "cluster", spikes, seed=seed)
        counts = tr.weekly_counts(df, week, "cluster")
        flags = tr.detect_spikes(counts)
        for rule in res:
            res[rule].append(tr.evaluate(flags, spikes, rule))
        # false alarms on a timeline with NO planted spike
        quiet = tr.detect_spikes(tr.weekly_counts(df, tr.assign_timeline(df, "cluster", [], seed=1000 + seed), "cluster"))
        quiet_fp.append(int(quiet.flag_z.sum()))
    print(f"\nover {n_timelines} random timelines (3 planted spikes each):")
    rows = []
    for rule, lst in res.items():
        d = pd.DataFrame(lst)
        rows.append({"rule": rule, "precision": round(d.precision.mean(), 3), "recall": round(d.recall.mean(), 3),
                     "precision@3": round(d.precision_at_k.mean(), 3), "false alarms / timeline": round(d.fp.mean(), 2)})
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"z-score false alarms on timelines with no planted spike: {np.mean(quiet_fp):.2f} per 12 weeks")

    week = tr.assign_timeline(df, "cluster", spikes, seed=0)
    flags = tr.detect_spikes(tr.weekly_counts(df, week, "cluster"))
    print("\nexample timeline (seed 0), flagged by z-score:")
    print(flags[flags.flag_z].to_string(index=False))
    df["week"] = week
    df.to_csv(PROC / "neg_clusters_timeline_seed0.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    {"issues": do_issues, "cluster": do_cluster, "label": do_label, "trends": do_trends}[sys.argv[1]]()
