"""Gradio demo: analyse a review and draft a reply, browse saved examples, run spike detection.

  python app.py                      # http://127.0.0.1:7860
Environment: ANTHROPIC_API_KEY (live tab only), MAX_SPEND_USD (default 0.5 here), DEMO_ACCESS_CODE (optional; if set,
the live tab asks for it), DEMO_MAX_CALLS (live requests per browser session, default 10).
The saved-examples and spike tabs never call Claude, so they work with no key and cost nothing.
"""
import json
import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("MAX_SPEND_USD", "0.5")  # tighter than the API default: this app can be public
os.environ.setdefault("RESPONSE_VERSION", "v2")

import gradio as gr  # noqa: E402
from fastapi import HTTPException  # noqa: E402

from src.api import main as api  # noqa: E402
from src.llmops import tracking  # noqa: E402

try:  # ZeroGPU Spaces refuse to start without a @spaces.GPU function; this app needs no GPU and never calls it
    import spaces

    @spaces.GPU
    def _gpu_placeholder():
        return None
except ImportError:  # running locally or on CPU hardware
    pass

ROOT = Path(__file__).resolve().parent
EXAMPLES = json.loads((ROOT / "data" / "samples" / "demo_examples.json").read_text(encoding="utf-8"))
REVIEWS_CSV = ROOT / "data" / "samples" / "dashboard_reviews.csv"
MAX_CALLS = int(os.getenv("DEMO_MAX_CALLS", "10"))
LANG = {"en": "English", "de": "German", "es": "Spanish", "fr": "French", "ja": "Japanese", "zh": "Chinese"}

INTRO = """# Customer Review Insights & Response Generator
Reads reviews in six languages, works out sentiment, topic and the issue behind a complaint, drafts a reply in the
customer's language, and spots sudden spikes. **Replies are drafts: negative reviews always go to a human first.**
Group 5, TalentSprint capstone. Sample reviews come from the Multilingual Amazon Reviews Corpus; review dates in the
spike tab are simulated."""


# ---------- saved examples (no Claude calls) ----------
def example_labels() -> list[str]:
    return [f"{i + 1}. [{LANG[e['language']]}] {e['sentiment']}: {e['review_text'][:60].strip()}..." for i, e in enumerate(EXAMPLES)]


def judge_markdown(j: dict, status: str) -> str:
    flags = [name for name, on in (("promised an outcome", j["promises_outcome"]), ("invented facts", j["invents_facts"]),
                                   ("wrong language", j["wrong_language"]), ("needs a human", j["needs_human_attention"])) if on]
    verdict = "accepted by the judge" if j["acceptable"] else "**not accepted by the judge**"
    return (f"**Status:** `{status}`  \n**Judge (Claude Sonnet 5):** empathy {j['empathy']}, specificity {j['specificity']}, "
            f"correctness {j['correctness']}, tone {j['tone']} (1 to 5), {verdict}.  \n"
            f"**Flags:** {', '.join(flags) if flags else 'none'}  \n_{j['reason']}_")


def show_example(label: str):
    if not label:
        return "", "", "", "", ""
    e = EXAMPLES[example_labels().index(label)]
    analysis = (f"**Sentiment:** {e['sentiment']}  \n**Topic:** {e['topic']}  \n**Estimated stars:** {e['stars_estimate']:.0f}  \n"
                f"**Issue:** {e['issue'] or '(not needed for non-negative reviews)'}  \n**Language:** {LANG[e['language']]}")
    return e["review_text"], analysis, e["reply"], judge_markdown(e["judge"], e["status"]), e.get("customer_name") or ""


# ---------- live analysis (calls Claude) ----------
def analyse_live(text: str, name: str, use_judge: bool, code: str, used: int):
    empty = ("", "", "", "")
    text = (text or "").strip()
    if not text:
        return (*empty[:3], "Paste a review first.", used)
    expected = os.getenv("DEMO_ACCESS_CODE")
    if expected and code != expected:
        return (*empty[:3], "Wrong or missing access code.", used)
    if used >= MAX_CALLS:
        return (*empty[:3], f"This session has used its {MAX_CALLS} live requests. Try the saved examples.", used)
    if not os.getenv("ANTHROPIC_API_KEY"):
        return (*empty[:3], "No ANTHROPIC_API_KEY is configured on this server. Try the saved examples.", used)
    try:
        client = tracking.TrackedClient(api.anthropic_client(), "analyze", api.CLASSIFY_PROMPT)
        c = api.classify(client, [{"id": "1", "text": text[:4000]}])[0]
        if "error" in c:
            return (*empty[:3], "Claude could not classify this review. Try again.", used + 1)
        result = api.draft_response(api.DraftRequest(text=text[:4000], customer_name=(name or "").strip() or None,
                                                     sentiment=c["sentiment"], topic=c["topic"], run_judge=use_judge))
    except (HTTPException, tracking.BudgetExceeded) as e:
        detail = getattr(e, "detail", str(e))
        return (*empty[:3], f"Stopped: {detail}", used)
    except Exception as e:  # network or API error: show it instead of a stack trace
        return (*empty[:3], f"Something went wrong ({type(e).__name__}). Try again.", used)
    analysis = (f"**Sentiment:** {result['sentiment']}  \n**Topic:** {result['topic']}  \n"
                f"**Estimated stars:** {c['stars_estimate']}  \n**Confidence:** {c['confidence']}  \n"
                f"**Issue:** {result['issue'] or '(not needed for non-negative reviews)'}")
    if result["judge"]:
        status = judge_markdown({**result["judge"], "reason": result["judge"].get("reason", "")}, result["status"])
    else:
        status = f"**Status:** `{result['status']}` (judge skipped, so the reply is held for a human)"
    return analysis, result["reply"], status, f"Done. Spend so far on this server: ${tracking.spent_usd():.4f} of ${tracking.cap_usd():.2f}.", used + 1


# ---------- spike detection (no Claude calls) ----------
def run_spikes(z_thresh: float, min_count: int):
    df = pd.read_csv(REVIEWS_CSV, parse_dates=["review_date"])
    neg = df[(df.sentiment == "negative") & df.issue_category.notna()]
    events = [api.TrendEvent(cluster=r.issue_category, date=r.review_date.date()) for r in neg.itertuples()]
    res = api.trends(api.TrendsRequest(events=events, z_thresh=float(z_thresh), min_count=int(min_count)))
    table = pd.DataFrame(res["spikes"])
    if table.empty:
        table = pd.DataFrame(columns=["issue", "week starting", "negative reviews", "previous week", "change", "baseline", "z"])
    else:
        table = table.rename(columns={"group": "issue", "week_start": "week starting", "count": "negative reviews", "prev_week": "previous week",
                                      "wow_delta": "change", "baseline_mean": "baseline", "z": "z"})[
            ["issue", "week starting", "negative reviews", "previous week", "change", "baseline", "z"]].sort_values("z", ascending=False)
    t0 = neg.review_date.min()
    weekly = (neg.assign(week=((neg.review_date - t0).dt.days // 7) + 1).groupby(["issue_category", "week"]).size()
              .rename("negative reviews").reset_index().rename(columns={"issue_category": "issue"}))
    weekly["week"] = weekly["week"].map(lambda w: f"W{int(w):02d}")  # labelled weeks instead of a numeric axis
    flagged = set(table["issue"]) if len(table) else set()
    weekly = weekly[weekly.issue.isin(flagged)] if flagged else weekly[weekly.issue.isin(weekly.issue.unique()[:4])]
    msg = f"{len(table)} spike(s) found (z >= {z_thresh}, at least {int(min_count)} reviews in the week, 4-week baseline). " \
          "Dates in this data are simulated, with three planted spikes (delayed shipments, damaged packaging, price complaints); any other alert is a false alarm."
    return msg, table, weekly


# ---------- UI ----------
def build() -> gr.Blocks:
    with gr.Blocks(title="Customer Review Insights & Response Generator") as demo:
        gr.Markdown(INTRO)
        with gr.Tab("Saved examples (no cost)"):
            gr.Markdown("Real reviews from the test set with the pipeline's stored output. No Claude call is made here.")
            pick = gr.Dropdown(choices=example_labels(), label="Choose a review", value=example_labels()[0])
            review_box = gr.Textbox(label="Review", lines=5, interactive=False)
            with gr.Row():
                analysis_box = gr.Markdown()
                reply_box = gr.Textbox(label="Drafted reply (in the customer's language)", lines=6, interactive=False)
            judge_box = gr.Markdown()
            hidden_name = gr.Textbox(visible=False)
            pick.change(show_example, pick, [review_box, analysis_box, reply_box, judge_box, hidden_name])
            demo.load(show_example, pick, [review_box, analysis_box, reply_box, judge_box, hidden_name])
        with gr.Tab("Try your own review (live Claude)"):
            gr.Markdown(f"Type or paste a review in any language. Each try costs about $0.005 and the server has a spend cap; "
                        f"each browser session is limited to {MAX_CALLS} tries.")
            text = gr.Textbox(label="Review", lines=5, placeholder="e.g. The parcel arrived late and the box was crushed.")
            with gr.Row():
                name = gr.Textbox(label="Customer first name (optional)", scale=2)
                judge = gr.Checkbox(label="Score the reply with the Sonnet 5 judge", value=True, scale=2)
                code = gr.Textbox(label="Access code (if required)", type="password", scale=2)
            go = gr.Button("Analyse and draft a reply", variant="primary")
            live_analysis = gr.Markdown()
            live_reply = gr.Textbox(label="Drafted reply", lines=6, interactive=False)
            live_status = gr.Markdown()
            live_note = gr.Markdown()
            used = gr.State(0)
            go.click(analyse_live, [text, name, judge, code, used], [live_analysis, live_reply, live_status, live_note, used])
        with gr.Tab("Spike detection (no cost)"):
            gr.Markdown("Runs the same detector as workflow 04 on the dashboard data: negative reviews per issue per week against the "
                        "previous four weeks.")
            with gr.Row():
                z = gr.Slider(2.0, 6.0, value=3.5, step=0.1, label="z-score threshold")
                mc = gr.Slider(3, 15, value=8, step=1, label="minimum reviews in the week")
            run = gr.Button("Run detector", variant="primary")
            spike_msg = gr.Markdown()
            spike_table = gr.Dataframe(label="Detected spikes", interactive=False)
            spike_plot = gr.LinePlot(x="week", y="negative reviews", color="issue", title="Weekly negative reviews (flagged issues)",
                                     x_title="Simulated week (W01 starts 6 Jul 2026)", y_title="Negative reviews")
            run.click(run_spikes, [z, mc], [spike_msg, spike_table, spike_plot])
            demo.load(run_spikes, [z, mc], [spike_msg, spike_table, spike_plot])
    return demo


if __name__ == "__main__":
    build().queue(default_concurrency_limit=2).launch(theme=gr.themes.Soft())
