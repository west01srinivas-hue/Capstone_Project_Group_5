"""LLM-as-judge for drafted replies (rubric: empathy, specificity, correctness, tone + guardrail flags)."""
import json
from pathlib import Path

from src.response.generate import parse_json, text_of

ROOT = Path(__file__).resolve().parents[2]
JUDGE_VERSION = "judge_v1"
DIMENSIONS = ["empathy", "specificity", "correctness", "tone"]
FLAGS = ["promises_outcome", "invents_facts", "wrong_language", "needs_human_attention"]


def load_prompt() -> str:
    return (ROOT / "prompts" / f"{JUDGE_VERSION}.txt").read_text(encoding="utf-8")


def judge(client, model: str, review: dict, reply: str, system: str):
    """Returns (scores dict, input tokens, output tokens)."""
    payload = {"review": review["review_text"][:1500], "reply": reply,
               "sentiment": review["sentiment"], "topic": review["topic"],
               "customer_name": review.get("customer_name")}
    resp = client.messages.create(
        model=model, max_tokens=2000, system=system,
        messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
    )
    return parse_json(text_of(resp)), resp.usage.input_tokens, resp.usage.output_tokens


def acceptable(scores: dict) -> bool:
    """Automatic proxy for human acceptance: every dimension >= 4 and no guardrail flag except needs_human_attention."""
    return (all(scores[d] >= 4 for d in DIMENSIONS)
            and not scores["promises_outcome"] and not scores["invents_facts"] and not scores["wrong_language"])
