"""Response drafting: pick the (sentiment, topic) playbook, apply the brand voice, ask Claude for a reply."""
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "prompts" / "response"
VERSION = os.getenv("RESPONSE_VERSION", "v1")
SYSTEM_VERSION = f"response_{VERSION}"
TOPICS = {"shipping", "quality", "support", "features", "price", "ux", "other"}


def load_assets() -> dict:
    brand = json.loads((PROMPTS / "brand.json").read_text(encoding="utf-8"))
    system = (PROMPTS / f"system_{VERSION}.txt").read_text(encoding="utf-8")
    for key, value in brand.items():
        system = system.replace("{" + key + "}", value)
    playbooks = PROMPTS / f"playbooks_{VERSION}.json"
    if not playbooks.exists():
        playbooks = PROMPTS / "playbooks_v1.json"
    return {"system": system, "playbooks": json.loads(playbooks.read_text(encoding="utf-8"))}


def build_user_message(review: dict, assets: dict) -> str:
    topic = review["topic"] if review["topic"] in TOPICS else "other"
    key = f"{review['sentiment']}/{topic}"
    payload = {
        "review": review["review_text"][:1500],
        "customer_name": review.get("customer_name"),
        "sentiment": review["sentiment"],
        "topic": topic,
        "issue": review.get("issue"),
        "playbook": assets["playbooks"][key],
    }
    return json.dumps(payload, ensure_ascii=False)


def text_of(resp) -> str:
    """First text block of a response (Sonnet 5 may put a thinking block first)."""
    return next(b.text for b in resp.content if b.type == "text")


def parse_json(text: str) -> dict:
    return json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip()))


def draft(client, model: str, review: dict, assets: dict):
    """Returns (reply text, input tokens, output tokens)."""
    resp = client.messages.create(
        model=model, max_tokens=600, system=assets["system"],
        messages=[{"role": "user", "content": build_user_message(review, assets)}],
    )
    return parse_json(text_of(resp))["reply"].strip(), resp.usage.input_tokens, resp.usage.output_tokens
