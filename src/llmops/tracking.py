"""Every Claude call from the service goes through here: logs prompt version, tokens, cost and latency,
and refuses new calls once the spend cap is reached (protects the small API credit)."""
import json
import os
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = Path(os.getenv("LLM_LOG_FILE", ROOT / "data" / "processed" / "llm_calls.jsonl"))
PRICES = {  # USD per million tokens (input, output)
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-5": (2.0, 10.0),
}
_lock = threading.Lock()


class BudgetExceeded(Exception):
    pass


def cap_usd() -> float:
    return float(os.getenv("MAX_SPEND_USD", "1.0"))


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    pin, pout = PRICES.get(model, (2.0, 10.0))  # unknown model: assume the dearer price
    return tokens_in / 1e6 * pin + tokens_out / 1e6 * pout


def spent_usd() -> float:
    if not LOG.exists():
        return 0.0
    total = 0.0
    for line in LOG.read_text(encoding="utf-8").splitlines():
        if line.strip():
            total += json.loads(line).get("cost_usd", 0.0)
    return total


class TrackedClient:
    """Drop-in for `anthropic.Anthropic()` in code that only calls `client.messages.create(...)`."""

    def __init__(self, client, purpose: str, prompt_version: str):
        self._client = client
        self.purpose = purpose
        self.prompt_version = prompt_version
        self.messages = self

    def create(self, **kwargs):
        with _lock:
            if spent_usd() >= cap_usd():
                raise BudgetExceeded(f"spend cap of ${cap_usd():.2f} reached")
        t0 = time.time()
        resp = self._client.messages.create(**kwargs)
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "purpose": self.purpose, "prompt_version": self.prompt_version,
            "model": kwargs.get("model"), "tokens_in": resp.usage.input_tokens, "tokens_out": resp.usage.output_tokens,
            "latency_s": round(time.time() - t0, 2),
        }
        rec["cost_usd"] = round(cost_usd(rec["model"], rec["tokens_in"], rec["tokens_out"]), 6)
        with _lock:
            LOG.parent.mkdir(parents=True, exist_ok=True)
            with LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + chr(10))
        return resp
