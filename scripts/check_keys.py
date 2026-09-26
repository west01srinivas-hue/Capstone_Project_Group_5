"""Smoke-test the Claude key with one tiny call. Reads the key from .env."""
import os
import sys

from dotenv import load_dotenv

load_dotenv()

PROMPT = "Reply with exactly one word: the sentiment (positive/negative/neutral) of: 'Arrived late and broken.'"

key = os.getenv("ANTHROPIC_API_KEY")
if not key:
    print("CLAUDE: no ANTHROPIC_API_KEY set")
    sys.exit(1)

import anthropic

model = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5")
r = anthropic.Anthropic(api_key=key).messages.create(
    model=model, max_tokens=16, messages=[{"role": "user", "content": PROMPT}]
)
print(f"CLAUDE OK  model={model}  reply={r.content[0].text!r}  tokens in/out={r.usage.input_tokens}/{r.usage.output_tokens}")
