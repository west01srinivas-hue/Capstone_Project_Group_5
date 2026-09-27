"""Upload space_build/ to the Hugging Face Space and put it on the free CPU tier.

  python scripts/build_space.py
  python scripts/deploy_space.py

Needs HF_TOKEN (a token with write access) in .env. The Anthropic key is NOT sent by this script: add it yourself as a
Space secret named ANTHROPIC_API_KEY in the Space settings.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
REPO = os.getenv("HF_SPACE_ID", "SrinivasanHF/review-insights-demo")
BUNDLE = ROOT / "space_build"

if __name__ == "__main__":
    token = os.getenv("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set in .env (create a write token at https://huggingface.co/settings/tokens)")
    if not (BUNDLE / "app.py").exists():
        sys.exit("space_build/ is missing: run python scripts/build_space.py first")
    api = HfApi(token=token)
    print("signed in as:", api.whoami()["name"])
    api.upload_folder(folder_path=str(BUNDLE), repo_id=REPO, repo_type="space", commit_message="Deploy demo app",
                      ignore_patterns=["__pycache__", "*.pyc", "data/processed"])
    try:
        api.request_space_hardware(repo_id=REPO, hardware="cpu-basic")  # free tier; no credit used
    except Exception as e:  # the Space may already be on it
        print("hardware request:", type(e).__name__, str(e)[:120])
    print(f"done: https://huggingface.co/spaces/{REPO}")
