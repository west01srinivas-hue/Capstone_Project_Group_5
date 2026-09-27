"""Assemble the folder that goes to the Hugging Face Space: python scripts/build_space.py  ->  space_build/
Only the files the demo needs are copied (no data/raw, no embeddings, no .env)."""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "space_build"

FILES = [
    "app.py",
    "src/__init__.py", "src/analysis/__init__.py", "src/analysis/trends.py",
    "src/api/__init__.py", "src/api/main.py",
    "src/llmops/__init__.py", "src/llmops/tracking.py",
    "src/response/__init__.py", "src/response/generate.py", "src/response/judge.py",
    "prompts/classify_v3.txt", "prompts/issue_v1.txt", "prompts/judge_v1.txt",
    "prompts/response/brand.json", "prompts/response/system_v2.txt", "prompts/response/playbooks_v1.json",
    "data/samples/demo_examples.json", "data/samples/dashboard_reviews.csv",
]

if __name__ == "__main__":
    if OUT.exists():
        shutil.rmtree(OUT)
    for rel in FILES:
        src = ROOT / rel
        if not src.exists():
            raise SystemExit(f"missing file: {rel}")
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    shutil.copy2(ROOT / "space" / "README.md", OUT / "README.md")
    shutil.copy2(ROOT / "space" / "requirements.txt", OUT / "requirements.txt")
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"built {OUT} ({sum(1 for p in OUT.rglob('*') if p.is_file())} files, {total / 1e3:.0f} KB)")
