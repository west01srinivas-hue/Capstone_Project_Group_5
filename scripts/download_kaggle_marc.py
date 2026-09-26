"""List / download the Kaggle dataset mexwell/amazon-reviews-multi (MARC) into data/raw/kaggle_marc/.

Credentials (any one of):
  - KAGGLE_API_TOKEN in .env  (new-style access token from kaggle.com/settings > API Tokens)
  - KAGGLE_USERNAME + KAGGLE_KEY in .env  (legacy key)
  - ~/.kaggle/kaggle.json

  python scripts/download_kaggle_marc.py --list                 # show files and sizes, downloads nothing
  python scripts/download_kaggle_marc.py test.csv validation.csv
"""
import argparse
import os
import sys
import zipfile
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
for var in ("KAGGLE_API_TOKEN", "KAGGLE_KEY"):  # empty values must not shadow other auth methods
    if var in os.environ and not os.environ[var].strip():
        del os.environ[var]

DATASET = "mexwell/amazon-reviews-multi"
OUT = ROOT / "data" / "raw" / "kaggle_marc"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="file names to download")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    listing = api.dataset_list_files(DATASET)
    files = listing.files if hasattr(listing, "files") else listing
    print(f"{DATASET}:")
    for f in files:
        print(f"  {f.name:40s} {getattr(f, 'total_bytes', getattr(f, 'size', 0)) / 1e6:10.1f} MB")
    if args.list or not args.files:
        return

    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.files:
        print(f"downloading {name} ...")
        api.dataset_download_file(DATASET, name, path=str(OUT), force=False)
        z = OUT / f"{name}.zip"
        if z.exists():
            with zipfile.ZipFile(z) as zf:
                zf.extractall(OUT)
            z.unlink()
        print("  ->", OUT / name)


if __name__ == "__main__":
    sys.exit(main())
