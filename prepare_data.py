"""
prepare_data.py
---------------
One-time local script to generate slim Parquet files from USDA ZIP downloads.
Run this locally whenever USDA publishes a new release (April and October).

Output: data/{release_key}_branded_food.parquet  (8 columns, deduped, ~30-50 MB each)

Usage:
    python prepare_data.py              # process all releases in USDA_DATASETS
    python prepare_data.py 2026-04      # process one release

After running:
  1. Upload the .parquet files from data/ to GitHub Releases as binary assets.
  2. Copy the asset download URLs into SLIM_DATA_URLS in usda_utils.py.
  3. Commit and push, then redeploy on Streamlit Cloud.
"""

import os
import sys
import pandas as pd
from usda_utils import (
    USDA_DATASETS,
    USDA_COLS,
    BASE_DATA_FOLDER,
    download_and_extract,
    find_branded_csv,
)

KEEP_COLS = list(USDA_COLS.values())


def prepare_parquet(release_key: str, base_folder: str = BASE_DATA_FOLDER) -> str:
    if release_key not in USDA_DATASETS:
        raise KeyError(f"Unknown release: '{release_key}'. Choose from {list(USDA_DATASETS)}")

    out_path = os.path.join(base_folder, f"{release_key}_branded_food.parquet")
    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / 1_048_576
        print(f"  ✅ Already exists ({size_mb:.1f} MB): {out_path}")
        return out_path

    url = USDA_DATASETS[release_key]
    extracted = download_and_extract(url, base_folder)
    csv_path = find_branded_csv(extracted)

    # Only load columns the app actually uses
    available = pd.read_csv(csv_path, nrows=0).columns.tolist()
    usecols = [c for c in KEEP_COLS if c in available]
    skipped = [c for c in KEEP_COLS if c not in available]
    if skipped:
        print(f"  ⚠  Columns not found in CSV (skipping): {skipped}")

    print(f"  📄 Reading {len(usecols)} columns from CSV...")
    df = pd.read_csv(csv_path, dtype=str, low_memory=False, usecols=usecols)

    before = len(df)
    df = df.drop_duplicates(subset=USDA_COLS["upc"])
    print(f"  🔢 Rows: {before:,} → {len(df):,} after dedup on gtin_upc")

    df.to_parquet(out_path, index=False)
    size_mb = os.path.getsize(out_path) / 1_048_576
    print(f"  💾 Saved ({size_mb:.1f} MB): {out_path}")
    return out_path


if __name__ == "__main__":
    keys = sys.argv[1:] or list(USDA_DATASETS.keys())
    print(f"Preparing Parquet files for: {keys}\n")
    results = []
    for key in keys:
        print(f"{'─' * 50}\nRelease: {key}")
        try:
            path = prepare_parquet(key)
            results.append((key, path, None))
        except Exception as exc:
            print(f"  ❌ {exc}")
            results.append((key, None, exc))

    print(f"\n{'─' * 50}")
    print("Summary:")
    for key, path, err in results:
        if path:
            print(f"  ✅ {key}  →  {path}")
        else:
            print(f"  ❌ {key}  →  {err}")
    print("\nNext steps:")
    print("  1. Upload .parquet files to GitHub Releases")
    print("  2. Add download URLs to SLIM_DATA_URLS in usda_utils.py")
    print("  3. Commit, push, redeploy on Streamlit Cloud")
