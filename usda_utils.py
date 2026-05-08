"""
usda_utils.py
-------------
Shared utilities for the USDA FoodData Central pipeline.
All other scripts import from here — do not duplicate these functions elsewhere.

USDA Release Schedule: April and October annually.
Dataset URL pattern: FoodData_Central_branded_food_csv_{YYYY-MM-DD}.zip
"""

import os
import zipfile
import requests
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────────
# DATASET REGISTRY
# Update these URLs each April and October when USDA releases new data.
# ─────────────────────────────────────────────
USDA_DATASETS = {
    "2026-04": "https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_branded_food_csv_2026-04-30.zip",
    "2025-04": "https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_branded_food_csv_2025-04-24.zip",
    "2024-04": "https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_branded_food_csv_2024-04-18.zip",
}

# GitHub Release URLs for slim Parquet files (8-column, deduped).
# Run prepare_data.py locally, upload the output to GitHub Releases,
# then paste the download URLs here. Until populated, the app falls back
# to downloading the full ZIP from USDA.
SLIM_DATA_URLS: dict = {
     "2026-04": "https://github.com/fishslayer1941/avena_sativa/releases/download/data/2026-04_branded_food.parquet",
     "2025-04": "https://github.com/fishslayer1941/avena_sativa/releases/download/data/2025-04_branded_food.parquet",
     "2024-04": "https://github.com/fishslayer1941/avena_sativa/releases/download/data/2024-04_branded_food.parquet",
}

# Canonical column names from the USDA branded_food CSV (snake_case).
# Reference these instead of hardcoding strings in each script.
USDA_COLS = {
    "upc":       "gtin_upc",
    "brand":     "brand_name",
    "owner":     "brand_owner",
    "desc":      "short_description",
    "ingred":    "ingredients",
    "srv_size":  "serving_size",
    "srv_unit":  "serving_size_unit",
    "category":  "branded_food_category",
}

BASE_DATA_FOLDER = os.environ.get("DATA_FOLDER", "data")

_DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


# ─────────────────────────────────────────────
# DOWNLOAD & EXTRACTION
# ─────────────────────────────────────────────

def download_and_extract(zip_url: str, base_folder: str = BASE_DATA_FOLDER) -> str:
    """
    Downloads a USDA ZIP (if not already cached) and extracts it.
    Returns the path to the extracted directory.
    Caches by filename — re-running will not re-download.
    """
    os.makedirs(base_folder, exist_ok=True)
    zip_filename = zip_url.split("/")[-1]
    zip_path = os.path.join(base_folder, zip_filename)
    extracted_dir = os.path.join(base_folder, zip_filename.replace(".zip", ""))

    if not os.path.exists(zip_path):
        print(f"  ⬇  Downloading {zip_filename}...")
        with requests.get(zip_url, stream=True, headers=_DOWNLOAD_HEADERS, timeout=300) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
        print(f"  ✅ Download complete: {zip_path}")
        if not zipfile.is_zipfile(zip_path):
            os.remove(zip_path)
            raise ValueError(
                f"Downloaded file is corrupt (not a valid ZIP): {zip_filename}\n"
                "Please try loading again to re-download."
            )
    else:
        print(f"  📦 Using cached ZIP: {zip_path}")
        if not zipfile.is_zipfile(zip_path):
            os.remove(zip_path)
            raise ValueError(
                f"Cached ZIP is corrupt and has been deleted: {zip_filename}\n"
                "Please try loading again to re-download."
            )

    if not os.path.exists(extracted_dir):
        print(f"  📂 Extracting {zip_filename}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extracted_dir)
        print(f"  ✅ Extracted to: {extracted_dir}")
    else:
        print(f"  📂 Using cached extract: {extracted_dir}")

    return extracted_dir


def find_branded_csv(folder: str) -> str:
    """
    Walks the extracted directory tree to find the branded_food CSV file.
    Raises FileNotFoundError if not found.
    """
    for root, _, files in os.walk(folder):
        for file in files:
            if "branded_food" in file.lower() and file.endswith(".csv"):
                return os.path.join(root, file)
    raise FileNotFoundError(
        f"Could not find a branded_food CSV in: {folder}\n"
        "Check that the ZIP extracted correctly."
    )


def download_slim_data(release_key: str, base_folder: str = BASE_DATA_FOLDER) -> str:
    """
    Downloads the slim Parquet file for a release from GitHub Releases (if not cached).
    Returns the local path to the .parquet file.
    """
    os.makedirs(base_folder, exist_ok=True)
    parquet_path = os.path.join(base_folder, f"{release_key}_branded_food.parquet")

    if os.path.exists(parquet_path):
        print(f"  📦 Using cached Parquet: {parquet_path}")
        return parquet_path

    if release_key not in SLIM_DATA_URLS:
        raise KeyError(
            f"No slim data URL configured for '{release_key}'. "
            "Run prepare_data.py, upload to GitHub Releases, then add the URL to SLIM_DATA_URLS."
        )

    url = SLIM_DATA_URLS[release_key]
    print(f"  ⬇  Downloading slim dataset: {release_key}...")
    with requests.get(url, stream=True, headers=_DOWNLOAD_HEADERS, timeout=300) as r:
        r.raise_for_status()
        with open(parquet_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
    print(f"  ✅ Slim dataset cached: {parquet_path}")
    return parquet_path


# ─────────────────────────────────────────────
# DATAFRAME LOADING
# ─────────────────────────────────────────────

def load_branded_food(release_key: str, base_folder: str = BASE_DATA_FOLDER) -> pd.DataFrame:
    """
    Loads the branded food CSV for a given release key (e.g. "2025-04").
    Deduplicates on gtin_upc and normalizes ingredient text to lowercase.

    Args:
        release_key: Key from USDA_DATASETS dict, e.g. "2025-04"
        base_folder:  Local cache directory

    Returns:
        Cleaned DataFrame with standard USDA columns.
    """
    if release_key not in USDA_DATASETS:
        raise KeyError(
            f"Release key '{release_key}' not found. "
            f"Available keys: {list(USDA_DATASETS.keys())}"
        )

    print(f"\n📥 Loading USDA dataset: {release_key}")
    parquet_path = os.path.join(base_folder, f"{release_key}_branded_food.parquet")

    if os.path.exists(parquet_path) or release_key in SLIM_DATA_URLS:
        # Fast path: slim Parquet (local cache or GitHub Releases download)
        path = download_slim_data(release_key, base_folder)
        print(f"  📄 Reading Parquet: {path}")
        df = pd.read_parquet(path)
    else:
        # Fallback: full ZIP from USDA (used locally before prepare_data.py has been run)
        url = USDA_DATASETS[release_key]
        extracted = download_and_extract(url, base_folder)
        csv_path = find_branded_csv(extracted)
        print(f"  📄 Reading CSV: {csv_path}")
        df = pd.read_csv(csv_path, dtype=str, low_memory=False)

        missing = [v for v in USDA_COLS.values() if v not in df.columns]
        if missing:
            print(f"  ⚠  Missing expected columns: {missing}")
            print(f"  ℹ  Columns found: {df.columns.tolist()}")

        before = len(df)
        df = df.drop_duplicates(subset=USDA_COLS["upc"])
        print(f"  🔢 Rows: {before:,} → {len(df):,} after deduplication on gtin_upc")

    # Normalize ingredients to lowercase for consistent matching
    if USDA_COLS["ingred"] in df.columns:
        df[USDA_COLS["ingred"]] = df[USDA_COLS["ingred"]].str.lower().str.strip()

    return df


# ─────────────────────────────────────────────
# INGREDIENT SEARCH
# ─────────────────────────────────────────────

def filter_by_ingredient(df: pd.DataFrame, search_term: str, exact: bool = False) -> pd.DataFrame:
    """
    Filters the DataFrame to rows where the ingredients column contains search_term.

    Args:
        df:          Branded food DataFrame (from load_branded_food)
        search_term: Ingredient string to search for
        exact:       If True, uses regex=False for exact substring match.
                     If False, uses regex=True for flexible matching.

    Returns:
        Filtered DataFrame.
    """
    col = USDA_COLS["ingred"]
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame.")

    term_lower = search_term.lower().strip()
    mask = df[col].str.contains(term_lower, case=False, na=False, regex=not exact)
    result = df.loc[mask].copy()
    print(f"  🔍 '{search_term}': {len(result):,} matches found")
    return result


# ─────────────────────────────────────────────
# OUTPUT HELPERS
# ─────────────────────────────────────────────

def save_output(df: pd.DataFrame, filename: str, output_folder: str = "output") -> str:
    """
    Saves a DataFrame to CSV in the output folder.
    Filename is automatically stamped with today's date.
    Returns the full output path.
    """
    os.makedirs(output_folder, exist_ok=True)
    date_str = datetime.today().strftime("%Y-%m-%d")
    base, ext = os.path.splitext(filename)
    out_path = os.path.join(output_folder, f"{base}_{date_str}{ext}")
    df.to_csv(out_path, index=False)
    print(f"  💾 Saved {len(df):,} rows → {out_path}")
    return out_path


def validate_output(df: pd.DataFrame, name: str) -> None:
    """
    Prints a basic quality summary for an output DataFrame.
    """
    print(f"\n  📊 Output QA — {name}")
    print(f"     Rows:    {len(df):,}")
    print(f"     Columns: {df.columns.tolist()}")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if not nulls.empty:
        print(f"     ⚠  Nulls:\n{nulls.to_string()}")
    else:
        print("     ✅ No null values in output columns")
