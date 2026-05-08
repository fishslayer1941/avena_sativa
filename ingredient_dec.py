"""
ingredient_dec.py
-----------------
Filters USDA branded food data for products containing a target ingredient,
then transforms the GTIN/UPC into the format required for Nielsen data joins.

GTIN Transform Logic:
    - Standard USDA GTIN is 14 digits (with leading zeros and a check digit).
    - Nielsen expects a 12-digit UPC-A: strip the check digit (last digit),
      then either remove a leading '00' prefix (if already 13-digit with '00')
      or prepend a single leading '0'.
    - The modified_gtin_upc column is used as the join key to Nielsen.

Usage:
    python ingredient_dec.py                              # uses default (oat)
    python ingredient_dec.py --term "oat fiber"
    python ingredient_dec.py --term "inulin" --release 2025-04
    python ingredient_dec.py --term "chicory root" --output-folder "C:/OneDrive/USDA_Pipeline/output"

Requirements:
    pip install pandas requests
"""

import argparse
import sys
import pandas as pd

sys.path.insert(0, ".")
from usda_utils import (
    load_branded_food,
    filter_by_ingredient,
    save_output,
    validate_output,
    USDA_DATASETS,
    USDA_COLS,
)

# ─────────────────────────────────────────────
# DEFAULTS
# ─────────────────────────────────────────────
DEFAULT_SEARCH_TERM = "oat"
DEFAULT_RELEASE     = "2025-12"
DEFAULT_OUTPUT      = "output"


# ─────────────────────────────────────────────
# GTIN TRANSFORMATION
# ─────────────────────────────────────────────

def modify_gtin(gtin: str | None) -> str | None:
    """
    Converts a USDA GTIN/UPC to Nielsen-compatible 12-digit UPC-A format.

    Rules:
        - Input is a 14-digit GTIN (padded with leading zeros by USDA).
        - Check digit (last digit) is always dropped.
        - If the result starts with '00' and is 13 chars → strip one leading '0'.
        - Otherwise, prepend a '0' to make it 12 digits.

    Examples:
        "00012345678905" → drop check → "0001234567890" → starts with '00' + len 13
                        → strip one '0' → "001234567890"  ✓ (12 digits)

        "01234567890123" → drop check → "0123456789012" → prepend '0' → not needed
                        → strip... handled by else branch.

    Note: If your Nielsen format requires a different transformation,
    update this function — it is the single source of truth for GTIN logic.
    """
    if pd.isna(gtin) or not isinstance(gtin, str):
        return None
    gtin = gtin.strip()
    if not gtin.isdigit():
        return None

    # Drop check digit (last character)
    stripped = gtin[:-1]

    if stripped.startswith("00") and len(stripped) == 13:
        # Remove one leading zero to get 12-digit UPC-A
        return stripped[1:]
    else:
        # Prepend a leading zero
        return "0" + stripped


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run(search_term: str, release: str, output_folder: str) -> None:
    print("\n" + "="*60)
    print("  USDA Ingredient Decoder + Nielsen GTIN Transform")
    print(f"  Term:    {search_term}")
    print(f"  Release: {release}")
    print("="*60)

    # Load full dataset
    df = load_branded_food(release)

    # Filter to target ingredient (word-boundary regex for short terms like "oat")
    # Using regex=True here to support \b word boundaries for short terms
    ingred_col = USDA_COLS["ingred"]

    # For short common words (e.g. "oat"), use word boundary to avoid
    # matching "oatmeal" when you only want "oat" — adjust as needed.
    use_word_boundary = len(search_term.split()) == 1 and len(search_term) <= 5
    if use_word_boundary:
        pattern = rf"\b{search_term.lower()}"
        mask = df[ingred_col].str.contains(pattern, case=False, na=False, regex=True)
        df_match = df.loc[mask].copy()
        print(f"  🔍 '{search_term}' (word-boundary): {len(df_match):,} matches")
    else:
        df_match = filter_by_ingredient(df, search_term)

    if df_match.empty:
        print(f"\n  ⚠  No results for '{search_term}'. Try a broader term.")
        return

    # Apply GTIN transformation for Nielsen join
    upc_col = USDA_COLS["upc"]
    df_match["modified_gtin_upc"] = df_match[upc_col].apply(modify_gtin)

    # Flag rows where GTIN transform produced a null (malformed source UPCs)
    null_gtins = df_match["modified_gtin_upc"].isnull().sum()
    if null_gtins > 0:
        print(f"  ⚠  {null_gtins:,} rows have null modified_gtin_upc (malformed source UPCs)")

    # Enrich with usage tier before column selection so we can place columns correctly
    from ingredient_position import enrich_with_usage
    df_match = enrich_with_usage(df_match, search_term)

    # Select and order output columns (usage tier placed after ingredients)
    desired_cols = [
        upc_col,
        "modified_gtin_upc",        # Nielsen join key
        USDA_COLS["owner"],
        USDA_COLS["brand"],
        USDA_COLS["desc"],
        USDA_COLS["category"],
        USDA_COLS["ingred"],
        "usage_tier",
        "ingredient_position",
        "total_ingredients",
        "usage_context",
        USDA_COLS["srv_size"],
        USDA_COLS["srv_unit"],
    ]
    output_cols = [c for c in desired_cols if c in df_match.columns]
    df_out = df_match[output_cols].reset_index(drop=True)

    # QA + save
    validate_output(df_out, f"Ingredient Decoder — '{search_term}'")
    safe_term = search_term.replace(" ", "_")
    save_output(df_out, f"ingredient_dec_{safe_term}.csv", output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter USDA data by ingredient and apply Nielsen GTIN transform."
    )
    parser.add_argument(
        "--term",
        type=str,
        default=DEFAULT_SEARCH_TERM,
        help=f"Ingredient to filter on (default: '{DEFAULT_SEARCH_TERM}')"
    )
    parser.add_argument(
        "--release",
        type=str,
        default=DEFAULT_RELEASE,
        choices=list(USDA_DATASETS.keys()),
        help=f"USDA dataset release key (default: {DEFAULT_RELEASE})"
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Output folder for CSVs. Can be a OneDrive/SharePoint path."
    )
    args = parser.parse_args()
    run(args.term, args.release, args.output_folder)
