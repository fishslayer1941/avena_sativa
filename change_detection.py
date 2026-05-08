"""
change_detection.py
-------------------
Detects year-over-year changes in USDA branded food data for a target ingredient:

    1. NEW PRODUCTS     — GTINs present in the newer release but not the older one,
                          where the ingredient is present in the new release.

    2. FORMULA ADDED    — GTINs present in both releases, where the target ingredient
                          was NOT in the older release but IS in the newer one.

    3. FORMULA REMOVED  — GTINs present in both releases, where the target ingredient
                          WAS in the older release but is NOT in the newer one.

USDA releases data twice annually: April and October.
Update USDA_DATASETS in usda_utils.py when new releases drop.

Usage:
    python change_detection.py                               # uses defaults
    python change_detection.py --term "oat fiber"
    python change_detection.py --term "resistant starch" --new-release 2025-04 --old-release 2024-04
    python change_detection.py --term "chicory root" --output-folder "C:/OneDrive/USDA_Pipeline/output"

Requirements:
    pip install pandas requests
"""

import argparse
import sys
import pandas as pd

sys.path.insert(0, ".")
from usda_utils import (
    load_branded_food,
    save_output,
    validate_output,
    USDA_DATASETS,
    USDA_COLS,
)

# ─────────────────────────────────────────────
# DEFAULTS
# ─────────────────────────────────────────────
DEFAULT_SEARCH_TERM = "oat fiber"
DEFAULT_NEW_RELEASE = "2025-12"   # More recent dataset
DEFAULT_OLD_RELEASE = "2025-04"   # Comparison baseline
DEFAULT_OUTPUT      = "output"


# ─────────────────────────────────────────────
# CHANGE DETECTION LOGIC
# ─────────────────────────────────────────────

def contains_term(series: pd.Series, term: str) -> pd.Series:
    """Returns a boolean mask: True where series contains term (case-insensitive)."""
    return series.str.contains(term.lower(), case=False, na=False, regex=False)


def detect_changes(
    df_old: pd.DataFrame,
    df_new: pd.DataFrame,
    search_term: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compares two USDA DataFrames and returns three change sets.

    Args:
        df_old:      Older USDA release DataFrame
        df_new:      Newer USDA release DataFrame
        search_term: Ingredient string to track

    Returns:
        Tuple of (new_products, formula_added, formula_removed)
    """
    upc  = USDA_COLS["upc"]
    ing  = USDA_COLS["ingred"]
    term = search_term.lower()

    # ── New Products ──────────────────────────────────────────────
    # GTINs in new release that are NOT in old release, and contain the term
    df_new_with_term = df_new[contains_term(df_new[ing], term)]
    new_gtins = set(df_new_with_term[upc])
    old_gtins = set(df_old[upc])
    new_products = df_new_with_term[~df_new_with_term[upc].isin(old_gtins)].copy()
    new_products["change_type"] = "New Product"

    # ── Formula Changes (shared GTINs only) ───────────────────────
    shared_gtins = new_gtins & old_gtins
    df_old_shared = df_old[df_old[upc].isin(shared_gtins)].set_index(upc)
    df_new_shared = df_new[df_new[upc].isin(shared_gtins)].set_index(upc)

    merged = df_old_shared[[ing]].join(
        df_new_shared[[ing]],
        lsuffix="_old",
        rsuffix="_new",
        how="inner"
    )

    old_has_term = contains_term(merged[f"{ing}_old"], term)
    new_has_term = contains_term(merged[f"{ing}_new"], term)

    # Added: was NOT in old, IS in new
    added_idx = merged[~old_has_term & new_has_term].index
    # Removed: WAS in old, NOT in new
    removed_idx = merged[old_has_term & ~new_has_term].index

    def build_change_df(idx, change_label):
        if len(idx) == 0:
            return pd.DataFrame()
        rows = df_new_shared.loc[idx].copy().reset_index()
        # Attach old ingredients for side-by-side comparison
        old_ing = df_old_shared.loc[idx, ing].rename(f"{ing}_old").reset_index()
        rows = rows.merge(old_ing, on=upc, how="left")
        rows["change_type"] = change_label
        return rows

    formula_added   = build_change_df(added_idx,   "Formula Added")
    formula_removed = build_change_df(removed_idx, "Formula Removed")

    return new_products, formula_added, formula_removed


# ─────────────────────────────────────────────
# OUTPUT FORMATTING
# ─────────────────────────────────────────────

def format_output(df: pd.DataFrame, include_old_ing: bool = False) -> pd.DataFrame:
    """
    Selects and orders columns for a clean output CSV.
    """
    upc  = USDA_COLS["upc"]
    ing  = USDA_COLS["ingred"]

    desired = [
        "change_type",
        upc,
        USDA_COLS["owner"],
        USDA_COLS["brand"],
        USDA_COLS["desc"],
        USDA_COLS["category"],
    ]

    if include_old_ing:
        desired += [f"{ing}_old", ing]
    else:
        desired += [ing]

    desired += [USDA_COLS["srv_size"], USDA_COLS["srv_unit"]]

    keep = [c for c in desired if c in df.columns]
    return df[keep].reset_index(drop=True)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run(search_term: str, new_release: str, old_release: str, output_folder: str) -> None:
    print("\n" + "="*60)
    print("  USDA Year-Over-Year Change Detection")
    print(f"  Term:        {search_term}")
    print(f"  New release: {new_release}")
    print(f"  Old release: {old_release}")
    print("="*60)

    if new_release == old_release:
        raise ValueError("--new-release and --old-release must be different.")

    # Load both datasets (cached after first run)
    df_new = load_branded_food(new_release)
    df_old = load_branded_food(old_release)

    # Run detection
    new_products, formula_added, formula_removed = detect_changes(
        df_old, df_new, search_term
    )

    # Format and save each output
    safe_term = search_term.replace(" ", "_")

    print(f"\n  📦 New Products:      {len(new_products):,}")
    print(f"  ➕ Formula Added:     {len(formula_added):,}")
    print(f"  ➖ Formula Removed:   {len(formula_removed):,}")

    from ingredient_position import enrich_with_usage

    datasets = [
        (new_products,    f"new_products_{safe_term}.csv",       False, "New Products"),
        (formula_added,   f"formula_added_{safe_term}.csv",      True,  "Formula Added"),
        (formula_removed, f"formula_removed_{safe_term}.csv",    True,  "Formula Removed"),
    ]

    for df, filename, inc_old, label in datasets:
        if df.empty:
            print(f"\n  ℹ  No {label} records — skipping file.")
            continue
        df_out = format_output(df, include_old_ing=inc_old)
        df_out = enrich_with_usage(df_out, search_term)
        validate_output(df_out, label)
        save_output(df_out, filename, output_folder)

    # Combined summary file (all three types in one CSV)
    all_changes = pd.concat(
        [enrich_with_usage(format_output(d, True), search_term)
         for d in [new_products, formula_added, formula_removed]
         if not d.empty],
        ignore_index=True
    )
    if not all_changes.empty:
        save_output(all_changes, f"all_changes_{safe_term}.csv", output_folder)
        print(f"\n  ✅ Combined summary: {len(all_changes):,} total change records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect YoY ingredient changes in USDA branded food data."
    )
    parser.add_argument(
        "--term",
        type=str,
        default=DEFAULT_SEARCH_TERM,
        help=f"Ingredient string to track (default: '{DEFAULT_SEARCH_TERM}')"
    )
    parser.add_argument(
        "--new-release",
        type=str,
        default=DEFAULT_NEW_RELEASE,
        choices=list(USDA_DATASETS.keys()),
        help=f"Newer USDA release key (default: {DEFAULT_NEW_RELEASE})"
    )
    parser.add_argument(
        "--old-release",
        type=str,
        default=DEFAULT_OLD_RELEASE,
        choices=list(USDA_DATASETS.keys()),
        help=f"Baseline USDA release key (default: {DEFAULT_OLD_RELEASE})"
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Output folder for CSVs. Can be a OneDrive/SharePoint path."
    )
    args = parser.parse_args()
    run(args.term, args.new_release, args.old_release, args.output_folder)
