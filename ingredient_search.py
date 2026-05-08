"""
ingredient_search.py
--------------------
Searches USDA FoodData Central branded food data for a target ingredient.
Classifies results by starch type (if applicable) and botanical source.
Exports a clean CSV ready for Power BI or Nielsen join.

Usage:
    python ingredient_search.py                              # uses default search term
    python ingredient_search.py --term "oat fiber"          # single term
    python ingredient_search.py --term "resistant starch" --release 2025-04
    python ingredient_search.py --term "inulin" --output-folder "C:/OneDrive/USDA_Pipeline/output"

Requirements:
    pip install pandas requests rapidfuzz
"""

import argparse
import sys
import pandas as pd
import re
from rapidfuzz import process, fuzz

# Import shared utilities
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
# DEFAULTS — change these without touching logic
# ─────────────────────────────────────────────
DEFAULT_SEARCH_TERM = "resistant starch"
DEFAULT_RELEASE     = "2025-12"          # Key from USDA_DATASETS in usda_utils.py
DEFAULT_OUTPUT      = "output"           # Local folder OR OneDrive path


# ─────────────────────────────────────────────
# FUZZY MATCHING
# ─────────────────────────────────────────────

# Canonical starch names to match against. Add new variants here as needed.
KNOWN_STARCHES = [
    "resistant maltodextrin",
    "resistant corn starch",
    "resistant wheat starch",
    "resistant tapioca starch",
    "resistant potato starch",
    "resistant oat starch",
    "rs2 starch",
    "rs3 starch",
    "retrograded starch",
    "modified resistant starch",
    "high-amylose maize starch",
    "resistant dextrin",
    "digestive-resistant starch",
]

# Synonym map: common label variants → canonical name
STARCH_SYNONYMS = {
    "resistant corn starch":    ["rs2 corn", "modified corn starch", "retrograded corn starch"],
    "resistant wheat starch":   ["rs3 wheat", "modified wheat starch", "retrograded wheat"],
    "resistant maltodextrin":   ["dietary fiber maltodextrin", "digestion-resistant maltodextrin"],
    "resistant tapioca starch": ["modified tapioca starch", "retrograded tapioca"],
}

FUZZY_THRESHOLD = 80  # Minimum score (0–100) to accept a fuzzy match


def fuzzy_match_starch(ingredients: str) -> str | None:
    """
    Attempts to match individual ingredient tokens against KNOWN_STARCHES
    using rapidfuzz + synonym lookup. Returns matched starch name(s) or None.
    """
    if not isinstance(ingredients, str) or not ingredients.strip():
        return None

    items = [i.strip() for i in ingredients.split(",") if i.strip()]
    matched = set()

    for item in items:
        # rapidfuzz fuzzy match
        result = process.extractOne(item, KNOWN_STARCHES, scorer=fuzz.token_set_ratio)
        if result and result[1] >= FUZZY_THRESHOLD:
            matched.add(result[0])

        # Synonym lookup for precision
        for canonical, synonyms in STARCH_SYNONYMS.items():
            if any(syn in item for syn in synonyms):
                matched.add(canonical)

    return ", ".join(sorted(matched)) if matched else None


# ─────────────────────────────────────────────
# STARCH TYPE CLASSIFICATION
# ─────────────────────────────────────────────

# Maps keyword → readable label. Applied to fragments near "resistant" keyword.
STARCH_TYPES = {
    "tapioca": "Tapioca",
    "corn":    "Corn",
    "potato":  "Potato",
    "wheat":   "Wheat",
    "oat":     "Oat",
}


def classify_starch_type(ingredients: str) -> str:
    """
    Extracts the botanical starch type by looking for STARCH_TYPES keywords
    within ingredient fragments that contain the word 'resistant'.
    Returns a type label or 'Other/Unknown' or 'Blend'.
    """
    if not isinstance(ingredients, str):
        return "Other/Unknown"

    fragments = re.split(r",|\(|\)", ingredients.lower())
    relevant = [f.strip() for f in fragments if "resistant" in f]

    found = set()
    for frag in relevant:
        for keyword, label in STARCH_TYPES.items():
            if keyword in frag:
                found.add(label)

    if len(found) == 0:
        return "Other/Unknown"
    elif len(found) == 1:
        return list(found)[0]
    else:
        return "Blend"


def extract_matching_phrases(ingredients: str, keyword: str = "resistant") -> str:
    """
    Extracts comma-delimited ingredient fragments that contain the keyword.
    Useful for auditing exactly what phrase triggered a match.
    """
    if not isinstance(ingredients, str):
        return ""
    fragments = re.split(r",|\(|\)", ingredients.lower())
    matches = [f.strip() for f in fragments if keyword in f and f.strip()]
    return "; ".join(matches)


# ─────────────────────────────────────────────
# BOTANICAL SOURCE
# ─────────────────────────────────────────────

BOTANICAL_KEYWORDS = {
    "corn":    "Corn",
    "wheat":   "Wheat",
    "oat":     "Oat",
    "tapioca": "Tapioca",
    "potato":  "Potato",
}


def categorize_botanical(ingredients: str) -> str:
    """
    Identifies botanical source(s) across the full ingredient string.
    Returns a single source, 'Blend', or 'Unknown'.
    """
    if not isinstance(ingredients, str):
        return "Unknown"

    items = [i.strip() for i in ingredients.split(",")]
    sources = set()
    for item in items:
        for keyword, label in BOTANICAL_KEYWORDS.items():
            if keyword in item:
                sources.add(label)

    if len(sources) == 1:
        return list(sources)[0]
    elif len(sources) > 1:
        return "Blend"
    return "Unknown"


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run(search_term: str, release: str, output_folder: str) -> None:
    print("\n" + "="*60)
    print("  USDA Ingredient Search")
    print(f"  Term:    {search_term}")
    print(f"  Release: {release}")
    print("="*60)

    # Load data
    df = load_branded_food(release)

    # Filter to matches
    df_match = filter_by_ingredient(df, search_term)

    if df_match.empty:
        print(f"\n  ⚠  No results found for '{search_term}' in release {release}.")
        print("  Try a broader term or check USDA_DATASETS in usda_utils.py.")
        return

    # Enrich with classification columns
    ingred_col = USDA_COLS["ingred"]
    df_match["starch_type"]       = df_match[ingred_col].apply(classify_starch_type)
    df_match["botanical_source"]  = df_match[ingred_col].apply(categorize_botanical)
    df_match["fuzzy_match"]       = df_match[ingred_col].apply(fuzzy_match_starch)
    df_match["matching_phrases"]  = df_match[ingred_col].apply(
        lambda x: extract_matching_phrases(x, keyword=search_term.split()[0])
    )

    # Select and order output columns (only include cols that exist)
    desired_cols = [
        USDA_COLS["upc"],
        USDA_COLS["owner"],
        USDA_COLS["brand"],
        USDA_COLS["desc"],
        USDA_COLS["category"],
        USDA_COLS["ingred"],
        USDA_COLS["srv_size"],
        USDA_COLS["srv_unit"],
        "starch_type",
        "botanical_source",
        "fuzzy_match",
        "matching_phrases",
    ]
    output_cols = [c for c in desired_cols if c in df_match.columns]
    df_out = df_match[output_cols].reset_index(drop=True)

    # Enrich with usage tier (FDA position-based estimation)
    from ingredient_position import enrich_with_usage
    df_out = enrich_with_usage(df_out, search_term)

    # QA + save
    validate_output(df_out, f"Ingredient Search — '{search_term}'")
    safe_term = search_term.replace(" ", "_")
    save_output(df_out, f"ingredient_search_{safe_term}.csv", output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Search USDA branded food data for a target ingredient."
    )
    parser.add_argument(
        "--term",
        type=str,
        default=DEFAULT_SEARCH_TERM,
        help=f"Ingredient string to search for (default: '{DEFAULT_SEARCH_TERM}')"
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
        help=f"Output folder for CSVs (default: '{DEFAULT_OUTPUT}'). "
             "Can be a OneDrive/SharePoint path."
    )
    args = parser.parse_args()
    run(args.term, args.release, args.output_folder)
