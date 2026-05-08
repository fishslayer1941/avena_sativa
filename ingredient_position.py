"""
ingredient_position.py
----------------------
Estimates ingredient usage tier based on FDA 21 CFR 101.4 ordering rules.

Ingredients must be declared in descending order of predominance by weight,
so position in the ingredient list is a reliable proxy for relative usage level.

Public API:
    parse_ingredients(raw)              -> (list[str], set[str])
    find_ingredient_position(lst, term) -> int | None
    estimate_usage_tier(pos, total, in_trace_clause) -> str
    get_usage_context(pos, total, tier, in_trace_clause=False) -> str
    enrich_with_usage(df, search_term)  -> pd.DataFrame

Usage:
    from ingredient_position import enrich_with_usage
    df_enriched = enrich_with_usage(df, "oat fiber")
"""

import re
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from usda_utils import USDA_COLS


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

TRACE_PATTERN = re.compile(
    r'(?:'
    r'contains\s+2\s*%\s+or\s+less'   # longest form first to avoid partial match
    r'|2\s*%\s+or\s+less'
    r'|2\s+percent\s+or\s+less'
    r'|less\s+than\s+2\s*%'
    r'|less\s+than\s+1\s*%'
    r'|1\s*%\s+or\s+less'
    r')'
    r'(?:\s+of(?:\s+the\s+following)?)?'   # optional "of" / "of the following"
    r'[\s:,]*',                             # trailing punctuation / whitespace
    re.IGNORECASE,
)

_TIER_DESCS = {
    "Primary":   "likely a primary component by weight",
    "Secondary": "significant secondary ingredient",
    "Minor":     "minor ingredient by weight",
    "Trace":     "small functional quantity",
}


# ─────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────

def _split_top_level(text: str) -> list[str]:
    """Split text on commas that are NOT inside parentheses."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in text:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth = max(depth - 1, 0)
            current.append(ch)
        elif ch == "," and depth == 0:
            token = "".join(current).strip()
            if token:
                parts.append(token)
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _expand_parens(token: str) -> list[str]:
    """Expand a token that may contain parenthetical sub-ingredients.

    'whole grain oats (oat flour, oat bran)' ->
    ['whole grain oats', 'oat flour', 'oat bran']

    Handles up to two levels of nesting recursively.
    """
    token = token.strip()
    if "(" not in token:
        return [token] if token else []

    paren_open = token.index("(")
    # Find matching closing paren (last one)
    paren_close = token.rfind(")")
    if paren_close == -1 or paren_close < paren_open:
        # Malformed — treat whole token as-is
        return [token] if token else []

    parent = token[:paren_open].strip()
    inside = token[paren_open + 1 : paren_close].strip()

    result: list[str] = []
    if parent:
        result.append(parent)

    sub_tokens = _split_top_level(inside)
    for sub in sub_tokens:
        result.extend(_expand_parens(sub))

    # Any text after the closing paren
    tail = token[paren_close + 1 :].strip().lstrip(",").strip()
    if tail:
        result.extend(_expand_parens(tail))

    return result


def _parse_section(text: str) -> list[str]:
    """Parse a section of ingredient text into a flat, cleaned list."""
    top_tokens = _split_top_level(text)
    items: list[str] = []
    for tok in top_tokens:
        items.extend(_expand_parens(tok))
    return [i.strip() for i in items if i.strip()]


# ─────────────────────────────────────────────
# PUBLIC FUNCTIONS
# ─────────────────────────────────────────────

def parse_ingredients(raw: str) -> tuple[list[str], set[str]]:
    """Parse a raw USDA ingredient string into a flat list and trace set.

    Args:
        raw: Raw ingredient string (may be mixed-case; USDA often lowercases).

    Returns:
        (ingredient_list, trace_set) where trace_set contains all ingredient
        strings that appear after a "2% or less" clause marker.
    """
    if not isinstance(raw, str) or not raw.strip():
        return [], set()

    # Normalize: lowercase, strip, replace semicolons with commas
    normalized = raw.lower().strip().replace(";", ",")

    match = TRACE_PATTERN.search(normalized)
    if match:
        pre_text = normalized[: match.start()]
        post_text = normalized[match.end() :].strip()

        pre_list = _parse_section(pre_text)
        post_list = _parse_section(post_text)
        trace_set = set(post_list)
        return pre_list + post_list, trace_set
    else:
        all_items = _parse_section(normalized)
        return all_items, set()


def find_ingredient_position(ingredient_list: list[str], search_term: str) -> int | None:
    """Return 1-based position of the first ingredient matching search_term.

    Matching is case-insensitive substring match.
    Returns None if not found.
    """
    term = search_term.lower().strip()
    for i, ing in enumerate(ingredient_list, 1):
        if term in ing.lower():
            return i
    return None


def estimate_usage_tier(
    position: int | None,
    total: int,
    in_trace_clause: bool,
) -> str:
    """Estimate usage tier from list position.

    Tiers (first match wins):
        Trace     — if in_trace_clause is True
        Primary   — position / total <= 0.25
        Secondary — position / total <= 0.60
        Minor     — position / total <= 0.90
        Trace     — position / total >  0.90
        Unknown   — total == 0 or position is None
    """
    if total == 0 or position is None:
        return "Unknown"
    if in_trace_clause:
        return "Trace"
    if total == 1 and position == 1:
        return "Primary"
    ratio = position / total
    if ratio <= 0.25:
        return "Primary"
    if ratio <= 0.60:
        return "Secondary"
    if ratio <= 0.90:
        return "Minor"
    return "Trace"


def get_usage_context(
    position: int | None,
    total: int,
    tier: str,
    in_trace_clause: bool = False,
) -> str:
    """Return a plain-English explanation of the usage tier for display."""
    if position is None or total == 0:
        return "Unknown — ingredient position could not be determined"
    if in_trace_clause:
        return f"Listed after '2% or less' clause — trace level ({tier})"
    desc = _TIER_DESCS.get(tier, "")
    return f"Ingredient #{position} of {total} — {desc} ({tier})"


def enrich_with_usage(df: pd.DataFrame, search_term: str) -> pd.DataFrame:
    """Add usage tier columns to a DataFrame that contains an ingredients column.

    New columns added:
        usage_tier          str   "Primary" / "Secondary" / "Minor" / "Trace" / "Unknown"
        ingredient_position int   1-based position of the match (or None)
        total_ingredients   int   total number of ingredients in that product
        usage_context       str   plain-English explanation

    Args:
        df:          DataFrame with a USDA_COLS["ingred"] column ("ingredients").
        search_term: Ingredient term to locate in the ingredient list.

    Returns:
        Copy of df with four new columns appended.
    """
    ing_col = USDA_COLS["ingred"]  # "ingredients"
    df = df.copy()

    if ing_col not in df.columns:
        df["usage_tier"] = "Unknown"
        df["ingredient_position"] = None
        df["total_ingredients"] = 0
        df["usage_context"] = "Unknown — ingredient position could not be determined"
        return df

    def _process(raw: str) -> pd.Series:
        ing_list, trace_set = parse_ingredients(raw)
        total = len(ing_list)
        pos = find_ingredient_position(ing_list, search_term)
        in_trace = (pos is not None) and (ing_list[pos - 1] in trace_set)
        tier = estimate_usage_tier(pos, total, in_trace)
        ctx = get_usage_context(pos, total, tier, in_trace)
        return pd.Series({
            "usage_tier":          tier,
            "ingredient_position": pos,
            "total_ingredients":   total,
            "usage_context":       ctx,
        })

    enriched = df[ing_col].apply(_process)
    for col in ["usage_tier", "ingredient_position", "total_ingredients", "usage_context"]:
        df[col] = enriched[col]

    return df
