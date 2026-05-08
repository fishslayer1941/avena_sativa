"""
Tests for ingredient_position.py

Run with: pytest usda_pipeline/tests/ -v
"""

import pandas as pd
import pytest

from ingredient_position import (
    enrich_with_usage,
    estimate_usage_tier,
    find_ingredient_position,
    get_usage_context,
    parse_ingredients,
)


# ─────────────────────────────────────────────
# parse_ingredients
# ─────────────────────────────────────────────

def test_parse_ingredients_basic():
    raw = "water, sugar, oat fiber, salt"
    result, trace = parse_ingredients(raw)
    assert result == ["water", "sugar", "oat fiber", "salt"]
    assert trace == set()


def test_parse_ingredients_parenthetical():
    raw = "whole grain oats (oat flour, oat bran), sugar"
    result, trace = parse_ingredients(raw)
    assert "whole grain oats" in result
    assert "oat flour" in result
    assert "oat bran" in result
    assert "sugar" in result
    # Parent comes before sub-ingredients
    assert result.index("whole grain oats") < result.index("oat flour")
    assert result.index("oat flour") < result.index("sugar")
    assert trace == set()


def test_parse_ingredients_trace_clause():
    raw = "water, sugar, contains 2% or less of: salt, natural flavors"
    result, trace = parse_ingredients(raw)
    assert "water" in result
    assert "sugar" in result
    assert "salt" in result
    assert "natural flavors" in result
    assert "salt" in trace
    assert "natural flavors" in trace
    assert "water" not in trace
    assert "sugar" not in trace


def test_parse_ingredients_empty():
    result, trace = parse_ingredients("")
    assert result == []
    assert trace == set()

    result2, trace2 = parse_ingredients(None)
    assert result2 == []
    assert trace2 == set()


# ─────────────────────────────────────────────
# find_ingredient_position
# ─────────────────────────────────────────────

def test_find_position_found():
    lst = ["water", "oat fiber", "salt", "sugar"]
    assert find_ingredient_position(lst, "oat fiber") == 2


def test_find_position_not_found():
    lst = ["water", "sugar", "salt"]
    assert find_ingredient_position(lst, "oat fiber") is None


def test_find_position_case_insensitive():
    lst = ["water", "OAT FIBER", "salt"]
    # USDA normalizes to lowercase, but function should handle either
    assert find_ingredient_position(lst, "oat fiber") == 2
    assert find_ingredient_position(["water", "oat fiber", "salt"], "OAT FIBER") == 2


# ─────────────────────────────────────────────
# estimate_usage_tier
# ─────────────────────────────────────────────

def test_estimate_tier_trace_clause():
    # in_trace_clause=True always returns Trace regardless of position
    assert estimate_usage_tier(15, 16, True) == "Trace"
    assert estimate_usage_tier(1, 10, True) == "Trace"


def test_estimate_tier_primary():
    # position 1 of 10 → ratio 0.10 ≤ 0.25 → Primary
    assert estimate_usage_tier(1, 10, False) == "Primary"
    # position 2 of 10 → ratio 0.20 ≤ 0.25 → Primary
    assert estimate_usage_tier(2, 10, False) == "Primary"


def test_estimate_tier_secondary():
    # position 5 of 10 → ratio 0.50 ≤ 0.60 → Secondary
    assert estimate_usage_tier(5, 10, False) == "Secondary"
    # position 3 of 10 → ratio 0.30 > 0.25, ≤ 0.60 → Secondary
    assert estimate_usage_tier(3, 10, False) == "Secondary"


def test_estimate_tier_minor():
    # position 8 of 10 → ratio 0.80 ≤ 0.90 → Minor
    assert estimate_usage_tier(8, 10, False) == "Minor"
    # position 7 of 10 → ratio 0.70 > 0.60, ≤ 0.90 → Minor
    assert estimate_usage_tier(7, 10, False) == "Minor"


def test_estimate_tier_trace_position():
    # position 10 of 10 → ratio 1.0 > 0.90 → Trace
    assert estimate_usage_tier(10, 10, False) == "Trace"
    # position 10 of 11 → ratio ~0.91 > 0.90 → Trace
    assert estimate_usage_tier(10, 11, False) == "Trace"


def test_estimate_tier_single_ingredient():
    # Only ingredient → Primary
    assert estimate_usage_tier(1, 1, False) == "Primary"


def test_estimate_tier_unknown():
    assert estimate_usage_tier(None, 10, False) == "Unknown"
    assert estimate_usage_tier(1, 0, False) == "Unknown"


# ─────────────────────────────────────────────
# enrich_with_usage
# ─────────────────────────────────────────────

def _make_df(ingredients_values: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"ingredients": ingredients_values})


def test_enrich_with_usage_columns():
    df = _make_df(["water, oat fiber, sugar, salt"])
    result = enrich_with_usage(df, "oat fiber")
    for col in ["usage_tier", "ingredient_position", "total_ingredients", "usage_context"]:
        assert col in result.columns, f"Missing column: {col}"
    # Original columns preserved
    assert "ingredients" in result.columns
    # Original df not mutated
    assert "usage_tier" not in df.columns


def test_enrich_with_usage_no_match():
    df = _make_df(["water, sugar, salt"])
    result = enrich_with_usage(df, "oat fiber")
    assert result.loc[0, "usage_tier"] == "Unknown"
    assert result.loc[0, "ingredient_position"] is None


def test_enrich_with_usage_primary():
    # "oat fiber" is ingredient #2 of 6 → ratio 0.33 → Secondary
    # Let's use a case where it's clearly Primary: #1 of 4
    df = _make_df(["oat fiber, water, sugar, salt"])
    result = enrich_with_usage(df, "oat fiber")
    assert result.loc[0, "usage_tier"] == "Primary"
    assert result.loc[0, "ingredient_position"] == 1
    assert result.loc[0, "total_ingredients"] == 4


def test_enrich_with_usage_trace_clause():
    raw = "water, sugar, contains 2% or less of: oat fiber, natural flavors"
    df = _make_df([raw])
    result = enrich_with_usage(df, "oat fiber")
    assert result.loc[0, "usage_tier"] == "Trace"
