"""
app.py
------
USDA Ingredient Intelligence — Streamlit web app.

Three pages:
  1. Ingredient Search  — plain ingredient substring search
  2. Nielsen GTIN Decoder — ingredient filter + Nielsen UPC transform
  3. Change Detection   — year-over-year new products / formula changes

Run:
    streamlit run app.py
"""

import os
import re
import sys
from datetime import date

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# Ensure local modules are importable regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from usda_utils import (
    USDA_COLS,
    USDA_DATASETS,
    filter_by_ingredient,
    load_branded_food,
    save_output,
)
from ingredient_dec import modify_gtin
from change_detection import detect_changes, format_output
from ingredient_position import enrich_with_usage

TIER_COLORS = {
    "Primary":   "#2d6a4f",
    "Secondary": "#52b788",
    "Minor":     "#f4a261",
    "Trace":     "#e76f51",
    "Unknown":   "#adb5bd",
}


# ─────────────────────────────────────────────
# SSL BYPASS — corporate proxy support
# ─────────────────────────────────────────────

import urllib3

_ORIG_GET = requests.get  # captured once at import time


def _apply_ssl_patch(enabled: bool) -> None:
    """Monkey-patch requests.get to skip SSL verification when enabled.
    usda_utils calls requests.get at call time, so patching the module
    object here propagates into it without modifying that file.
    """
    if enabled:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        requests.get = lambda url, **kw: _ORIG_GET(url, verify=kw.pop("verify", False), **kw)
    else:
        requests.get = _ORIG_GET


# ─────────────────────────────────────────────
# CACHED LOADER
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_dataset(release_key: str) -> pd.DataFrame:
    return load_branded_food(release_key)


def _load_and_track(release_key: str) -> pd.DataFrame:
    """Load a dataset and record it in session_state.cached_releases."""
    df = _load_dataset(release_key)
    st.session_state.cached_releases.add(release_key)
    return df


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [v for v in USDA_COLS.values() if v not in df.columns]
    if missing:
        st.warning(
            f"Missing expected columns: {missing}. "
            "This release may have a schema change."
        )


def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _date_str() -> str:
    return date.today().strftime("%Y-%m-%d")


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

def _render_sidebar() -> str:
    """Render sidebar and return the user-configured output folder."""
    with st.sidebar:
        st.title("USDA Ingredient Intelligence")
        st.caption("Search and track ingredients in USDA FoodData Central")
        st.markdown("[USDA FoodData Central](https://fdc.nal.usda.gov)")
        st.divider()

        if "output_folder" not in st.session_state:
            st.session_state.output_folder = "output"

        output_folder = st.text_input(
            "Output folder",
            value=st.session_state.output_folder,
            help="Local path or OneDrive/SharePoint path for CSV exports.",
            key="output_folder_input",
        )
        st.session_state.output_folder = output_folder
        st.divider()

        cached = st.session_state.get("cached_releases", set())
        if cached:
            st.caption("Cached in memory:")
            for k in sorted(cached):
                st.success(f"  {k}", icon="📦")
        else:
            st.caption("No datasets cached yet.")
        st.divider()

        # SSL bypass toggle for corporate proxy environments
        ssl_bypass = st.checkbox(
            "Skip SSL certificate verification",
            value=st.session_state.get("ssl_bypass", False),
            help="Enable if downloads fail with a certificate error (corporate proxy).",
            key="ssl_bypass_toggle",
        )
        st.session_state.ssl_bypass = ssl_bypass
        if ssl_bypass:
            st.warning("SSL verification disabled. Use only on trusted networks.", icon="⚠️")

        if st.button("Clear cached data", key="clear_cache"):
            _load_dataset.clear()
            st.session_state.cached_releases = set()
            st.success("Cache cleared — next search will re-download data.")
        st.divider()

        # Ingredient Search filters (shown only when results exist)
        results = st.session_state.get("search1_results")
        if results is not None and not results.empty:
            st.subheader("Ingredient Search Filters")
            cat_col = USDA_COLS["category"]
            cat_opts = sorted(results[cat_col].dropna().unique())
            selected_cat = st.multiselect(
                "Category", options=cat_opts, default=cat_opts,
                key="filter_category"
            )
            tier_opts = ["Primary", "Secondary", "Minor", "Trace", "Unknown"]
            selected_tiers = st.multiselect(
                "Usage Tier", options=tier_opts, default=tier_opts,
                key="filter_tier"
            )
            st.session_state.search1_filters = {
                "category": selected_cat,
                "tier":     selected_tiers,
            }

    return output_folder


# ─────────────────────────────────────────────
# PAGE 1 — INGREDIENT SEARCH
# ─────────────────────────────────────────────

def page_ingredient_search(output_folder: str) -> None:
    st.header("Ingredient Search")
    st.caption("Search for USDA branded food products containing a target ingredient.")

    col1, col2 = st.columns([3, 1])
    with col1:
        search_term = st.text_input(
            "Search ingredient", value="resistant starch", key="s1_term"
        )
    with col2:
        release_keys = list(USDA_DATASETS.keys())
        release = st.selectbox(
            "USDA Release", options=release_keys, index=0, key="s1_release"
        )

    run_search = st.button("Search", key="s1_run")

    if run_search:
        if not search_term.strip():
            st.warning("Please enter a search term.")
            return
        try:
            with st.spinner("Searching USDA database..."):
                df = _load_and_track(release)
                _validate_columns(df)
                df_match = filter_by_ingredient(df, search_term)

            if df_match.empty:
                st.info(
                    f"No results for **'{search_term}'** in release {release}. "
                    "Try a shorter or broader term."
                )
                st.session_state.search1_results = pd.DataFrame()
                return

            df_match = enrich_with_usage(df_match, search_term)
            desired_cols = [
                USDA_COLS["upc"],
                USDA_COLS["owner"],
                USDA_COLS["brand"],
                USDA_COLS["desc"],
                USDA_COLS["category"],
                USDA_COLS["ingred"],
                USDA_COLS["srv_size"],
                USDA_COLS["srv_unit"],
                "usage_tier",
                "ingredient_position",
                "total_ingredients",
                "usage_context",
            ]
            output_cols = [c for c in desired_cols if c in df_match.columns]
            st.session_state.search1_results = df_match[output_cols].reset_index(drop=True)
            st.session_state.search1_meta = {"term": search_term, "release": release}

        except requests.exceptions.SSLError:
            st.error(
                "SSL certificate verification failed. "
                "If you are on a corporate network, enable "
                "**Skip SSL certificate verification** in the sidebar and try again."
            )
            return
        except requests.HTTPError as e:
            st.error(
                f"Download failed: {getattr(e.response, 'url', str(e))}\n"
                "Check your internet connection and try again."
            )
            return
        except KeyError as e:
            st.error(f"Missing column: {e}. The release may have a schema change.")
            return
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            return

    results = st.session_state.get("search1_results")
    meta = st.session_state.get("search1_meta", {})

    if results is None or results.empty:
        return

    # Apply sidebar filters
    filters = st.session_state.get("search1_filters", {})
    filtered = results.copy()
    if filters.get("category"):
        cat_col = USDA_COLS["category"]
        filtered = filtered[filtered[cat_col].isin(filters["category"])]
    if filters.get("tier") and "usage_tier" in filtered.columns:
        filtered = filtered[filtered["usage_tier"].isin(filters["tier"])]

    term = meta.get("term", "")
    rel = meta.get("release", "")
    st.markdown(
        f"**Found {len(filtered):,} products** containing **'{term}'** "
        f"(USDA release {rel})"
    )

    owner_col = USDA_COLS["owner"]
    cat_col = USDA_COLS["category"]

    if "usage_tier" in filtered.columns:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total matches", f"{len(filtered):,}")
        c2.metric("Unique brand owners", f"{filtered[owner_col].nunique():,}")
        c3.metric("Unique categories", f"{filtered[cat_col].nunique():,}")
        ps_count = int(filtered["usage_tier"].isin(["Primary", "Secondary"]).sum())
        tr_count = int((filtered["usage_tier"] == "Trace").sum())
        c4.metric("Primary / Secondary", f"{ps_count:,}")
        c5.metric("Trace", f"{tr_count:,}")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total matches", f"{len(filtered):,}")
        c2.metric("Unique brand owners", f"{filtered[owner_col].nunique():,}")
        c3.metric("Unique categories", f"{filtered[cat_col].nunique():,}")

    # Build display DataFrame: combine position/total and truncate context
    display_df = filtered.copy()
    if "ingredient_position" in display_df.columns:
        display_df["pos / total"] = display_df.apply(
            lambda r: (
                f"{int(r['ingredient_position'])} / {int(r['total_ingredients'])}"
                if pd.notna(r["ingredient_position"]) else "—"
            ),
            axis=1,
        )
        display_df["usage_context"] = display_df["usage_context"].str[:60]
        display_df = display_df.drop(columns=["ingredient_position", "total_ingredients"])

    def _style_tier(val: str) -> str:
        color = TIER_COLORS.get(val, "")
        return f"background-color: {color}; color: white" if color else ""

    if "usage_tier" in display_df.columns:
        styled = display_df.style.map(_style_tier, subset=["usage_tier"])
        st.dataframe(styled, use_container_width=True)
    else:
        st.dataframe(display_df, use_container_width=True)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        top_owners = (
            filtered[owner_col]
            .value_counts()
            .head(15)
            .reset_index()
        )
        top_owners.columns = ["brand_owner", "count"]
        fig = px.bar(
            top_owners,
            x="count",
            y="brand_owner",
            orientation="h",
            title="Top 15 Brand Owners",
            labels={"count": "Products", "brand_owner": ""},
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        if "usage_tier" in filtered.columns:
            tier_order = ["Primary", "Secondary", "Minor", "Trace", "Unknown"]
            tier_dist = (
                filtered["usage_tier"]
                .value_counts()
                .reindex(tier_order, fill_value=0)
                .reset_index()
            )
            tier_dist.columns = ["usage_tier", "count"]
            fig2 = px.bar(
                tier_dist,
                x="count",
                y="usage_tier",
                orientation="h",
                color="usage_tier",
                color_discrete_map=TIER_COLORS,
                title="Usage Tier Distribution",
                labels={"count": "Products", "usage_tier": ""},
            )
            fig2.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig2, use_container_width=True)

    # ─────────────────────────────────────────────
    # CHART 4: USAGE TIER × FOOD CATEGORY HEATMAP
    # ─────────────────────────────────────────────
    st.divider()
    
    if "usage_tier" in filtered.columns:
        # Pivot table: rows = category, columns = usage_tier
        pivot_data = pd.crosstab(
            filtered[cat_col],
            filtered["usage_tier"],
            margins=False
        )
        
        # Ensure all tier columns exist (fill missing with 0)
        tier_order = ["Primary", "Secondary", "Minor", "Trace", "Unknown"]
        for tier in tier_order:
            if tier not in pivot_data.columns:
                pivot_data[tier] = 0
        pivot_data = pivot_data[tier_order]
        
        # Get top 20 categories by total count
        pivot_data["_total"] = pivot_data.sum(axis=1)
        pivot_data = pivot_data.nlargest(20, "_total").drop(columns=["_total"])
        pivot_data = pivot_data.sort_values(tier_order[0], ascending=False)
        
        # Truncate category names to 35 chars
        pivot_data.index = [cat[:35] + "..." if len(cat) > 35 else cat for cat in pivot_data.index]
        
        # Create heatmap with zeros hidden
        heatmap_data = pivot_data.copy()
        heatmap_data = heatmap_data.mask(heatmap_data == 0)
        
        fig4 = px.imshow(
            heatmap_data,
            labels=dict(x="Usage Tier", y="", color="Product Count"),
            color_continuous_scale="Blues",
            title="Ingredient Concentration by Food Category",
            text_auto=True,
            aspect="auto",
            height=600,
        )
        fig4.update_layout(
            xaxis_title="Usage Tier",
            yaxis_title="",
            coloraxis_colorbar=dict(title="Product Count"),
        )
        
        with st.expander("📊 Concentration by Category", expanded=True):
            st.markdown("*Top 20 categories — darker = more products with ingredient at that concentration level*")
            st.plotly_chart(fig4, use_container_width=True)

    # ─────────────────────────────────────────────
    # CHART 5: TOP 15 BRANDS × USAGE TIER STACKED BAR
    # ─────────────────────────────────────────────
    st.divider()
    
    owner_col = USDA_COLS["owner"]
    
    # Get top 15 brands by total product count
    top_brands = (
        filtered[owner_col]
        .value_counts()
        .head(15)
        .index.tolist()
    )
    
    # Filter to top brands and create stacked data
    top_brands_df = filtered[filtered[owner_col].isin(top_brands)].copy()
    
    # Crosstab: brands × usage_tier
    brand_tier_pivot = pd.crosstab(
        top_brands_df[owner_col],
        top_brands_df["usage_tier"],
        margins=False
    )
    
    # Ensure all tier columns exist and are in correct order
    tier_order = ["Primary", "Secondary", "Minor", "Trace", "Unknown"]
    for tier in tier_order:
        if tier not in brand_tier_pivot.columns:
            brand_tier_pivot[tier] = 0
    brand_tier_pivot = brand_tier_pivot[tier_order]
    
    # Sort by total count descending
    brand_tier_pivot["_total"] = brand_tier_pivot.sum(axis=1)
    brand_tier_pivot = brand_tier_pivot.sort_values("_total", ascending=True).drop(columns=["_total"])
    
    # Reshape for stacked bar chart
    brand_tier_long = brand_tier_pivot.reset_index().melt(
        id_vars=[owner_col],
        value_vars=tier_order,
        var_name="usage_tier",
        value_name="count"
    )
    
    fig5 = px.bar(
        brand_tier_long,
        x="count",
        y=owner_col,
        color="usage_tier",
        color_discrete_map=TIER_COLORS,
        orientation="h",
        title="Top 15 Brands by Ingredient Concentration",
        labels={"count": "Number of Products", owner_col: ""},
        barmode="stack",
        height=500,
    )
    fig5.update_layout(
        legend=dict(title="Usage Tier"),
        xaxis_title="Number of Products",
        yaxis_title="",
    )
    st.plotly_chart(fig5, use_container_width=True)

    # ─────────────────────────────────────────────
    # CHART 3: CATEGORY DISTRIBUTION (RANKED) — TOP 25
    # ─────────────────────────────────────────────
    st.divider()
    cat_col = USDA_COLS["category"]
    cat_dist = (
        filtered[cat_col]
        .value_counts()
        .head(25)
        .reset_index()
    )
    cat_dist.columns = ["category", "count"]
    cat_dist = cat_dist.sort_values("count", ascending=True)  # Ascending for horizontal bar
    
    chart_height_3 = max(400, len(cat_dist) * 28)
    
    fig3 = px.bar(
        cat_dist,
        x="count",
        y="category",
        orientation="h",
        title="Products by Food Category (Top 25)",
        labels={"count": "Number of Products", "category": ""},
    )
    fig3.update_traces(text=cat_dist["count"], textposition="outside")
    fig3.update_layout(height=chart_height_3, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig3, use_container_width=True)

    safe_term = term.replace(" ", "_")
    filename = f"ingredient_search_{safe_term}_{_date_str()}.csv"
    csv = _csv_bytes(filtered)

    dl_col, _ = st.columns([1, 3])
    with dl_col:
        if st.download_button(
            label="Download CSV",
            data=csv,
            file_name=filename,
            mime="text/csv",
            key="s1_download",
        ):
            try:
                save_output(filtered, f"ingredient_search_{safe_term}.csv", output_folder)
            except Exception as e:
                st.warning(f"Could not save to output folder: {e}")


# ─────────────────────────────────────────────
# PAGE 2 — NIELSEN GTIN DECODER
# ─────────────────────────────────────────────

def page_gtin_decoder(output_folder: str) -> None:
    st.header("Nielsen GTIN Decoder")
    st.caption(
        "Filter products by ingredient and transform the USDA GTIN/UPC "
        "into Nielsen's 12-digit UPC-A format for data joins."
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        search_term = st.text_input(
            "Ingredient to decode", value="oat", key="s2_term"
        )
    with col2:
        release_keys = list(USDA_DATASETS.keys())
        release = st.selectbox(
            "USDA Release", options=release_keys, index=0, key="s2_release"
        )

    use_word_boundary = st.checkbox(
        "Use word-boundary matching",
        value=True,
        help=(
            'Prevents "oatmeal" from matching when searching "oat". '
            "Recommended for short single-word terms."
        ),
        key="s2_wb",
    )

    run_decode = st.button("Run", key="s2_run")

    if run_decode:
        if not search_term.strip():
            st.warning("Please enter an ingredient term.")
            return
        try:
            with st.spinner("Loading USDA database..."):
                df = _load_and_track(release)
                _validate_columns(df)

            ing_col = USDA_COLS["ingred"]
            upc_col = USDA_COLS["upc"]

            with st.spinner("Filtering ingredients..."):
                if use_word_boundary:
                    pattern = rf"\b{re.escape(search_term.lower())}"
                    mask = df[ing_col].str.contains(
                        pattern, case=False, na=False, regex=True
                    )
                    df_match = df.loc[mask].copy()
                else:
                    df_match = filter_by_ingredient(df, search_term)

            if df_match.empty:
                st.info(
                    f"No results for **'{search_term}'**. "
                    "Try unchecking word-boundary matching or broadening the term."
                )
                return

            df_match["modified_gtin_upc"] = df_match[upc_col].apply(modify_gtin)

            desired_cols = [
                upc_col,
                "modified_gtin_upc",
                USDA_COLS["owner"],
                USDA_COLS["brand"],
                USDA_COLS["desc"],
                USDA_COLS["category"],
                USDA_COLS["ingred"],
                USDA_COLS["srv_size"],
                USDA_COLS["srv_unit"],
            ]
            output_cols = [c for c in desired_cols if c in df_match.columns]
            df_out = df_match[output_cols].reset_index(drop=True)

            df_out = enrich_with_usage(df_out, search_term)
            st.session_state.search2_results = df_out
            st.session_state.search2_meta = {"term": search_term, "release": release}

        except requests.exceptions.SSLError:
            st.error(
                "SSL certificate verification failed. "
                "Enable **Skip SSL certificate verification** in the sidebar and try again."
            )
            return
        except requests.HTTPError as e:
            st.error(
                f"Download failed: {getattr(e.response, 'url', str(e))}\n"
                "Check your internet connection."
            )
            return
        except KeyError as e:
            st.error(f"Missing column: {e}. The release may have a schema change.")
            return
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            return

    results = st.session_state.get("search2_results")
    meta = st.session_state.get("search2_meta", {})

    if results is None or results.empty:
        return

    term = meta.get("term", "")
    null_count = int(results["modified_gtin_upc"].isnull().sum())

    m1, m2 = st.columns(2)
    m1.metric("Products matched", f"{len(results):,}")
    if null_count > 0:
        m2.metric("Null Nielsen GTINs", f"{null_count:,}", delta="⚠ check GTINs", delta_color="inverse")
        st.error(
            f"{null_count:,} rows have a null **modified_gtin_upc** "
            "(malformed or missing source UPCs)."
        )
    else:
        m2.metric("Null Nielsen GTINs", "0")

    upc_col = USDA_COLS["upc"]
    st.subheader("Nielsen Join Key — sample (first 5 rows)")
    sample_cols = [upc_col, "modified_gtin_upc"]
    available_sample = [c for c in sample_cols if c in results.columns]
    sample = results[available_sample].head(5).copy()
    sample.columns = ["Original GTIN (USDA)", "Modified GTIN (Nielsen)"][: len(available_sample)]
    sc1, sc2 = st.columns(2)
    sc1.dataframe(
        results[[upc_col]].head(5).rename(columns={upc_col: "Original GTIN (USDA)"}),
        use_container_width=True,
        hide_index=True,
    )
    sc2.dataframe(
        results[["modified_gtin_upc"]].head(5).rename(
            columns={"modified_gtin_upc": "Modified GTIN — Nielsen Join Key"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(f"All results — {len(results):,} rows")
    display = results.copy()
    if "ingredient_position" in display.columns:
        display["pos / total"] = display.apply(
            lambda r: (
                f"{int(r['ingredient_position'])} / {int(r['total_ingredients'])}"
                if pd.notna(r["ingredient_position"]) else "—"
            ),
            axis=1,
        )
        display = display.drop(columns=["ingredient_position", "total_ingredients", "usage_context"])
    display = display.rename(columns={"modified_gtin_upc": "modified_gtin_upc (Nielsen Join Key)"})
    st.dataframe(display, use_container_width=True)

    safe_term = term.replace(" ", "_")
    filename = f"ingredient_dec_{safe_term}_{_date_str()}.csv"
    csv = _csv_bytes(results)

    dl_col, _ = st.columns([1, 3])
    with dl_col:
        if st.download_button(
            label="Download CSV",
            data=csv,
            file_name=filename,
            mime="text/csv",
            key="s2_download",
        ):
            try:
                save_output(results, f"ingredient_dec_{safe_term}.csv", output_folder)
            except Exception as e:
                st.warning(f"Could not save to output folder: {e}")


# ─────────────────────────────────────────────
# PAGE 3 — CHANGE DETECTION
# ─────────────────────────────────────────────

def _change_section(
    label: str,
    slug: str,
    df: pd.DataFrame,
    include_old: bool,
    key_prefix: str,
    output_folder: str,
    safe_term: str,
    search_term: str = "",
) -> None:
    """Render one expandable change-type section."""
    count = len(df)
    with st.expander(f"{label} ({count:,})", expanded=count > 0):
        if count == 0:
            st.info(f"No {label.lower()} records found for this ingredient and release pair.")
            return
        st.metric("Records", f"{count:,}")

        display = format_output(df, include_old_ing=include_old)
        if search_term:
            display = enrich_with_usage(display, search_term)

        owner_col = USDA_COLS["owner"]
        if owner_col in display.columns:
            display = display.sort_values(owner_col)

        # Warn if any product added the ingredient as a primary component
        if "➕" in label and "usage_tier" in display.columns:
            primary_n = int((display["usage_tier"] == "Primary").sum())
            if primary_n > 0:
                st.warning(
                    f"⚠️ {primary_n} product(s) added this ingredient "
                    "as a **Primary** component"
                )

        st.dataframe(display, use_container_width=True)

        filename = f"{slug}_{safe_term}_{_date_str()}.csv"
        dl_col, _ = st.columns([1, 3])
        with dl_col:
            if st.download_button(
                label=f"Download {label}",
                data=_csv_bytes(display),
                file_name=filename,
                mime="text/csv",
                key=f"{key_prefix}_dl",
            ):
                try:
                    save_output(display, f"{slug}_{safe_term}.csv", output_folder)
                except Exception as e:
                    st.warning(f"Could not save to output folder: {e}")


def page_change_detection(output_folder: str) -> None:
    st.header("Change Detection (Year-over-Year)")
    st.caption(
        "Compare two USDA releases to find new products and formula changes "
        "for a target ingredient."
    )

    search_term = st.text_input(
        "Ingredient to track", value="oat fiber", key="s3_term"
    )

    release_keys = list(USDA_DATASETS.keys())
    rc1, rc2 = st.columns(2)
    with rc1:
        new_release = st.selectbox(
            "New Release", options=release_keys, index=0, key="s3_new"
        )
    with rc2:
        old_index = 1 if len(release_keys) > 1 else 0
        old_release = st.selectbox(
            "Old Release", options=release_keys, index=old_index, key="s3_old"
        )

    run_detect = st.button("Run", key="s3_run")

    if run_detect:
        if not search_term.strip():
            st.warning("Please enter an ingredient term.")
            return
        if new_release == old_release:
            st.error(
                "New Release and Old Release must be different. "
                "Please select two distinct USDA releases."
            )
            return
        try:
            with st.status("Loading datasets and comparing...", expanded=True) as status:
                status.update(label=f"Loading {new_release} dataset...")
                df_new = _load_and_track(new_release)
                _validate_columns(df_new)

                status.update(label=f"Loading {old_release} dataset...")
                df_old = _load_and_track(old_release)
                _validate_columns(df_old)

                status.update(label="Comparing releases...")
                new_products, formula_added, formula_removed = detect_changes(
                    df_old, df_new, search_term
                )
                status.update(label="Done.", state="complete")

            st.session_state.search3_results = {
                "new_products": new_products,
                "formula_added": formula_added,
                "formula_removed": formula_removed,
            }
            st.session_state.search3_meta = {
                "term": search_term,
                "new_release": new_release,
                "old_release": old_release,
            }

        except requests.exceptions.SSLError:
            st.error(
                "SSL certificate verification failed. "
                "Enable **Skip SSL certificate verification** in the sidebar and try again."
            )
            return
        except requests.HTTPError as e:
            st.error(
                f"Download failed: {getattr(e.response, 'url', str(e))}\n"
                "Check your internet connection."
            )
            return
        except KeyError as e:
            st.error(f"Missing column: {e}. The release may have a schema change.")
            return
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            return

    results = st.session_state.get("search3_results")
    meta = st.session_state.get("search3_meta", {})

    if results is None:
        return

    term = meta.get("term", "")
    new_rel = meta.get("new_release", "")
    old_rel = meta.get("old_release", "")
    safe_term = term.replace(" ", "_")

    st.markdown(
        f"Showing changes for **'{term}'** between "
        f"**{new_rel}** (new) and **{old_rel}** (old)"
    )

    new_products = results["new_products"]
    formula_added = results["formula_added"]
    formula_removed = results["formula_removed"]

    _change_section("📦 New Products",  "new_products",    new_products,    False, "s3_np", output_folder, safe_term, search_term)
    _change_section("➕ Formula Added",  "formula_added",   formula_added,   True,  "s3_fa", output_folder, safe_term, search_term)
    _change_section("➖ Formula Removed", "formula_removed", formula_removed, True,  "s3_fr", output_folder, safe_term, search_term)

    # Combined download
    non_empty = [
        format_output(df, inc)
        for df, inc in [
            (new_products, False),
            (formula_added, True),
            (formula_removed, True),
        ]
        if not df.empty
    ]
    if non_empty:
        all_changes = pd.concat(non_empty, ignore_index=True)
        all_filename = f"all_changes_{safe_term}_{_date_str()}.csv"
        st.divider()
        dl_col, _ = st.columns([1, 3])
        with dl_col:
            if st.download_button(
                label="Download All Changes",
                data=_csv_bytes(all_changes),
                file_name=all_filename,
                mime="text/csv",
                key="s3_all_dl",
            ):
                try:
                    save_output(all_changes, f"all_changes_{safe_term}.csv", output_folder)
                except Exception as e:
                    st.warning(f"Could not save to output folder: {e}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="USDA Ingredient Intelligence",
        page_icon="🌾",
        layout="wide",
    )

    if "cached_releases" not in st.session_state:
        st.session_state.cached_releases = set()

    output_folder = _render_sidebar()
    _apply_ssl_patch(st.session_state.get("ssl_bypass", False))

    tab1, tab2, tab3 = st.tabs(
        ["🔍 Ingredient Search", "🔗 Nielsen GTIN Decoder", "📊 Change Detection"]
    )

    with tab1:
        page_ingredient_search(output_folder)

    with tab2:
        page_gtin_decoder(output_folder)

    with tab3:
        page_change_detection(output_folder)


if __name__ == "__main__":
    main()
