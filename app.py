"""
Streamlit Dashboard — Google Sheets (Published URL)
- Paste your Google Sheets *published-to-web* URL (the one ending with /pubhtml).
- The app auto-converts it to a downloadable XLSX and loads all sheets.
- Generic KPIs + filters (date/branch) and a couple of charts.
- Ready to deploy from GitHub → Streamlit Cloud.

Author: ChatGPT (for Jaseer)
"""

from __future__ import annotations

import io
import re
import time
import json
import math
import typing as t
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px

# ----------------------------
# Configuration (edit if you like)
# ----------------------------
APP_TITLE = "AlKhair Supermarket — Daily Dashboard"
COMPANY_NAME = "AlKhair Supermarket"
DEFAULT_PUBLISHED_URL = (
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vQG7boLWl2bNLCPR05NXv6EFpPPcFfXsiXPQ7rAGYr3q8Nkc2Ijg8BqEwVofcMLSg/pubhtml"
)

# ----------------------------
# Small helpers
# ----------------------------

def _to_xlsx_download_url(url: str) -> str:
    """Convert a Google Sheets *published* URL to an XLSX download URL.
    Supports links of the form /d/e/{id}/pubhtml → /d/e/{id}/pub?output=xlsx.
    """
    if "docs.google.com/spreadsheets/d/e/" in url:
        return re.sub(r"/pubhtml(.*)$", "/pub?output=xlsx", url.strip())
    return url


def _human(n: float | int | None) -> str:
    if n is None or (isinstance(n, float) and (math.isnan(n) or math.isinf(n))):
        return "–"
    try:
        n = float(n)
    except Exception:
        return str(n)
    absn = abs(n)
    if absn >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    if absn >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if absn >= 1_000:
        return f"{n/1_000:.2f}K"
    return f"{n:,.2f}"


def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in d.columns]
    # Create a lookup map (lowercase, no spaces/punct)
    return d


def _col_finder(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_norm = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in df.columns}
    for want in candidates:
        key = re.sub(r"[^a-z0-9]", "", want.lower())
        # exact key
        if key in cols_norm:
            return cols_norm[key]
        # partial contains match
        for k, orig in cols_norm.items():
            if key in k:
                return orig
    return None


@st.cache_data(show_spinner=False)
def fetch_workbook(published_url: str) -> tuple[list[str], dict[str, pd.DataFrame], str]:
    """Download the XLSX for a published Google Sheet and parse all sheets.

    Returns: (sheet_names, dataframes_by_sheet, last_modified_str)
    """
    dl = _to_xlsx_download_url(published_url)

    # Grab headers for Last-Modified
    last_modified = ""
    try:
        head = requests.head(dl, timeout=20)
        last_modified = head.headers.get("Last-Modified", "")
    except Exception:
        pass

    r = requests.get(dl, timeout=60)
    r.raise_for_status()
    bio = io.BytesIO(r.content)

    xls = pd.ExcelFile(bio)
    sheets = {}
    for sn in xls.sheet_names:
        try:
            df = xls.parse(sn)
            # Drop fully-empty rows/cols
            df = df.dropna(how="all").dropna(axis=1, how="all")
            if not df.empty:
                sheets[sn] = df
        except Exception:
            continue

    return list(sheets.keys()), sheets, last_modified


def kpi_section(df: pd.DataFrame) -> dict[str, float]:
    d = _normalize_cols(df)

    date_col = _col_finder(d, ["date", "trans date", "sales date", "invoice date"])
    if date_col:
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce", dayfirst=True)

    sales_col = _col_finder(d, ["sales", "net sales", "amount", "total", "net amount"])
    profit_col = _col_finder(d, ["profit", "gross profit", "gp"])
    basket_col = _col_finder(d, ["baskets", "tickets", "orders", "invoices", "no of basket", "no of tickets"])

    kpis: dict[str, float] = {"sales": np.nan, "profit": np.nan, "baskets": np.nan, "avg_ticket": np.nan}

    if sales_col and sales_col in d:
        d[sales_col] = _safe_numeric(d[sales_col])
        kpis["sales"] = float(d[sales_col].sum())
    if profit_col and profit_col in d:
        d[profit_col] = _safe_numeric(d[profit_col])
        kpis["profit"] = float(d[profit_col].sum())
    if basket_col and basket_col in d:
        d[basket_col] = _safe_numeric(d[basket_col])
        kpis["baskets"] = float(d[basket_col].sum())

    if not math.isnan(kpis["sales"]) and not math.isnan(kpis["baskets"]) and kpis["baskets"] != 0:
        kpis["avg_ticket"] = kpis["sales"] / kpis["baskets"]

    return kpis


def apply_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str | tuple]]:
    d = _normalize_cols(df)

    # Find columns
    date_col = _col_finder(d, ["date", "trans date", "sales date", "invoice date"])
    branch_col = _col_finder(d, ["branch", "store", "location"])

    # Sidebar filters
    flt_info: dict[str, t.Any] = {}

    if date_col:
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce", dayfirst=True)
        min_d = pd.to_datetime(d[date_col].min()) if not d.empty else None
        max_d = pd.to_datetime(d[date_col].max()) if not d.empty else None
        if pd.notna(min_d) and pd.notna(max_d):
            st.sidebar.subheader("Date filter")
            start, end = st.sidebar.date_input(
                "Range", value=(min_d.date(), max_d.date()), min_value=min_d.date(), max_value=max_d.date()
            )
            if start and end:
                mask = (d[date_col] >= pd.to_datetime(start)) & (d[date_col] <= pd.to_datetime(end))
                d = d.loc[mask]
                flt_info["date_range"] = (str(start), str(end))

    if branch_col and branch_col in d:
        st.sidebar.subheader("Branch filter")
        opts = ["All"] + sorted([x for x in d[branch_col].dropna().astype(str).unique()])
        pick = st.sidebar.selectbox("Branch", opts)
        if pick != "All":
            d = d.loc[d[branch_col].astype(str) == pick]
            flt_info["branch"] = pick

    return d, flt_info


def chart_sales_over_time(df: pd.DataFrame) -> None:
    date_col = _col_finder(df, ["date", "trans date", "sales date", "invoice date"])
    sales_col = _col_finder(df, ["sales", "net sales", "amount", "total", "net amount"])
    if not date_col or not sales_col:
        st.info("Add columns: Date and Sales to enable this chart.")
        return
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce", dayfirst=True)
    d[sales_col] = _safe_numeric(d[sales_col])
    g = d.groupby(d[date_col].dt.date, as_index=False)[sales_col].sum()
    fig = px.line(g, x=g.columns[0], y=sales_col, markers=True, title="Sales over time")
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), hovermode="x unified", showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def chart_sales_by_branch(df: pd.DataFrame) -> None:
    branch_col = _col_finder(df, ["branch", "store", "location"])
    sales_col = _col_finder(df, ["sales", "net sales", "amount", "total", "net amount"])
    if not branch_col or not sales_col:
        st.info("Add columns: Branch and Sales to enable this chart.")
        return
    d = df.copy()
    d[sales_col] = _safe_numeric(d[sales_col])
    g = d.groupby(branch_col, as_index=False)[sales_col].sum().sort_values(sales_col, ascending=False)
    fig = px.bar(g, x=branch_col, y=sales_col, title="Sales by Branch")
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), hovermode="x", showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def chart_top_items(df: pd.DataFrame, top_n: int = 10) -> None:
    item_col = _col_finder(df, ["item", "product", "sku", "description", "item name"]) 
    sales_col = _col_finder(df, ["sales", "net sales", "amount", "total", "net amount"])
    if not item_col or not sales_col:
        st.info("Add columns: Item and Sales to enable this chart.")
        return
    d = df.copy()
    d[sales_col] = _safe_numeric(d[sales_col])
    g = (
        d.groupby(item_col, as_index=False)[sales_col]
        .sum()
        .sort_values(sales_col, ascending=False)
        .head(top_n)
    )
    fig = px.bar(g, x=item_col, y=sales_col, title=f"Top {top_n} Items by Sales")
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), hovermode="x", showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ----------------------------
# UI
# ----------------------------

def main() -> None:
    st.set_page_config(page_title="Google Sheets Dashboard", layout="wide")

    # Header
    left, right = st.columns([0.8, 0.2], gap="large")
    with left:
        st.title(APP_TITLE)
        st.caption(f"Data source: Google Sheets (published). Company: {COMPANY_NAME}")
    with right:
        st.image(
            "https://static.streamlit.io/examples/dice.jpg",
            caption="Streamlit",
            use_column_width=True,
        )

    # Sidebar — data source + refresh controls
    st.sidebar.header("Controls")
    url = st.sidebar.text_input("Published Google Sheets URL", value=DEFAULT_PUBLISHED_URL)

    auto = st.sidebar.toggle("Auto-refresh", value=False, help="Reload data automatically")
    every = st.sidebar.slider("Interval (seconds)", 15, 600, 60, help="How often to refresh when Auto is ON")
    if auto:
        st.sidebar.caption("Auto-refreshing…")
        st.experimental_set_query_params(_=str(int(time.time())))  # Bust cache in URL
        st.autorefresh(interval=every * 1000, key="_autorefresh_key")

    if st.sidebar.button("Reload now", use_container_width=True):
        fetch_workbook.clear()  # clear cache
        st.experimental_rerun()

    # Load workbook
    with st.spinner("Downloading workbook and reading sheets…"):
        try:
            sheet_names, dfs, last_mod = fetch_workbook(url)
        except Exception as e:
            st.error(
                "Couldn't load the published Google Sheet. Please confirm the URL is public (File → Share → Publish to web).\n\n" + str(e)
            )
            return

    if not dfs:
        st.warning("No non-empty sheets found in the workbook.")
        return

    # Choose a sheet
    st.subheader("Data selection")
    pick = st.selectbox("Sheet", sheet_names, index=0)
    df = dfs[pick].copy()

    # Show metadata
    meta_cols = st.columns(3)
    with meta_cols[0]:
        st.metric("Rows", f"{len(df):,}")
    with meta_cols[1]:
        st.metric("Columns", f"{len(df.columns):,}")
    with meta_cols[2]:
        ts = last_mod or datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        st.metric("Last Modified (server)", ts)

    # Filters
    df_filtered, flt_info = apply_filters(df)

    # KPIs
    st.subheader("Key Metrics")
    kpi = kpi_section(df_filtered)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sales", _human(kpi.get("sales")))
    c2.metric("Total Profit", _human(kpi.get("profit")))
    c3.metric("Baskets / Tickets", _human(kpi.get("baskets")))
    c4.metric("Avg Ticket", _human(kpi.get("avg_ticket")))

    # Quick insights
    with st.expander("Quick insights"):
        msgs = []
        if not math.isnan(kpi.get("profit", float("nan"))) and kpi.get("profit", 0) < 0:
            msgs.append("⚠️ Profit is negative on the current selection.")
        if not math.isnan(kpi.get("sales", float("nan"))) and kpi.get("sales", 0) == 0:
            msgs.append("ℹ️ Sales are 0 for the selected range — check filters.")
        if not msgs:
            st.write("Looks good! No alerts based on the available columns.")
        else:
            for m in msgs:
                st.write(m)

    # Charts
    st.subheader("Charts")
    chart_cols = st.columns(2)
    with chart_cols[0]:
        chart_sales_over_time(df_filtered)
    with chart_cols[1]:
        chart_sales_by_branch(df_filtered)

    chart_top_items(df_filtered, top_n=10)

    # Data table
    st.subheader("Data table (filtered)")
    st.dataframe(df_filtered, use_container_width=True)

    # Download filtered CSV
    csv = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", data=csv, file_name=f"{pick}_filtered.csv", mime="text/csv")

    st.caption("Built with ❤️ in Streamlit. Update your Google Sheet and refresh here to see changes.")


if __name__ == "__main__":
    main()
