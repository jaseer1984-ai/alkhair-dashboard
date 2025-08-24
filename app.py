from __future__ import annotations

import io
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

# =========================================
# CONFIG
# =========================================
class Config:
    PAGE_TITLE = "Al Khair Business Performance"
    LAYOUT = "wide"
    # Default published URL (kept; can be replaced via the data-source expander)
    DEFAULT_PUBLISHED_URL = (
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vQG7boLWl2bNLCPR05NXv6EFpPPcFfXsiXPQ7rAGYr3q8Nkc2Ijg8BqEwVofcMLSg/pubhtml"
    )
    BRANCHES = {
        "Al khair - 102": {"name": "Al Khair", "code": "102", "color": "#3b82f6"},
        "Noora - 104": {"name": "Noora", "code": "104", "color": "#10b981"},
        "Hamra - 109": {"name": "Hamra", "code": "109", "color": "#f59e0b"},
        "Magnus - 107": {"name": "Magnus", "code": "107", "color": "#8b5cf6"},
    }
    TARGETS = {"sales_achievement": 95.0, "nob_achievement": 90.0, "abv_achievement": 90.0}


config = Config()
st.set_page_config(page_title=config.PAGE_TITLE, layout=config.LAYOUT, initial_sidebar_state="expanded")


# =========================================
# CSS (soft pills + light cards + tidy UI)
# =========================================
def apply_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&display=swap');
        html, body, [data-testid="stAppViewContainer"] { font-family: 'Inter', sans-serif; }
        .main .block-container { padding: 1.2rem 2rem; max-width: 1500px; }
        .title { font-weight:900; font-size:1.6rem; margin-bottom:.25rem; }
        .subtitle { color:#6b7280; font-size:.95rem; margin-bottom:1rem; }

        /* Metric tiles */
        [data-testid="metric-container"] {
            background:#fff !important; border-radius:16px !important; border:1px solid rgba(0,0,0,.06)!important; padding:18px!important;
        }

        /* Soft light cards */
        .card {
            background: #fcfcfc;
            border: 1px solid #f1f5f9;
            border-radius: 16px;
            padding: 16px;
            height: 100%;
        }

        .pill { display:inline-block; padding:4px 10px; border-radius:999px; font-weight:700; font-size:.80rem; }
        .pill.excellent { background:#ecfdf5; color:#059669; }
        .pill.good      { background:#eff6ff; color:#2563eb; }
        .pill.warn      { background:#fffbeb; color:#d97706; }
        .pill.danger    { background:#fef2f2; color:#dc2626; }

        /* Multiselect chips — stronger overrides to stop red tags */
        .stMultiSelect [data-baseweb="tag"],
        div[data-baseweb="tag"],
        .stMultiSelect [class*="tag"] {
            background-color: #f9fafb !important;
            color: #111827 !important;
            border-radius: 12px !important;
            padding: 4px 10px !important;
            font-weight: 600 !important;
            border: 1px solid #e5e7eb !important;
            box-shadow: none !important;
        }
        .stMultiSelect [data-baseweb="tag"]:hover { background-color: #f0f9ff !important; border-color: #38bdf8 !important; }
        .stMultiSelect [data-baseweb="tag"] svg { fill: #6b7280 !important; }
        /* Input border softer */
        .stMultiSelect [data-baseweb="select"] > div {
            border-radius: 12px !important;
            border-color: #e5e7eb !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================
# LOADERS / HELPERS
# =========================================
def _pubhtml_to_xlsx(url: str) -> str:
    if "docs.google.com/spreadsheets/d/e/" in url:
        return re.sub(r"/pubhtml(.*)$", "/pub?output=xlsx", url.strip())
    return url


def _parse_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False).str.strip(),
        errors="coerce",
    )


@st.cache_data(show_spinner=False)
def load_workbook_from_gsheet(published_url: str) -> Dict[str, pd.DataFrame]:
    url = _pubhtml_to_xlsx((published_url or config.DEFAULT_PUBLISHED_URL).strip())
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    xls = pd.ExcelFile(io.BytesIO(r.content))
    sheets: Dict[str, pd.DataFrame] = {}
    for sn in xls.sheet_names:
        df = xls.parse(sn)
        df = df.dropna(how="all").dropna(axis=1, how="all")
        if not df.empty:
            sheets[sn] = df
    return sheets


def load_workbook_from_upload(file) -> Dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(file)
    out = {}
    for sn in xls.sheet_names:
        df = xls.parse(sn)
        df = df.dropna(how="all").dropna(axis=1, how="all")
        if not df.empty:
            out[sn] = df
    return out


def process_branch_data(excel_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    combined: List[pd.DataFrame] = []
    mapping = {
        "Date": "Date",
        "Day": "Day",
        "Sales Target": "SalesTarget",
        "SalesTarget": "SalesTarget",
        "Sales Achivement": "SalesActual",
        "Sales Achievement": "SalesActual",
        "SalesActual": "SalesActual",
        "Sales %": "SalesPercent",
        " Sales %": "SalesPercent",
        "SalesPercent": "SalesPercent",
        "NOB Target": "NOBTarget",
        "NOBTarget": "NOBTarget",
        "NOB Achievemnet": "NOBActual",
        "NOB Achievement": "NOBActual",
        "NOBActual": "NOBActual",
        "NOB %": "NOBPercent",
        " NOB %": "NOBPercent",
        "NOBPercent": "NOBPercent",
        "ABV Target": "ABVTarget",
        "ABVTarget": "ABVTarget",
        "ABV Achievement": "ABVActual",
        " ABV Achievement": "ABVActual",
        "ABVActual": "ABVActual",
        "ABV %": "ABVPercent",
        "ABVPercent": "ABVPercent",
    }
    for sheet_name, df in excel_data.items():
        if df.empty:
            continue
        d = df.copy()
        d.columns = d.columns.astype(str).str.strip()
        for old, new in mapping.items():
            if old in d.columns and new not in d.columns:
                d.rename(columns={old: new}, inplace=True)
        if "Date" not in d.columns:
            maybe = [c for c in d.columns if "date" in c.lower()]
            if maybe:
                d.rename(columns={maybe[0]: "Date"}, inplace=True)
        d["Branch"] = sheet_name
        meta = config.BRANCHES.get(sheet_name, {})
        d["BranchName"] = meta.get("name", sheet_name)
        d["BranchCode"] = meta.get("code", "000")
        if "Date" in d.columns:
            d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        for col in [
            "SalesTarget",
            "SalesActual",
            "SalesPercent",
            "NOBTarget",
            "NOBActual",
            "NOBPercent",
            "ABVTarget",
            "ABVActual",
            "ABVPercent",
        ]:
            if col in d.columns:
                d[col] = _parse_numeric(d[col])
        combined.append(d)
    return pd.concat(combined, ignore_index=True) if combined else pd.DataFrame()


def calc_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {}
    k: Dict[str, Any] = {}
    k["total_sales_target"] = float(df.get("SalesTarget", pd.Series(dtype=float)).sum())
    k["total_sales_actual"] = float(df.get("SalesActual", pd.Series(dtype=float)).sum())
    k["total_sales_variance"] = k["total_sales_actual"] - k["total_sales_target"]
    k["overall_sales_percent"] = (
        k["total_sales_actual"] / k["total_sales_target"] * 100 if k["total_sales_target"] > 0 else 0.0
    )
    k["total_nob_target"] = float(df.get("NOBTarget", pd.Series(dtype=float)).sum())
    k["total_nob_actual"] = float(df.get("NOBActual", pd.Series(dtype=float)).sum())
    k["overall_nob_percent"] = (
        k["total_nob_actual"] / k["total_nob_target"] * 100 if k["total_nob_target"] > 0 else 0.0
    )
    k["avg_abv_target"] = float(df.get("ABVTarget", pd.Series(dtype=float)).mean()) if "ABVTarget" in df else 0.0
    k["avg_abv_actual"] = float(df.get("ABVActual", pd.Series(dtype=float)).mean()) if "ABVActual" in df else 0.0
    k["overall_abv_percent"] = (
        k["avg_abv_actual"] / k["avg_abv_target"] * 100 if k["avg_abv_target"] > 0 else 0.0
    )

    if "BranchName" in df.columns:
        k["branch_performance"] = (
            df.groupby("BranchName")
            .agg(
                {
                    "SalesTarget": "sum",
                    "SalesActual": "sum",
                    "SalesPercent": "mean",
                    "NOBTarget": "sum",
                    "NOBActual": "sum",
                    "NOBPercent": "mean",
                    "ABVTarget": "mean",
                    "ABVActual": "mean",
                    "ABVPercent": "mean",
                }
            )
            .round(2)
        )

    if "Date" in df.columns and df["Date"].notna().any():
        k["date_range"] = {"start": df["Date"].min(), "end": df["Date"].max(), "days": int(df["Date"].dt.date.nunique())}

    score = w = 0.0
    if k.get("overall_sales_percent", 0) > 0:
        score += min(k["overall_sales_percent"] / 100, 1.2) * 40
        w += 40
    if k.get("overall_nob_percent", 0) > 0:
        score += min(k["overall_nob_percent"] / 100, 1.2) * 35
        w += 35
    if k.get("overall_abv_percent", 0) > 0:
        score += min(k["overall_abv_percent"] / 100, 1.2) * 25
        w += 25
    k["performance_score"] = (score / w * 100) if w > 0 else 0.0
    return k


# =========================================
# CHARTS (new daily model)
# =========================================
def _metric_area(df: pd.DataFrame, y_col: str, title: str, *, show_target: bool = True) -> go.Figure:
    """
    Render a daily area chart for the given metric. If show_target is True and a corresponding
    target column exists (e.g. SalesTarget for SalesActual), a secondary line will be drawn
    for the target values to facilitate comparison between actual and target.

    Parameters
    ----------
    df : pd.DataFrame
        Pre-filtered dataframe containing at least a Date, BranchName and the metric columns.
    y_col : str
        Column name for the actual metric values (e.g. 'SalesActual').
    title : str
        Plot title.
    show_target : bool, optional
        Whether to plot the target line if available. Defaults to True.
    """
    if df.empty or "Date" not in df.columns or df["Date"].isna().all():
        return go.Figure()

    # Determine aggregation function: ABVActual uses mean (average basket value), others use sum
    agg_func = "sum" if y_col != "ABVActual" else "mean"
    daily_actual = df.groupby(["Date", "BranchName"]).agg({y_col: agg_func}).reset_index()
    target_col_map = {"SalesActual": "SalesTarget", "NOBActual": "NOBTarget", "ABVActual": "ABVTarget"}
    target_col = target_col_map.get(y_col)

    # Prepare target data if requested and available
    daily_target: Optional[pd.DataFrame] = None
    if show_target and target_col and target_col in df.columns:
        agg_func_target = "sum" if y_col != "ABVActual" else "mean"
        daily_target = df.groupby(["Date", "BranchName"]).agg({target_col: agg_func_target}).reset_index()

    fig = go.Figure()
    palette = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444", "#14b8a6"]
    for i, br in enumerate(sorted(daily_actual["BranchName"].unique())):
        d_actual = daily_actual[daily_actual["BranchName"] == br]
        color = palette[i % len(palette)]
        # Plot actual values as filled area
        fig.add_trace(
            go.Scatter(
                x=d_actual["Date"],
                y=d_actual[y_col],
                name=f"{br} - Actual",
                mode="lines+markers",
                line=dict(width=3, color=color),
                fill="tozeroy",
                hovertemplate=f"<b>{br}</b><br>Date: %{{x|%Y-%m-%d}}<br>Actual: %{{y:,.0f}}<extra></extra>",
            )
        )
        # Plot target line if available
        if daily_target is not None:
            d_target = daily_target[daily_target["BranchName"] == br]
            fig.add_trace(
                go.Scatter(
                    x=d_target["Date"],
                    y=d_target[target_col],
                    name=f"{br} - Target",
                    mode="lines",
                    line=dict(width=2, color=color, dash="dash"),
                    fill=None,
                    hovertemplate=f"<b>{br}</b><br>Date: %{{x|%Y-%m-%d}}<br>Target: %{{y:,.0f}}<extra></extra>",
                    showlegend=False if len(sorted(daily_actual["BranchName"].unique())) > 1 else True,
                )
            )
    fig.update_layout(
        title=title,
        height=420,
        showlegend=True,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        xaxis_title="Date",
        yaxis_title="Value",
    )
    return fig


def _branch_comparison_chart(bp: pd.DataFrame) -> go.Figure:
    """
    Create a grouped bar chart comparing Sales %, NOB %, and ABV % across branches.

    Parameters
    ----------
    bp : pd.DataFrame
        Branch performance summary indexed by BranchName with percent columns.
    """
    if bp.empty:
        return go.Figure()
    metrics = {
        "SalesPercent": "Sales %",
        "NOBPercent": "NOB %",
        "ABVPercent": "ABV %",
    }
    fig = go.Figure()
    x = bp.index.tolist()
    palette = ["#3b82f6", "#10b981", "#f59e0b"]
    for i, (col, label) in enumerate(metrics.items()):
        y = bp[col].tolist()
        fig.add_trace(
            go.Bar(
                x=x,
                y=y,
                name=label,
                marker=dict(color=palette[i % len(palette)]),
            )
        )
    fig.update_layout(
        barmode="group",
        title="Branch Performance Comparison (%)",
        xaxis_title="Branch",
        yaxis_title="Percent",
        height=400,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    )
    return fig


# =========================================
# RENDER
# =========================================
def branch_status_class(pct: float) -> str:
    if pct >= 95:
        return "excellent"
    if pct >= 85:
        return "good"
    if pct >= 75:
        return "warn"
    return "danger"


def _branch_color_by_name(name: str) -> str:
    for _, meta in config.BRANCHES.items():
        if meta.get("name") == name:
            return meta.get("color", "#6b7280")
    return "#6b7280"


def render_branch_cards(bp: pd.DataFrame):
    st.markdown("### 🏪 Branch Overview")
    if bp.empty:
        st.info("No branch summary available.")
        return
    # 2 rows of light cards (auto wrap)
    cols_per_row = 3
    items = list(bp.index)
    for i in range(0, len(items), cols_per_row):
        row = st.columns(cols_per_row, gap="medium")
        for j, br in enumerate(items[i : i + cols_per_row]):
            with row[j]:
                r = bp.loc[br]
                avg_pct = float(np.nanmean([r.get("SalesPercent", 0), r.get("NOBPercent", 0), r.get("ABVPercent", 0)]) or 0)
                cls = branch_status_class(avg_pct)
                # ultra-light tint based on branch color (alpha ~ 10%)
                color = _branch_color_by_name(br)
                st.markdown(
                    f"""
                    <div class="card" style="background: linear-gradient(180deg, {color}1a 0%, #ffffff 40%);">
                      <div style="display:flex;justify-content:space-between;align-items:center">
                        <div style="font-weight:800">{br}</div>
                        <span class="pill {cls}">{avg_pct:.1f}%</span>
                      </div>
                      <div style="margin-top:8px;font-size:.92rem;color:#6b7280">Overall score</div>
                      <div style="display:flex;gap:12px;margin-top:10px;flex-wrap:wrap">
                        <div><div style="font-size:.8rem;color:#64748b">Sales %</div><div style="font-weight:700">{r.get('SalesPercent',0):.1f}%</div></div>
                        <div><div style="font-size:.8rem;color:#64748b">NOB %</div><div style="font-weight:700">{r.get('NOBPercent',0):.1f}%</div></div>
                        <div><div style="font-size:.8rem;color:#64748b">ABV %</div><div style="font-weight:700">{r.get('ABVPercent',0):.1f}%</div></div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_overview(df: pd.DataFrame, k: Dict[str, Any]):
    st.markdown("### 🏆 Overall Performance")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💰 Total Sales", f"SAR {k.get('total_sales_actual',0):,.0f}", delta=f"vs Target: {k.get('overall_sales_percent',0):.1f}%")
    variance_color = "normal" if k.get("total_sales_variance", 0) >= 0 else "inverse"
    c2.metric("📊 Sales Variance", f"SAR {k.get('total_sales_variance',0):,.0f}", delta=f"{k.get('overall_sales_percent',0)-100:+.1f}%", delta_color=variance_color)
    nob_color = "normal" if k.get("overall_nob_percent", 0) >= config.TARGETS['nob_achievement'] else "inverse"
    c3.metric("🛍️ Total Baskets", f"{k.get('total_nob_actual',0):,.0f}", delta=f"Achievement: {k.get('overall_nob_percent',0):.1f}%", delta_color=nob_color)
    abv_color = "normal" if k.get("overall_abv_percent", 0) >= config.TARGETS['abv_achievement'] else "inverse"
    c4.metric("💎 Avg Basket Value", f"SAR {k.get('avg_abv_actual',0):,.2f}", delta=f"vs Target: {k.get('overall_abv_percent',0):.1f}%", delta_color=abv_color)
    score_color = "normal" if k.get("performance_score", 0) >= 80 else "off"
    c5.metric("⭐ Performance Score", f"{k.get('performance_score',0):.0f}/100", delta="Weighted Score", delta_color=score_color)

    if "branch_performance" in k and not k["branch_performance"].empty:
        render_branch_cards(k["branch_performance"])

    # Comparison (table only; chart removed here to keep the layout clean)
    if "branch_performance" in k and not k["branch_performance"].empty:
        st.markdown("### 📊 Comparison Table")
        bp = k["branch_performance"].copy()
        st.dataframe(
            bp[["SalesPercent", "NOBPercent", "ABVPercent", "SalesActual", "SalesTarget"]]
            .rename(
                columns={
                    "SalesPercent": "Sales %",
                    "NOBPercent": "NOB %",
                    "ABVPercent": "ABV %",
                    "SalesActual": "Sales (Actual)",
                    "SalesTarget": "Sales (Target)",
                }
            )
            .round(1),
            use_container_width=True,
        )
        # New chart: branch performance comparison bar chart
        st.markdown("### 📉 Branch Performance Comparison")
        fig_cmp = _branch_comparison_chart(bp)
        st.plotly_chart(fig_cmp, use_container_width=True, config={"displayModeBar": False})


# =========================================
# MAIN
# =========================================
def main():
    apply_css()

    # ----- Sidebar (no "Advanced" checkbox)
    with st.sidebar:
        st.markdown(f"### 📊 {config.PAGE_TITLE}")
        st.caption("Filters affect all tabs.")

        # Small expander for data source (kept minimal and closed by default)
        with st.expander("Data source (optional)", expanded=False):
            url = st.text_input("Published Google Sheets URL", value=config.DEFAULT_PUBLISHED_URL, help="Paste a 'Publish to web' URL.")
            uploaded_file = st.file_uploader("Or upload Excel", type=["xlsx", "xls"])
            if st.button("🔄 Refresh data"):
                load_workbook_from_gsheet.clear()
                st.rerun()

    # Load data (prefer upload if provided)
    if 'uploaded_file' in locals() and uploaded_file is not None:
        sheets_map = load_workbook_from_upload(uploaded_file)
    else:
        # If user never opened expander, url won't exist in locals; use default
        url = locals().get("url", config.DEFAULT_PUBLISHED_URL)
        sheets_map = load_workbook_from_gsheet(url)

    if not sheets_map:
        st.warning("No non-empty sheets found.")
        st.stop()

    df = process_branch_data(sheets_map)
    if df.empty:
        st.error("Could not process data. Check column names and sheet structure.")
        st.stop()

    # Clean header (no success banner)
    st.markdown(f"<div class='title'>📊 {config.PAGE_TITLE}</div>", unsafe_allow_html=True)
    date_span = ""
    if "Date" in df.columns and df["Date"].notna().any():
        date_span = f"{df['Date'].min().date()} → {df['Date'].max().date()}"
    st.markdown(f"<div class='subtitle'>Branches: {df['BranchName'].nunique() if 'BranchName' in df else 0} • Rows: {len(df):,} {('• ' + date_span) if date_span else ''}</div>", unsafe_allow_html=True)

    # Branch filter (soft pills)
    if "BranchName" in df.columns:
        with st.sidebar:
            all_branches = sorted([b for b in df["BranchName"].dropna().unique()])
            selected = st.multiselect("Filter Branches", options=all_branches, default=all_branches)
        if selected:
            df = df[df["BranchName"].isin(selected)].copy()
        if df.empty:
            st.warning("No rows after branch filter.")
            st.stop()

    # Date filter (main)
    if "Date" in df.columns and df["Date"].notna().any():
        c1, c2, _ = st.columns([2, 2, 6])
        with c1:
            start_d = st.date_input("Start Date", value=df["Date"].min().date())
        with c2:
            end_d = st.date_input("End Date", value=df["Date"].max().date())
        mask = (df["Date"].dt.date >= start_d) & (df["Date"].dt.date <= end_d)
        df = df.loc[mask].copy()
        if df.empty:
            st.warning("No rows in selected date range.")
            st.stop()

    # KPIs + Quick insights
    k = calc_kpis(df)
    with st.expander("⚡ Quick Insights", expanded=True):
        insights: List[str] = []
        if "branch_performance" in k and not k["branch_performance"].empty:
            bp = k["branch_performance"]
            insights.append(f"🥇 Best Sales %: {bp['SalesPercent'].idxmax()} — {bp['SalesPercent'].max():.1f}%")
            insights.append(f"🔻 Lowest Sales %: {bp['SalesPercent'].idxmin()} — {bp['SalesPercent'].min():.1f}%")
            below = bp[bp["SalesPercent"] < config.TARGETS["sales_achievement"]].index.tolist()
            if below:
                insights.append("⚠️ Below 95% target: " + ", ".join(below))
        if k.get("total_sales_variance", 0) < 0:
            insights.append(f"🟥 Overall variance negative by SAR {abs(k['total_sales_variance']):,.0f}")
        if insights:
            for it in insights:
                st.markdown("- " + it)
        else:
            st.write("All metrics look healthy for the current selection.")

    # Tabs
    t1, t2, t3 = st.tabs(["🏠 Branch Overview", "📈 Daily Trends", "📥 Export"])
    with t1:
        render_overview(df, k)

    with t2:
        st.markdown("#### Choose Metric")
        mtab1, mtab2, mtab3 = st.tabs(["💰 Sales", "🛍️ NOB", "💎 ABV"])
        # Time window controls for trends
        if "Date" in df.columns and df["Date"].notna().any():
            opts = ["Last 7 Days", "Last 30 Days", "Last 3 Months", "All Time"]
            choice = st.selectbox("Time Period", opts, index=1, key="trend_window")
            today = (df["Date"].max() if df["Date"].notna().any() else pd.Timestamp.today()).date()
            start = {"Last 7 Days": today - timedelta(days=7),
                     "Last 30 Days": today - timedelta(days=30),
                     "Last 3 Months": today - timedelta(days=90)}.get(choice, df["Date"].min().date())
            f = df[(df["Date"].dt.date >= start) & (df["Date"].dt.date <= today)].copy()
        else:
            f = df.copy()

        with mtab1:
            st.plotly_chart(_metric_area(f, "SalesActual", "Daily Sales (Actual vs Target)"), use_container_width=True, config={"displayModeBar": False})
        with mtab2:
            st.plotly_chart(_metric_area(f, "NOBActual", "Number of Baskets (Actual vs Target)"), use_container_width=True, config={"displayModeBar": False})
        with mtab3:
            st.plotly_chart(_metric_area(f, "ABVActual", "Average Basket Value (Actual vs Target)"), use_container_width=True, config={"displayModeBar": False})

    with t3:
        if df.empty:
            st.info("No data to export.")
        else:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                df.to_excel(writer, sheet_name="All Branches", index=False)
                if "branch_performance" in k:
                    k["branch_performance"].to_excel(writer, sheet_name="Branch Summary")
            st.download_button(
                "📊 Download Excel Report",
                buf.getvalue(),
                f"Alkhair_Branch_Analytics_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
            st.download_button(
                "📄 Download CSV",
                df.to_csv(index=False).encode("utf-8"),
                f"Alkhair_Branch_Data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        st.error("❌ Application Error. Please adjust filters or refresh.")
