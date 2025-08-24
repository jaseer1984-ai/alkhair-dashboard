# app.py — Al Khair Business Performance
# - Sidebar: pinned Branch filter (restyled soft pills)
# - URL input hidden by default (toggle in sidebar)
# - Light-colored Branch Overview cards (soft gradient per branch)
# - Daily Trends (metric tabs, area charts)
# - Upload Excel fallback

from __future__ import annotations

import io, re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

# ----------------------------
# Configuration
# ----------------------------
class Config:
    PAGE_TITLE = "Al Khair Business Performance"
    LAYOUT = "wide"
    DEFAULT_PUBLISHED_URL = (
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vQG7boLWl2bNLCPR05NXv6EFpPPcFfXsiXPQ7rAGYr3q8Nkc2Ijg8BqEwVofcMLSg/pubhtml"
    )
    BRANCHES = {
        "Al khair - 102": {"name": "Al Khair", "code": "102", "color": "#3b82f6"},  # indigo
        "Noora - 104":    {"name": "Noora",   "code": "104", "color": "#10b981"},  # emerald
        "Hamra - 109":    {"name": "Hamra",   "code": "109", "color": "#f59e0b"},  # amber
        "Magnus - 107":   {"name": "Magnus",  "code": "107", "color": "#8b5cf6"},  # violet
    }
    TARGETS = {"sales_achievement": 95.0, "nob_achievement": 90.0, "abv_achievement": 90.0}
    HIDE_URL_INPUT_BY_DEFAULT = True

config = Config()
st.set_page_config(page_title=config.PAGE_TITLE, layout=config.LAYOUT, initial_sidebar_state="expanded")

# ----------------------------
# CSS (restyle multiselect pills + light cards)
# ----------------------------
def apply_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&display=swap');
        html, body, [data-testid="stAppViewContainer"] { font-family: 'Inter', sans-serif; }
        .main .block-container { padding: 1.5rem 2rem; max-width: 1500px; }
        .title { font-weight:900; font-size:1.6rem; margin-bottom:.5rem; }

        /* Metric cards */
        [data-testid="metric-container"] {
          background:#ffffff !important;
          border-radius:16px !important;
          border:1px solid rgba(0,0,0,.06)!important;
          padding:18px!important;
        }

        /* Light "content" cards */
        .card {
          background: #fafafa;  /* light neutral base */
          border: 1px solid #eef2f7;
          border-radius: 16px;
          padding: 16px;
          height: 100%;
        }

        /* Status pills */
        .pill { display:inline-block; padding:4px 10px; border-radius:999px; font-weight:700; font-size:.80rem; }
        .pill.excellent { background: rgba(16,185,129,.15); color:#059669; }
        .pill.good      { background: rgba(59,130,246,.15);  color:#2563eb; }
        .pill.warn      { background: rgba(245,158,11,.15);  color:#d97706; }
        .pill.danger    { background: rgba(239,68,68,.15);   color:#dc2626; }

        /* Sidebar multiselect — pretty soft tags */
        /* Selected tag “cards” */
        div[data-baseweb="tag"] {
          background-color: #f3f4f6 !important;      /* soft gray */
          color: #111827 !important;                 /* near-black text */
          border-radius: 12px !important;
          padding: 4px 10px !important;
          font-weight: 600 !important;
          border: 1px solid #e5e7eb !important;      /* subtle border */
          transition: all 0.15s ease-in-out;
        }
        div[data-baseweb="tag"]:hover {
          background-color: #e0f2fe !important;      /* very light blue hover */
          border-color: #38bdf8 !important;
        }
        /* Close (x) icon softer */
        div[data-baseweb="tag"] svg { fill: #6b7280 !important; }

        /* Dropdown options a bit cleaner */
        [data-baseweb="select"] div[role="listbox"] div[role="option"] {
          border-radius: 8px; margin: 2px 6px; padding: 6px 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------
# Helpers / Loading
# ----------------------------
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
    out: Dict[str, pd.DataFrame] = {}
    for sn in xls.sheet_names:
        df = xls.parse(sn).dropna(how="all").dropna(axis=1, how="all")
        if not df.empty:
            out[sn] = df
    return out

def load_workbook_from_upload(file) -> Dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(file)
    out: Dict[str, pd.DataFrame] = {}
    for sn in xls.sheet_names:
        df = xls.parse(sn).dropna(how="all").dropna(axis=1, how="all")
        if not df.empty:
            out[sn] = df
    return out

def process_branch_data(excel_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    combined: List[pd.DataFrame] = []
    mapping = {
        "Date":"Date","Day":"Day",
        "Sales Target":"SalesTarget","SalesTarget":"SalesTarget",
        "Sales Achivement":"SalesActual","Sales Achievement":"SalesActual","SalesActual":"SalesActual",
        "Sales %":"SalesPercent"," Sales %":"SalesPercent","SalesPercent":"SalesPercent",
        "NOB Target":"NOBTarget","NOBTarget":"NOBTarget",
        "NOB Achievemnet":"NOBActual","NOB Achievement":"NOBActual","NOBActual":"NOBActual",
        "NOB %":"NOBPercent"," NOB %":"NOBPercent","NOBPercent":"NOBPercent",
        "ABV Target":"ABVTarget","ABVTarget":"ABVTarget",
        "ABV Achievement":"ABVActual"," ABV Achievement":"ABVActual","ABVActual":"ABVActual",
        "ABV %":"ABVPercent","ABVPercent":"ABVPercent",
    }
    for sheet_name, df in excel_data.items():
        d = df.copy()
        d.columns = d.columns.astype(str).str.strip()
        for old,new in mapping.items():
            if old in d.columns and new not in d.columns:
                d.rename(columns={old:new}, inplace=True)
        if "Date" not in d.columns:
            maybe = [c for c in d.columns if "date" in c.lower()]
            if maybe: d.rename(columns={maybe[0]:"Date"}, inplace=True)
        d["Branch"] = sheet_name
        meta = config.BRANCHES.get(sheet_name, {})
        d["BranchName"] = meta.get("name", sheet_name)
        d["BranchCode"] = meta.get("code", "000")
        if "Date" in d.columns:
            d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        for col in ["SalesTarget","SalesActual","SalesPercent","NOBTarget","NOBActual","NOBPercent","ABVTarget","ABVActual","ABVPercent"]:
            if col in d.columns: d[col] = _parse_numeric(d[col])
        combined.append(d)
    return pd.concat(combined, ignore_index=True) if combined else pd.DataFrame()

def calc_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty: return {}
    k: Dict[str, Any] = {}
    k["total_sales_target"] = float(df.get("SalesTarget", pd.Series(dtype=float)).sum())
    k["total_sales_actual"] = float(df.get("SalesActual", pd.Series(dtype=float)).sum())
    k["total_sales_variance"] = k["total_sales_actual"] - k["total_sales_target"]
    k["overall_sales_percent"] = (k["total_sales_actual"] / k["total_sales_target"] * 100) if k["total_sales_target"] > 0 else 0.0
    k["total_nob_target"] = float(df.get("NOBTarget", pd.Series(dtype=float)).sum())
    k["total_nob_actual"] = float(df.get("NOBActual", pd.Series(dtype=float)).sum())
    k["overall_nob_percent"] = (k["total_nob_actual"] / k["total_nob_target"] * 100) if k["total_nob_target"] > 0 else 0.0
    k["avg_abv_target"] = float(df.get("ABVTarget", pd.Series(dtype=float)).mean()) if "ABVTarget" in df else 0.0
    k["avg_abv_actual"] = float(df.get("ABVActual", pd.Series(dtype=float)).mean()) if "ABVActual" in df else 0.0
    k["overall_abv_percent"] = (k["avg_abv_actual"] / k["avg_abv_target"] * 100) if k["avg_abv_target"] > 0 else 0.0
    if "BranchName" in df.columns:
        k["branch_performance"] = (
            df.groupby("BranchName")
              .agg({
                  "SalesTarget":"sum","SalesActual":"sum","SalesPercent":"mean",
                  "NOBTarget":"sum","NOBActual":"sum","NOBPercent":"mean",
                  "ABVTarget":"mean","ABVActual":"mean","ABVPercent":"mean",
              }).round(2)
        )
    if "Date" in df.columns and df["Date"].notna().any():
        k["date_range"] = {"start": df["Date"].min(), "end": df["Date"].max(), "days": int(df["Date"].dt.date.nunique())}
    score=w=0.0
    if k.get("overall_sales_percent",0)>0: score+=min(k["overall_sales_percent"]/100,1.2)*40; w+=40
    if k.get("overall_nob_percent",0)>0:   score+=min(k["overall_nob_percent"]/100,1.2)*35;   w+=35
    if k.get("overall_abv_percent",0)>0:   score+=min(k["overall_abv_percent"]/100,1.2)*25;   w+=25
    k["performance_score"]=(score/w*100) if w>0 else 0.0
    return k

# ----------------------------
# Charts (new trends model)
# ----------------------------
def _metric_area(df: pd.DataFrame, y_col: str, title: str) -> go.Figure:
    if df.empty or "Date" not in df.columns or df["Date"].isna().all():
        return go.Figure()
    agg_func = "sum" if y_col != "ABVActual" else "mean"
    daily = df.groupby(["Date","BranchName"]).agg({y_col: agg_func}).reset_index()
    if daily.empty:
        return go.Figure()
    fig = go.Figure()
    palette = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444", "#14b8a6"]
    for i, br in enumerate(daily["BranchName"].unique()):
        d = daily[daily["BranchName"] == br]
        fig.add_trace(
            go.Scatter(
                x=d["Date"], y=d[y_col], name=br, mode="lines+markers",
                line=dict(width=3, color=palette[i % len(palette)]),
                fill="tozeroy",
                hovertemplate=f"<b>{br}</b><br>Date: %{x|%Y-%m-%d}<br>Value: %{y:,.0f}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title, height=420, showlegend=True,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    )
    return fig

# ----------------------------
# Rendering
# ----------------------------
def branch_status_class(pct: float) -> str:
    if pct >= 95: return "excellent"
    if pct >= 85: return "good"
    if pct >= 75: return "warn"
    return "danger"

def _branch_color_by_name(name: str) -> str:
    for key, meta in config.BRANCHES.items():
        if meta.get("name") == name:
            return meta.get("color", "#6b7280")
    return "#6b7280"

def render_branch_cards(bp: pd.DataFrame):
    st.markdown("#### 🏪 Branch Overview — Cards")
    if bp.empty:
        st.info("No branch summary available.")
        return
    cols_per_row = 3
    items = list(bp.index)
    for i in range(0, len(items), cols_per_row):
        row = st.columns(cols_per_row, gap="medium")
        for j, br in enumerate(items[i:i+cols_per_row]):
            with row[j]:
                r = bp.loc[br]
                avg_pct = float(np.nanmean([r.get("SalesPercent",0), r.get("NOBPercent",0), r.get("ABVPercent",0)]) or 0)
                cls = branch_status_class(avg_pct)
                color = _branch_color_by_name(br)
                # light gradient using branch color -> transparent white
                st.markdown(
                    f"""
                    <div class="card" style="
                        background: linear-gradient(180deg, {color}1f 0%, #ffffff 45%);
                    ">
                      <div style="display:flex;justify-content:space-between;align-items:center">
                        <div style="font-weight:800">{br}</div>
                        <span class="pill {cls}">{avg_pct:.1f}%</span>
                      </div>
                      <div style="margin-top:8px;font-size:.92rem;color:#6b7280">Overall score</div>
                      <div style="display:flex;gap:12px;margin-top:10px">
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
    c1.metric("💰 Total Sales", f"SAR {k.get('total_sales_actual',0):,.0f}",
              delta=f"vs Target: {k.get('overall_sales_percent',0):.1f}%")
    variance_color = "normal" if k.get("total_sales_variance", 0) >= 0 else "inverse"
    c2.metric("📊 Sales Variance", f"SAR {k.get('total_sales_variance',0):,.0f}",
              delta=f"{k.get('overall_sales_percent',0)-100:+.1f}%", delta_color=variance_color)
    nob_color = "normal" if k.get("overall_nob_percent", 0) >= 90 else "inverse"
    c3.metric("🛍️ Total Baskets", f"{k.get('total_nob_actual',0):,.0f}",
              delta=f"Achievement: {k.get('overall_nob_percent',0):.1f}%", delta_color=nob_color)
    abv_color = "normal" if k.get("overall_abv_percent", 0) >= 90 else "inverse"
    c4.metric("💎 Avg Basket Value", f"SAR {k.get('avg_abv_actual',0):,.2f}",
              delta=f"vs Target: {k.get('overall_abv_percent',0):.1f}%", delta_color=abv_color)
    score_color = "normal" if k.get("performance_score", 0) >= 80 else "off"
    c5.metric("⭐ Performance Score", f"{k.get('performance_score',0):.0f}/100",
              delta="Weighted Score", delta_color=score_color)

    if "branch_performance" in k and not k["branch_performance"].empty:
        render_branch_cards(k["branch_performance"])

# ----------------------------
# Main
# ----------------------------
def main():
    apply_css()

    # Sidebar
    with st.sidebar:
        st.markdown(f"### 📊 {config.PAGE_TITLE}")
        st.caption("Filters affect all tabs.")
        show_url = st.checkbox("Advanced: change data source URL", value=not config.HIDE_URL_INPUT_BY_DEFAULT)
        url = config.DEFAULT_PUBLISHED_URL
        uploaded_file: Optional[io.BytesIO] = None
        if show_url:
            url = st.text_input("Published Google Sheets URL", value=config.DEFAULT_PUBLISHED_URL,
                                help="Paste a 'Publish to web' URL.")
            uploaded_file = st.file_uploader("…or upload Excel", type=["xlsx", "xls"])
        if st.button("🔄 Refresh"):
            load_workbook_from_gsheet.clear()
            st.rerun()

    # Load
    if uploaded_file is not None:
        sheets_map = load_workbook_from_upload(uploaded_file)
    else:
        sheets_map = load_workbook_from_gsheet(url)
    if not sheets_map:
        st.warning("No non-empty sheets found."); st.stop()

    df = process_branch_data(sheets_map)
    if df.empty:
        st.error("Could not process data. Check column names and sheet structure."); st.stop()

    # Title + info
    st.markdown(f"<div class='title'>📊 {config.PAGE_TITLE}</div>", unsafe_allow_html=True)
    st.success(f"Loaded data for {df['BranchName'].nunique()} branches • {len(df):,} rows")

    # Branch filter (sidebar, pretty pills)
    if "BranchName" in df.columns:
        with st.sidebar:
            all_branches = sorted([b for b in df["BranchName"].dropna().unique()])
            selected = st.multiselect("Filter Branches", options=all_branches, default=all_branches)
        if selected:
            df = df[df["BranchName"].isin(selected)].copy()
        if df.empty:
            st.warning("No rows after branch filter."); st.stop()

    # Date filter
    if "Date" in df.columns and df["Date"].notna().any():
        c1, c2, _ = st.columns([2, 2, 6])
        with c1: start_d = st.date_input("Start Date", value=df["Date"].min().date())
        with c2: end_d   = st.date_input("End Date",   value=df["Date"].max().date())
        mask = (df["Date"].dt.date >= start_d) & (df["Date"].dt.date <= end_d)
        df = df.loc[mask].copy()
        if df.empty:
            st.warning("No rows in selected date range."); st.stop()

    # KPIs + insights
    k = calc_kpis(df)
    with st.expander("⚡ Quick Insights", expanded=True):
        notes: List[str] = []
        if "branch_performance" in k and not k["branch_performance"].empty:
            bp = k["branch_performance"]
            notes.append(f"🥇 **Best Sales %:** {bp['SalesPercent'].idxmax()} — {bp['SalesPercent'].max():.1f}%")
            notes.append(f"🔻 **Lowest Sales %:** {bp['SalesPercent'].idxmin()} — {bp['SalesPercent'].min():.1f}%")
            below = bp[bp["SalesPercent"] < config.TARGETS["sales_achievement"]].index.tolist()
            if below: notes.append("⚠️ Below 95% target: " + ", ".join(below))
        if k.get("total_sales_variance",0) < 0:
            notes.append(f"🟥 Overall variance negative by SAR {abs(k['total_sales_variance']):,.0f}")
        if notes:
            for n in notes: st.markdown("- " + n)
        else:
            st.write("All metrics look healthy for the current selection.")

    # Tabs
    t1, t2, t3 = st.tabs(["🏠 Branch Overview", "📈 Daily Trends (New Model)", "📥 Export"])
    with t1:
        render_overview(df, k)
        if "branch_performance" in k and not k["branch_performance"].empty:
            st.markdown("### 📊 Comparison Table")
            bp = k["branch_performance"]
            st.dataframe(
                bp[["SalesPercent","NOBPercent","ABVPercent","SalesActual","SalesTarget"]]
                .rename(columns={"SalesPercent":"Sales %","NOBPercent":"NOB %","ABVPercent":"ABV %","SalesActual":"Sales (Actual)","SalesTarget":"Sales (Target)"})
                .round(1),
                use_container_width=True,
            )

    with t2:
        st.markdown("#### Choose Metric")
        mtab1, mtab2, mtab3 = st.tabs(["💰 Sales","🛍️ NOB","💎 ABV"])
        # Time window
        if "Date" in df.columns and df["Date"].notna().any():
            opts = ["Last 7 Days","Last 30 Days","Last 3 Months","All Time"]
            choice = st.selectbox("Time Period", opts, index=1, key="trend_window")
            today = (df["Date"].max() if df["Date"].notna().any() else pd.Timestamp.today()).date()
            start = {"Last 7 Days": today - timedelta(days=7),
                     "Last 30 Days": today - timedelta(days=30),
                     "Last 3 Months": today - timedelta(days=90)}.get(choice, df["Date"].min().date())
            f = df[(df["Date"].dt.date >= start) & (df["Date"].dt.date <= today)].copy()
        else:
            f = df.copy()

        with mtab1:
            st.plotly_chart(_metric_area(f, "SalesActual", "Daily Sales (Area)"), use_container_width=True, config={"displayModeBar": False})
        with mtab2:
            st.plotly_chart(_metric_area(f, "NOBActual", "Number of Baskets (Area)"), use_container_width=True, config={"displayModeBar": False})
        with mtab3:
            st.plotly_chart(_metric_area(f, "ABVActual", "Average Basket Value (Area)"), use_container_width=True, config={"displayModeBar": False})

    with t3:
        st.markdown("### Export")
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
