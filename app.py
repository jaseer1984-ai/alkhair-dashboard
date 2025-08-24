# app.py — Al Khair Business Performance (URL box hidden + left KPI cards)

from __future__ import annotations

import io, re
from datetime import datetime, timedelta
from typing import Dict, Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------------
# CONFIG
# ----------------------------------
class Config:
    PAGE_TITLE = "Al Khair Business Performance"
    LAYOUT = "wide"
    # Hidden default Google Sheet (published → web). We auto-convert /pubhtml → /pub?output=xlsx
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
st.set_page_config(page_title=config.PAGE_TITLE, layout=config.LAYOUT, initial_sidebar_state="collapsed")

# ----------------------------------
# STYLES (URL/alerts hidden, left cards)
# ----------------------------------
def apply_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&display=swap');
        html, body, [data-testid="stAppViewContainer"] { font-family: 'Inter', sans-serif; }
        .main .block-container { padding: 1.5rem 2rem; max-width: 1500px; }
        [data-testid='stSidebar'] { display: none !important; }  /* hide sidebar */

        /* Hide any success/info banners if they slip in */
        .stAlert { display: none !important; }

        .page-title { font-weight:900; font-size:1.7rem; margin-bottom:.25rem; }
        .subtle { color:#64748b; font-weight:600; }

        /* Left KPI card stack */
        .kpi-card {
            background:#fff; border:1px solid rgba(0,0,0,.06); border-radius:16px;
            padding:14px 16px; margin-bottom:10px;
        }
        .kpi-title { font-weight:800; font-size:.95rem; color:#334155; margin-bottom:4px; }
        .kpi-value { font-weight:900; font-size:1.35rem; }
        .kpi-delta { font-weight:700; font-size:.85rem; color:#64748b; }

        /* Plotly container */
        .stPlotlyChart > div { background:#fff; border:1px solid rgba(0,0,0,.06); border-radius:16px; padding:12px; }
        .stTabs [data-baseweb="tab-list"]{ gap:6px; background:#fff; border-radius:14px; padding:8px; border:1px solid rgba(0,0,0,0.05)}
        .stTabs [data-baseweb="tab"]{ border-radius:12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------------
# LOADING & PROCESSING
# ----------------------------------
def _pubhtml_to_xlsx(url: str) -> str:
    if "docs.google.com/spreadsheets/d/e/" in url:
        return re.sub(r"/pubhtml(.*)$", "/pub?output=xlsx", url.strip())
    return url

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

def process_branch_data(sheets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    combined = []
    mapping = {
        "Date":"Date","Day":"Day",
        "Sales Target":"SalesTarget",
        "Sales Achivement":"SalesActual","Sales Achievement":"SalesActual",
        "Sales %":"SalesPercent"," Sales %":"SalesPercent",
        "NOB Target":"NOBTarget",
        "NOB Achievemnet":"NOBActual","NOB Achievement":"NOBActual",
        "NOB %":"NOBPercent"," NOB %":"NOBPercent",
        "ABV Target":"ABVTarget","ABV Achievement":"ABVActual"," ABV Achievement":"ABVActual",
        "ABV %":"ABVPercent"
    }
    for sheet, df in sheets.items():
        d = df.copy()
        d.columns = d.columns.astype(str).str.strip()
        for old,new in mapping.items():
            if old in d.columns: d.rename(columns={old:new}, inplace=True)
        d["Branch"] = sheet
        d["BranchName"] = config.BRANCHES.get(sheet, {}).get("name", sheet)
        d["BranchCode"] = config.BRANCHES.get(sheet, {}).get("code", "000")
        if "Date" in d.columns: d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        for col in ["SalesTarget","SalesActual","SalesPercent","NOBTarget","NOBActual","NOBPercent","ABVTarget","ABVActual","ABVPercent"]:
            if col in d.columns: d[col] = pd.to_numeric(d[col], errors="coerce")
        combined.append(d)
    return pd.concat(combined, ignore_index=True) if combined else pd.DataFrame()

def calc_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty: return {}
    k: Dict[str, Any] = {}
    k["total_sales_target"] = float(df.get("SalesTarget", pd.Series(dtype=float)).sum())
    k["total_sales_actual"] = float(df.get("SalesActual", pd.Series(dtype=float)).sum())
    k["total_sales_variance"] = k["total_sales_actual"] - k["total_sales_target"]
    k["overall_sales_percent"] = k["total_sales_actual"]/k["total_sales_target"]*100 if k["total_sales_target"]>0 else 0
    k["total_nob_target"] = float(df.get("NOBTarget", pd.Series(dtype=float)).sum())
    k["total_nob_actual"] = float(df.get("NOBActual", pd.Series(dtype=float)).sum())
    k["overall_nob_percent"] = k["total_nob_actual"]/k["total_nob_target"]*100 if k["total_nob_target"]>0 else 0
    k["avg_abv_target"] = float(df.get("ABVTarget", pd.Series(dtype=float)).mean()) if "ABVTarget" in df else 0
    k["avg_abv_actual"] = float(df.get("ABVActual", pd.Series(dtype=float)).mean()) if "ABVActual" in df else 0
    k["overall_abv_percent"] = k["avg_abv_actual"]/k["avg_abv_target"]*100 if k["avg_abv_target"]>0 else 0
    if "BranchName" in df.columns:
        k["branch_performance"] = df.groupby("BranchName").agg({
            "SalesTarget":"sum","SalesActual":"sum","SalesPercent":"mean",
            "NOBTarget":"sum","NOBActual":"sum","NOBPercent":"mean",
            "ABVTarget":"mean","ABVActual":"mean","ABVPercent":"mean"
        }).round(2)
    if "Date" in df.columns and df["Date"].notna().any():
        k["date_range"] = {
            "start": df["Date"].min(), "end": df["Date"].max(),
            "days": int(df["Date"].dt.date.nunique())
        }
    # weighted score
    score=w=0
    for val,wt in [(k.get("overall_sales_percent",0),40),(k.get("overall_nob_percent",0),35),(k.get("overall_abv_percent",0),25)]:
        if val>0: score += min(val/100,1.2)*wt; w += wt
    k["performance_score"] = (score/w*100) if w else 0
    return k

# ----------------------------------
# CHARTS
# ----------------------------------
def chart_branch_comparison(branch_data: pd.DataFrame) -> go.Figure:
    if branch_data.empty: return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=branch_data.index, y=branch_data["SalesPercent"], name="Sales Achievement %",
        marker=dict(color=branch_data["SalesPercent"], colorscale="RdYlGn", cmin=0, cmax=120, showscale=True,
                    colorbar=dict(title="Achievement %", x=1.02)),
        text=[f"{v:.1f}%" for v in branch_data["SalesPercent"]], textposition="outside",
        customdata=branch_data["SalesTarget"],
        hovertemplate="<b>%{x}</b><br>Achievement: %{y:.1f}%<br>Target: SAR %{customdata:,.0f}<extra></extra>"
    ))
    fig.add_hline(y=100, line_dash="dash", line_color="#ef4444", line_width=3,
                  annotation_text="💯 Target", annotation_position="top right")
    fig.add_hline(y=95, line_dash="dot", line_color="#10b981", line_width=2,
                  annotation_text="🌟 Excellence", annotation_position="bottom right")
    fig.update_layout(title="Branch Sales Performance Comparison", xaxis_title="Branch", yaxis_title="Achievement (%)",
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=460, showlegend=False)
    return fig

def chart_daily_trends(df: pd.DataFrame) -> go.Figure:
    if df.empty or "Date" not in df or df["Date"].isna().all(): return go.Figure()
    daily = df.groupby(["Date","BranchName"]).agg({
        "SalesActual":"sum","NOBActual":"sum","ABVActual":"mean"
    }).reset_index()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("Daily Sales","Number of Baskets","Average Basket Value"), vertical_spacing=0.08)
    colors = ["#3b82f6","#10b981","#f59e0b","#8b5cf6","#ef4444"]
    for i, br in enumerate(daily["BranchName"].unique()):
        d = daily[daily["BranchName"]==br]; col = colors[i%len(colors)]
        fig.add_trace(go.Scatter(x=d["Date"], y=d["SalesActual"], name=f"{br} — Sales",
                                 line=dict(color=col, width=3), mode="lines+markers"), row=1,col=1)
        fig.add_trace(go.Scatter(x=d["Date"], y=d["NOBActual"], name=f"{br} — NOB",
                                 line=dict(color=col, width=3), mode="lines+markers", showlegend=False), row=2,col=1)
        fig.add_trace(go.Scatter(x=d["Date"], y=d["ABVActual"], name=f"{br} — ABV",
                                 line=dict(color=col, width=3), mode="lines+markers", showlegend=False), row=3,col=1)
    fig.update_layout(height=800, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"))
    return fig

# ----------------------------------
# RIGHT-PANE SECTIONS
# ----------------------------------
def render_overview(df: pd.DataFrame, k: Dict[str, Any]):
    st.markdown("### 🏪 Branch Dashboard")
    if "branch_performance" in k:
        st.plotly_chart(chart_branch_comparison(k["branch_performance"]),
                        use_container_width=True, config={"displayModeBar": False}, key="comp_overview")

def render_daily(df: pd.DataFrame):
    st.markdown("### 📈 Daily Trends")
    st.plotly_chart(chart_daily_trends(df), use_container_width=True,
                    config={"displayModeBar": False}, key="daily_trends")

def render_export(df: pd.DataFrame, k: Dict[str, Any]):
    st.markdown("### 📥 Export")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="All Branches", index=False)
        if "branch_performance" in k:
            k["branch_performance"].to_excel(writer, sheet_name="Branch Summary")
        if "Date" in df and df["Date"].notna().any():
            daily = df.groupby(["Date","BranchName"]).agg({
                "SalesActual":"sum","SalesTarget":"sum","NOBActual":"sum","ABVActual":"mean"}).reset_index()
            daily.to_excel(writer, sheet_name="Daily Summary", index=False)
    st.download_button("📊 Download Excel Report", buf.getvalue(),
        f"Alkhair_Branch_Analytics_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    st.download_button("📄 Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        f"Alkhair_Branch_Data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv", use_container_width=True)

# ----------------------------------
# MAIN
# ----------------------------------
def main():
    apply_css()

    # Header: Hidden source; show only title + refresh button
    c1, c2 = st.columns([0.8, 0.2])
    with c1:
        st.markdown(f"<div class='page-title'>📊 {config.PAGE_TITLE}</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtle'>Executive summary & multi-branch analytics</div>", unsafe_allow_html=True)
    with c2:
        if st.button("🔄 Refresh", use_container_width=True):
            load_workbook_from_gsheet.clear()
            st.experimental_rerun()

    # Load & process (URL box is hidden; we use config.DEFAULT_PUBLISHED_URL)
    with st.spinner("Fetching data…"):
        sheets_map = load_workbook_from_gsheet(config.DEFAULT_PUBLISHED_URL)
        df = process_branch_data(sheets_map)
    if df.empty:
        st.error("No data found in the Google Sheet.")
        st.stop()

    # Global filters (RIGHT PANE will use filtered df)
    # Build the two-pane layout
    left, right = st.columns([0.27, 0.73], gap="large")

    # Compute KPIs on full DF first; then apply filters for right-pane visuals
    k_full = calc_kpis(df)

    with left:
        # LEFT PANE KPI CARDS (compact)
        st.markdown("<div class='kpi-card'><div class='kpi-title'>Total Sales</div>"
                    f"<div class='kpi-value'>SAR {k_full.get('total_sales_actual',0):,.0f}</div>"
                    f"<div class='kpi-delta'>vs Target: {k_full.get('overall_sales_percent',0):.1f}%</div></div>",
                    unsafe_allow_html=True)
        st.markdown("<div class='kpi-card'><div class='kpi-title'>Target Sales</div>"
                    f"<div class='kpi-value'>SAR {k_full.get('total_sales_target',0):,.0f}</div>"
                    f"<div class='kpi-delta'>Variance: {k_full.get('total_sales_variance',0):,.0f}</div></div>",
                    unsafe_allow_html=True)
        st.markdown("<div class='kpi-card'><div class='kpi-title'>Baskets (NOB)</div>"
                    f"<div class='kpi-value'>{k_full.get('total_nob_actual',0):,.0f}</div>"
                    f"<div class='kpi-delta'>Achievement: {k_full.get('overall_nob_percent',0):.1f}%</div></div>",
                    unsafe_allow_html=True)
        st.markdown("<div class='kpi-card'><div class='kpi-title'>Avg Basket Value</div>"
                    f"<div class='kpi-value'>SAR {k_full.get('avg_abv_actual',0):,.2f}</div>"
                    f"<div class='kpi-delta'>vs Target: {k_full.get('overall_abv_percent',0):.1f}%</div></div>",
                    unsafe_allow_html=True)

        # Best/Worst branch
        if "branch_performance" in k_full and not k_full["branch_performance"].empty:
            bp = k_full["branch_performance"]
            best_b = bp["SalesPercent"].idxmax(); best_v = bp.loc[best_b,"SalesPercent"]
            worst_b = bp["SalesPercent"].idxmin(); worst_v = bp.loc[worst_b,"SalesPercent"]
            st.markdown("<div class='kpi-card'><div class='kpi-title'>Best Sales Achievement</div>"
                        f"<div class='kpi-value'>{best_b}</div>"
                        f"<div class='kpi-delta'>{best_v:.1f}%</div></div>", unsafe_allow_html=True)
            st.markdown("<div class='kpi-card'><div class='kpi-title'>Lowest Sales Achievement</div>"
                        f"<div class='kpi-value'>{worst_b}</div>"
                        f"<div class='kpi-delta'>{worst_v:.1f}%</div></div>", unsafe_allow_html=True)

        # Date range
        if k_full.get("date_range"):
            dr = k_full["date_range"]
            st.markdown("<div class='kpi-card'><div class='kpi-title'>Date Range</div>"
                        f"<div class='kpi-value'>{dr['start'].date()} → {dr['end'].date()}</div>"
                        f"<div class='kpi-delta'>{dr['days']} days of data</div></div>", unsafe_allow_html=True)

    # RIGHT PANE – filters & tabs
    with right:
        # Filters
        if "BranchName" in df.columns:
            all_branches = sorted([b for b in df["BranchName"].dropna().unique()])
            sel = st.multiselect("Filter Branches", all_branches, default=all_branches, help="Affects all reports on the right")
            if sel:
                df = df[df["BranchName"].isin(sel)].copy()

        if "Date" in df.columns and df["Date"].notna().any():
            cA, cB, _ = st.columns([1.2, 1.2, 2])
            with cA:
                start_d = st.date_input("Start Date", value=df["Date"].min().date())
            with cB:
                end_d = st.date_input("End Date", value=df["Date"].max().date())
            mask = (df["Date"].dt.date >= start_d) & (df["Date"].dt.date <= end_d)
            df = df.loc[mask].copy()

        k = calc_kpis(df)

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["🏠 Overview", "📈 Daily Trends", "🏆 Competition", "📥 Export", "📚 More"]
        )

        with tab1:
            render_overview(df, k)

        with tab2:
            render_daily(df)

        with tab3:
            if "branch_performance" in k and not k["branch_performance"].empty:
                bp = k["branch_performance"].copy()
                bp["Rank by Sales %"] = (-bp["SalesPercent"]).rank(method="dense").astype(int)
                st.dataframe(
                    bp.sort_values("SalesPercent", ascending=False)[
                        ["Rank by Sales %","SalesPercent","NOBPercent","ABVPercent","SalesActual","SalesTarget"]
                    ].rename(columns={
                        "SalesPercent":"Sales %","NOBPercent":"NOB %","ABVPercent":"ABV %",
                        "SalesActual":"Sales (Actual)","SalesTarget":"Sales (Target)"
                    }).round(1),
                    use_container_width=True,
                )
                st.plotly_chart(
                    chart_branch_comparison(k["branch_performance"]),
                    use_container_width=True,
                    config={"displayModeBar": False},
                    key="comp_analysis"
                )
            else:
                st.info("No branch performance data available.")

        with tab4:
            render_export(df, k)

        with tab5:
            st.info("Add custom reports here (e.g., item-wise trends, staff leaderboard, hourly metrics).")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("❌ Application Error. Please adjust filters or refresh.")
