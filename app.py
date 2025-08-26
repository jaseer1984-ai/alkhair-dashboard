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
    DEFAULT_PUBLISHED_URL = (
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vROaePKJN3XhRor42BU4Kgd9fCAgW7W8vWJbwjveQWoJy8HCRBUAYh2s0AxGsBa3w/pub?output=xlsx"
    )
    BRANCHES = {
        "Al khair - 102": {"name": "Al Khair", "code": "102", "color": "#3b82f6"},
        "Noora - 104": {"name": "Noora", "code": "104", "color": "#10b981"},
        "Hamra - 109": {"name": "Hamra", "code": "109", "color": "#f59e0b"},
        "Magnus - 107": {"name": "Magnus", "code": "107", "color": "#8b5cf6"},
    }
    TARGETS = {"sales_achievement": 95.0, "nob_achievement": 90.0, "abv_achievement": 90.0}
    SHOW_SUBTITLE = False  # hide "Branches • Rows • Date range"


config = Config()
st.set_page_config(page_title=config.PAGE_TITLE, layout=config.LAYOUT, initial_sidebar_state="expanded")


# =========================================
# CSS (center metrics + table headers + responsive)
# =========================================
def apply_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&display=swap');
        html, body, [data-testid="stAppViewContainer"] { font-family: 'Inter', sans-serif; }
        .main .block-container { padding: 1.2rem 2rem; max-width: 1500px; }

        /* ---- HERO HEADER ---- */
        .hero { text-align:center; margin: 0 0 10px 0; }
        .hero-bar { width: min(1100px,95%); height:18px; margin:8px auto 18px;
          border-radius:14px; background:linear-gradient(90deg,#3b82f6 0%, #4338ca 100%);
          box-shadow:0 12px 30px rgba(67,56,202,.25);}
        .hero-title { font-weight:900; font-size:clamp(26px,3.6vw,42px); letter-spacing:.2px;
          color:#111827; display:inline-flex; align-items:center; gap:.55rem; }
        .hero-emoji { font-size:1.15em; }
        .subtitle { color:#6b7280; font-size:.95rem; margin-bottom:1rem; text-align:center; }

        /* --- Metric tiles: centered + smaller numbers --- */
        [data-testid="metric-container"]{
          background:#fff!important;border-radius:16px!important;border:1px solid rgba(0,0,0,.06)!important;
          padding:16px!important;text-align:center!important;
        }
        [data-testid="stMetricValue"]{
          font-size:clamp(16px,2.2vw,22px)!important;
          line-height:1.2!important;
        }
        [data-testid="stMetricLabel"]{ font-size:.85rem!important;color:#6b7280!important; }

        .card { background:#fcfcfc; border:1px solid #f1f5f9; border-radius:16px; padding:16px; height:100%; }
        .pill { display:inline-block; padding:4px 10px; border-radius:999px; font-weight:700; font-size:.80rem; }
        .pill.excellent { background:#ecfdf5; color:#059669; }
        .pill.good      { background:#eff6ff; color:#2563eb; }
        .pill.warn      { background:#fffbeb; color:#d97706; }
        .pill.danger    { background:#fef2f2; color:#dc2626; }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: none; }
        .stTabs [data-baseweb="tab"] {
          border-radius: 10px 10px 0 0 !important; padding: 10px 18px !important; font-weight: 700 !important;
          background:#f3f4f6!important; color:#374151!important; border:1px solid #e5e7eb!important; border-bottom:none!important;
        }
        .stTabs [data-baseweb="tab"]:hover { background:#e5e7eb!important; }
        .stTabs [aria-selected="true"] { color:#fff!important; box-shadow:0 4px 12px rgba(0,0,0,.15); }

        /* Sidebar basics */
        [data-testid="stSidebar"]{ background:#f7f9fc; border-right:1px solid #e5e7eb; min-width:280px; max-width:320px; }
        [data-testid="stSidebar"] .block-container { padding:18px 16px 20px; }
        .sb-title{ display:flex; gap:10px; align-items:center; font-weight:900; font-size:1.05rem; }
        .sb-subtle{ color:#6b7280; font-size:.85rem; margin:6px 0 12px; }
        .sb-section{ font-size:.80rem; font-weight:800; letter-spacing:.02em; text-transform:uppercase; color:#64748b; margin:12px 4px 6px; }
        .sb-hr{ height:1px; background:#e5e7eb; margin:12px 0; border-radius:999px; }

        /* Center DataFrame headers only */
        .stDataFrame table thead tr th { text-align: center !important; }

        /* Responsive */
        [data-testid="stPlotlyChart"], [data-testid="stDataFrame"] { overflow:auto; }
        .js-plotly-plot, .plotly, .js-plotly-plot .plotly, .js-plotly-plot .main-svg { max-width:100%!important; }

        @media (max-width: 820px){
          .main .block-container { padding:.6rem .8rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================
# LOADERS / HELPERS
# =========================================
def _normalize_gsheet_url(url: str) -> str:
    u = (url or "").strip()
    u = re.sub(r"/pubhtml(\?.*)?$", "/pub?output=xlsx", u)
    u = re.sub(r"/pub\?output=csv(\&.*)?$", "/pub?output=xlsx", u)
    return u or url


@st.cache_data(show_spinner=False)
def load_workbook_from_gsheet(published_url: str) -> Dict[str, pd.DataFrame]:
    url = _normalize_gsheet_url((published_url or config.DEFAULT_PUBLISHED_URL).strip())
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    content_type = (r.headers.get("Content-Type") or "").lower()

    if "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" in content_type or "output=xlsx" in url:
        try:
            xls = pd.ExcelFile(io.BytesIO(r.content))
            out: Dict[str, pd.DataFrame] = {}
            for sn in xls.sheet_names:
                df = xls.parse(sn).dropna(how="all").dropna(axis=1, how="all")
                if not df.empty:
                    out[sn] = df
            if out:
                return out
        except Exception:
            pass

    try:
        df = pd.read_csv(io.BytesIO(r.content))
    except Exception:
        df = pd.read_csv(io.BytesIO(r.content), sep=";")
    df = df.dropna(how="all").dropna(axis=1, how="all")
    return {"Sheet1": df} if not df.empty else {}


def _parse_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False).str.strip(),
        errors="coerce",
    )


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
        "BANK": "Bank",
        "Bank": "Bank",
        "CASH": "Cash",
        "Cash": "Cash",
        "TOTAL LIQUIDITY": "TotalLiquidity",
        "Total Liquidity": "TotalLiquidity",
        "Change in Liquidity": "ChangeLiquidity",
        "% of Change": "PctChange",
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
            "SalesTarget", "SalesActual", "SalesPercent",
            "NOBTarget", "NOBActual", "NOBPercent",
            "ABVTarget", "ABVActual", "ABVPercent",
            "Bank", "Cash", "TotalLiquidity", "ChangeLiquidity", "PctChange",
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
    k["overall_sales_percent"] = k["total_sales_actual"] / k["total_sales_target"] * 100 if k["total_sales_target"] > 0 else 0.0
    k["total_nob_target"] = float(df.get("NOBTarget", pd.Series(dtype=float)).sum())
    k["total_nob_actual"] = float(df.get("NOBActual", pd.Series(dtype=float)).sum())
    k["overall_nob_percent"] = k["total_nob_actual"] / k["total_nob_target"] * 100 if k["total_nob_target"] > 0 else 0.0
    k["avg_abv_target"] = float(df.get("ABVTarget", pd.Series(dtype=float)).mean()) if "ABVTarget" in df else 0.0
    k["avg_abv_actual"] = float(df.get("ABVActual", pd.Series(dtype=float)).mean()) if "ABVActual" in df else 0.0
    k["overall_abv_percent"] = (k["avg_abv_actual"] / k["avg_abv_target"] * 100) if ("ABVTarget" in df and k["avg_abv_target"] > 0) else 0.0
    if "BranchName" in df.columns:
        k["branch_performance"] = (
            df.groupby("BranchName")
            .agg({
                "SalesTarget": "sum", "SalesActual": "sum", "SalesPercent": "mean",
                "NOBTarget": "sum", "NOBActual": "sum", "NOBPercent": "mean",
                "ABVTarget": "mean", "ABVActual": "mean", "ABVPercent": "mean",
            }).round(2)
        )
    if "Date" in df.columns and df["Date"].notna().any():
        k["date_range"] = {"start": df["Date"].min(), "end": df["Date"].max(), "days": int(df["Date"].dt.date.nunique())}
    score = w = 0.0
    if k.get("overall_sales_percent", 0) > 0: score += min(k["overall_sales_percent"]/100, 1.2)*40; w += 40
    if k.get("overall_nob_percent", 0) > 0:   score += min(k["overall_nob_percent"]/100, 1.2)*35; w += 35
    if k.get("overall_abv_percent", 0) > 0:   score += min(k["overall_abv_percent"]/100, 1.2)*25; w += 25
    k["performance_score"] = (score / w * 100) if w > 0 else 0.0
    return k


# =========================================
# CHARTS
# =========================================
def _metric_area(df: pd.DataFrame, y_col: str, title: str, *, show_target: bool = True) -> go.Figure:
    if df.empty or "Date" not in df.columns or df["Date"].isna().all():
        return go.Figure()
    agg_func = "sum" if y_col != "ABVActual" else "mean"
    daily_actual = df.groupby(["Date", "BranchName"]).agg({y_col: agg_func}).reset_index()
    target_col = {"SalesActual": "SalesTarget", "NOBActual": "NOBTarget", "ABVActual": "ABVTarget"}.get(y_col)
    daily_target: Optional[pd.DataFrame] = None
    if show_target and target_col and target_col in df.columns:
        agg_func_target = "sum" if y_col != "ABVActual" else "mean"
        daily_target = df.groupby(["Date", "BranchName"]).agg({target_col: agg_func_target}).reset_index()
    fig = go.Figure()
    palette = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444", "#14b8a6"]
    for i, br in enumerate(sorted(daily_actual["BranchName"].unique())):
        d_actual = daily_actual[daily_actual["BranchName"] == br]
        color = palette[i % len(palette)]
        fig.add_trace(go.Scatter(
            x=d_actual["Date"], y=d_actual[y_col], name=f"{br} - Actual",
            mode="lines+markers", line=dict(width=3, color=color), fill="tozeroy",
            hovertemplate=f"<b>{br}</b><br>Date: %{{x|%Y-%m-%d}}<br>Actual: %{{y:,.0f}}<extra></extra>",
        ))
        if daily_target is not None:
            d_target = daily_target[daily_target["BranchName"] == br]
            fig.add_trace(go.Scatter(
                x=d_target["Date"], y=d_target[target_col], name=f"{br} - Target",
                mode="lines", line=dict(width=2, color=color, dash="dash"),
                hovertemplate=f"<b>{br}</b><br>Date: %{{x|%Y-%m-%d}}<br>Target: %{{y:,.0f}}<extra></extra>",
                showlegend=False if len(sorted(daily_actual["BranchName"].unique())) > 1 else True,
            ))
    fig.update_layout(
        title=title, height=420, showlegend=True,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        xaxis_title="Date", yaxis_title="Value",
        autosize=True, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def _branch_comparison_chart(bp: pd.DataFrame) -> go.Figure:
    if bp.empty: return go.Figure()
    metrics = {"SalesPercent": "Sales %", "NOBPercent": "NOB %", "ABVPercent": "ABV %"}
    fig = go.Figure()
    x = bp.index.tolist()
    palette = ["#3b82f6", "#10b981", "#f59e0b"]
    for i, (col, label) in enumerate(metrics.items()):
        fig.add_trace(go.Bar(x=x, y=bp[col].tolist(), name=label, marker=dict(color=palette[i % len(palette)])))
    fig.update_layout(
        barmode="group", title="Branch Performance Comparison (%)",
        xaxis_title="Branch", yaxis_title="Percent", height=400,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        autosize=True, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# ========= Liquidity Trend (used in Liquidity tab) =========
def _liquidity_total_trend_fig(daily: pd.DataFrame, title: str = "Total Liquidity Trend") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["Date"], y=daily["TotalLiquidity"], mode="lines+markers",
        line=dict(width=4, color="#3b82f6"), marker=dict(size=6),
        hovertemplate="Date: %{x|%b %d, %Y}<br>Liquidity: SAR %{y:,.0f}<extra></extra>",
        name="Total Liquidity",
    ))
    fig.update_layout(
        title=title, height=420, showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Date", yaxis_title="Liquidity (SAR)",
        autosize=True, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# =========================================
# RENDER HELPERS
# =========================================
def branch_status_class(pct: float) -> str:
    if pct >= 95: return "excellent"
    if pct >= 85: return "good"
    if pct >= 75: return "warn"
    return "danger"


def _branch_color_by_name(name: str) -> str:
    for _, meta in config.BRANCHES.items():
        if meta.get("name") == name: return meta.get("color", "#6b7280")
    return "#6b7280"


def _balanced_rows(n: int) -> List[int]:
    """Return a list of column counts per row to avoid lonely last cards."""
    rows: List[int] = []
    remaining = n
    while remaining > 0:
        if remaining % 3 == 1 and remaining != 1:
            take = 2
        else:
            take = min(3, remaining)
        rows.append(take)
        remaining -= take
    return rows


def render_branch_cards(bp: pd.DataFrame):
    """Old-model look but with balanced rows to keep alignment nice."""
    st.markdown("### 🏪 Branch Overview")
    if bp.empty:
        st.info("No branch summary available.")
        return

    items = list(bp.index)
    layout = _balanced_rows(len(items))  # e.g., 4 -> [2,2], 5 -> [3,2], 7 -> [2,3,2]
    idx = 0

    for cols_in_row in layout:
        row = st.columns(cols_in_row, gap="medium")
        for j in range(cols_in_row):
            br = items[idx]; idx += 1
            with row[j]:
                r = bp.loc[br]
                avg_pct = float(np.nanmean([r.get("SalesPercent", 0), r.get("NOBPercent", 0), r.get("ABVPercent", 0)]) or 0)
                cls = branch_status_class(avg_pct)
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

    if "branch_performance" in k and not k["branch_performance"].empty:
        st.markdown("### 📊 Comparison Table")
        bp = k["branch_performance"].copy()
        df_table = (
            bp[["SalesPercent", "NOBPercent", "ABVPercent", "SalesActual", "SalesTarget"]]
            .rename(columns={
                "SalesPercent": "Sales %",
                "NOBPercent": "NOB %",
                "ABVPercent": "ABV %",
                "SalesActual": "Sales (Actual)",
                "SalesTarget": "Sales (Target)",
            })
            .round(1)
        )
        # Center headers only
        styler = (
            df_table.style
            .format({"Sales %": "{:,.1f}", "NOB %": "{:,.1f}", "ABV %": "{:,.1f}", "Sales (Actual)": "{:,.0f}", "Sales (Target)": "{:,.0f}"})
            .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])
        )
        st.dataframe(styler, use_container_width=True)

        st.markdown("### 📉 Branch Performance Comparison")
        st.plotly_chart(_branch_comparison_chart(bp), use_container_width=True, config={"displayModeBar": False})


# =========================================
# MAIN
# =========================================
def main():
    apply_css()

    # Load data
    sheets_map = load_workbook_from_gsheet(config.DEFAULT_PUBLISHED_URL)
    if not sheets_map:
        st.warning("No non-empty sheets found."); st.stop()
    df_all = process_branch_data(sheets_map)
    if df_all.empty:
        st.error("Could not process data. Check column names and sheet structure."); st.stop()

    # Sidebar filters
    all_branches = sorted(df_all["BranchName"].dropna().unique()) if "BranchName" in df_all else []
    if "selected_branches" not in st.session_state:
        st.session_state.selected_branches = list(all_branches)

    with st.sidebar:
        st.markdown('<div class="sb-title">📊 <span>AL KHAIR DASHBOARD</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-subtle">Filters affect all tabs.</div>', unsafe_allow_html=True)

        st.markdown('<div class="sb-section">Actions</div>', unsafe_allow_html=True)
        if st.button("🔄 Refresh Data", use_container_width=True):
            load_workbook_from_gsheet.clear(); st.rerun()
        st.markdown('<div class="sb-hr"></div>', unsafe_allow_html=True)

        st.markdown('<div class="sb-section">Branches</div>', unsafe_allow_html=True)
        ca, cb = st.columns(2)
        if ca.button("All", use_container_width=True): st.session_state.selected_branches = list(all_branches); st.rerun()
        if cb.button("None", use_container_width=True): st.session_state.selected_branches = []; st.rerun()
        sel = []
        for i, b in enumerate(all_branches):
            if st.checkbox(b, value=(b in st.session_state.selected_branches), key=f"sb_br_{i}"): sel.append(b)
        st.session_state.selected_branches = sel

        df_for_bounds = df_all[df_all["BranchName"].isin(st.session_state.selected_branches)].copy() if st.session_state.selected_branches else df_all.copy()
        if "Date" in df_for_bounds.columns and df_for_bounds["Date"].notna().any():
            dmin_sb, dmax_sb = df_for_bounds["Date"].min().date(), df_for_bounds["Date"].max().date()
        else:
            dmin_sb = dmax_sb = datetime.today().date()

        if "start_date" not in st.session_state: st.session_state.start_date = dmin_sb
        if "end_date" not in st.session_state:   st.session_state.end_date = dmax_sb

        st.markdown('<div class="sb-section">Custom Date</div>', unsafe_allow_html=True)
        _sd = st.date_input("Start:", value=st.session_state.start_date, min_value=dmin_sb, max_value=dmax_sb, key="sb_sd")
        _ed = st.date_input("End:", value=st.session_state.end_date,   min_value=dmin_sb, max_value=dmax_sb, key="sb_ed")
        st.session_state.start_date = max(dmin_sb, min(_sd, dmax_sb))
        st.session_state.end_date   = max(dmin_sb, min(_ed, dmax_sb))
        if st.session_state.start_date > st.session_state.end_date:
            st.session_state.start_date, st.session_state.end_date = dmin_sb, dmax_sb

    # Hero
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-bar"></div>
          <div class="hero-title"><span class="hero-emoji">📊</span>{config.PAGE_TITLE}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if config.SHOW_SUBTITLE and "Date" in df_all.columns and df_all["Date"].notna().any():
        date_span = f"{df_all['Date'].min().date()} → {df_all['Date'].max().date()}"
        st.markdown(f"<div class='subtitle'>Branches: {df_all['BranchName'].nunique()} • Rows: {len(df_all):,} • {date_span}</div>", unsafe_allow_html=True)

    # Apply filters
    df = df_all.copy()
    if "BranchName" in df.columns and st.session_state.selected_branches:
        df = df[df["BranchName"].isin(st.session_state.selected_branches)].copy()
    if "Date" in df.columns and df["Date"].notna().any():
        mask = (df["Date"].dt.date >= st.session_state.start_date) & (df["Date"].dt.date <= st.session_state.end_date)
        df = df.loc[mask].copy()
        if df.empty: st.warning("No rows in selected date range."); st.stop()

    # ======================
    # Quick Insights (with Liquidity)
    # ======================
    k = calc_kpis(df)
    exp_qi = st.expander("⚡ Quick Insights", expanded=True)
    with exp_qi:
        insights: List[str] = []

        # Branch-level sales highlights
        if "branch_performance" in k and isinstance(k["branch_performance"], pd.DataFrame) and not k["branch_performance"].empty:
            bp = k["branch_performance"]
            if not bp["SalesPercent"].isna().all():
                try:
                    insights.append(f"🥇 Best Sales %: {bp['SalesPercent'].idxmax()} — {bp['SalesPercent'].max():.1f}%")
                    insights.append(f"🔻 Lowest Sales %: {bp['SalesPercent'].idxmin()} — {bp['SalesPercent'].min():.1f}%")
                except Exception:
                    pass
            below = bp[bp["SalesPercent"] < config.TARGETS["sales_achievement"]].index.tolist()
            if below: insights.append("⚠️ Below 95% target: " + ", ".join(below))
        if k.get("total_sales_variance", 0) < 0:
            insights.append(f"🟥 Overall variance negative by SAR {abs(k['total_sales_variance']):,.0f}")

        # Liquidity insights (last 30 days in filtered range)
        if {"TotalLiquidity", "Date"}.issubset(df.columns):
            dliq = df.dropna(subset=["Date", "TotalLiquidity"]).copy()
            if not dliq.empty:
                daily = dliq.groupby("Date", as_index=False)["TotalLiquidity"].sum().sort_values("Date")
                last30 = daily.tail(30)
                if not last30.empty:
                    cur = float(last30["TotalLiquidity"].iloc[-1])
                    first = float(last30["TotalLiquidity"].iloc[0])
                    trend_30 = ((cur - first) / first * 100.0) if first not in (0, np.nan) else np.nan
                    dmax = last30.loc[last30["TotalLiquidity"].idxmax()]
                    dmin = last30.loc[last30["TotalLiquidity"].idxmin()]
                    insights.append(f"💧 Liquidity now: SAR {cur:,.0f} • 30-day trend: {('+' if trend_30>=0 else '') + (f'{trend_30:.1f}%' if not np.isnan(trend_30) else '—')}")
                    insights.append(f"📅 Max day (30d): {dmax['Date'].date()} — SAR {dmax['TotalLiquidity']:,.0f}")
                    insights.append(f"📅 Min day (30d): {dmin['Date'].date()} — SAR {dmin['TotalLiquidity']:,.0f}")

                if "ChangeLiquidity" in dliq.columns:
                    win = dliq[dliq["Date"].isin(last30["Date"])] if not last30.empty else dliq
                    by_branch = win.groupby("BranchName", dropna=True)["ChangeLiquidity"].sum().sort_values()
                    if not by_branch.empty:
                        down_br, down_val = by_branch.index[0], float(by_branch.iloc[0])
                        up_br, up_val = by_branch.index[-1], float(by_branch.iloc[-1])
                        if up_val > 0:
                            insights.append(f"🏦 Biggest ↑ branch (30d): {up_br} — SAR {up_val:,.0f}")
                        if down_val < 0:
                            insights.append(f"🏦 Biggest ↓ branch (30d): {down_br} — SAR {down_val:,.0f}")

        if insights: st.markdown("\n".join(f"- {line}" for line in insights))
        else:       st.caption("All metrics look healthy for the current selection.")

    # Tabs
    t1, t2, t3, t4 = st.tabs(["🏠 Branch Overview", "📈 Daily Trends", "💧 Liquidity", "📥 Export"])
    with t1:
        render_overview(df, k)

    with t2:
        st.markdown("#### Choose Metric")
        m1, m2, m3 = st.tabs(["💰 Sales", "🛍️ NOB", "💎 ABV"])
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
        with m1: st.plotly_chart(_metric_area(f, "SalesActual", "Daily Sales (Actual vs Target)"), use_container_width=True, config={"displayModeBar": False})
        with m2: st.plotly_chart(_metric_area(f, "NOBActual", "Number of Baskets (Actual vs Target)"), use_container_width=True, config={"displayModeBar": False})
        with m3: st.plotly_chart(_metric_area(f, "ABVActual", "Average Basket Value (Actual vs Target)"), use_container_width=True, config={"displayModeBar": False})

    with t3:
        if {"TotalLiquidity","Date"}.issubset(df.columns):
            dd = df.dropna(subset=["Date", "TotalLiquidity"]).groupby("Date", as_index=False)["TotalLiquidity"].sum().sort_values("Date")
            last30 = dd.tail(30)
            col_chart, col_metrics = st.columns([3,1], gap="large")
            with col_chart:
                st.markdown("#### 🧪 Liquidity Trend Analysis")
                st.plotly_chart(_liquidity_total_trend_fig(last30), use_container_width=True, config={"displayModeBar": False})
            with col_metrics:
                cur = float(last30["TotalLiquidity"].iloc[-1]) if not last30.empty else np.nan
                prev = float(last30["TotalLiquidity"].iloc[-2]) if len(last30)>=2 else np.nan
                trend = ((cur-prev)/prev*100) if (len(last30)>=2 and prev not in (0, np.nan)) else np.nan
                mx, mn, avg = (float(last30["TotalLiquidity"].max()) if not last30.empty else np.nan,
                               float(last30["TotalLiquidity"].min()) if not last30.empty else np.nan,
                               float(last30["TotalLiquidity"].mean()) if not last30.empty else np.nan)
                st.markdown(f"**Current:** {('SAR ' + format(cur,',.0f')) if not np.isnan(cur) else '—'}")
                st.markdown(f"**Trend:** {('+' if (not np.isnan(trend) and trend>=0) else '') + (f'{trend:.1f}%' if not np.isnan(trend) else '—')}")
                st.markdown(f"**Max (30d):** {('SAR ' + format(mx,',.0f')) if not np.isnan(mx) else '—'}")
                st.markdown(f"**Min (30d):** {('SAR ' + format(mn,',.0f')) if not np.isnan(mn) else '—'}")
                st.markdown(f"**Avg (30d):** {('SAR ' + format(avg,',.0f')) if not np.isnan(avg) else '—'}")
        else:
            st.info("Liquidity columns not found. Include 'TOTAL LIQUIDITY' in your sheet.")

    with t4:
        if df.empty:
            st.info("No data to export.")
        else:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                df.to_excel(writer, sheet_name="All Branches", index=False)
                if "branch_performance" in k and isinstance(k["branch_performance"], pd.DataFrame):
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
    except Exception as e:
        st.exception(e)
