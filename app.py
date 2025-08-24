from __future__ import annotations

import io
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List

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
    AUTO_REFRESH_URL = (
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vQG7boLWl2bNLCPR05NXv6EFpPPcFfXsiXPQ7rAGYr3q8Nkc2Ijg8BqEwVofcMLSg/pubhtml"
    )
    BRANCHES = {
        "Al khair - 102": {"name": "Al Khair", "code": "102", "color": "#6366f1", "icon": "🏢"},
        "Noora - 104": {"name": "Noora", "code": "104", "color": "#10b981", "icon": "🌟"},
        "Hamra - 109": {"name": "Hamra", "code": "109", "color": "#ef4444", "icon": "🏪"},
        "Magnus - 107": {"name": "Magnus", "code": "107", "color": "#8b5cf6", "icon": "⭐"},
    }
    TARGETS = {"sales_achievement": 95.0, "nob_achievement": 90.0, "abv_achievement": 90.0}
    REFRESH_INTERVAL = 300  # seconds

config = Config()
st.set_page_config(page_title=config.PAGE_TITLE, layout=config.LAYOUT, initial_sidebar_state="collapsed")

# =========================================
# MODERN CSS DESIGN
# =========================================
def apply_modern_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
        html, body, [data-testid="stAppViewContainer"] { 
            font-family: 'Poppins', sans-serif !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: #1a202c;
        }
        .stApp > header { background-color: transparent !important; }
        .main .block-container { padding: 0.5rem !important; max-width: 100% !important; margin: 0 !important; }

        .dashboard-container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 25px;
            padding: 2rem;
            margin: 1rem;
            backdrop-filter: blur(20px);
            box-shadow: 0 20px 60px rgba(0,0,0,0.1);
            border: 1px solid rgba(255,255,255,0.3);
        }

        .modern-header {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 20px;
            color: white;
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
        }
        .modern-header .header-title { font-size: 2rem; font-weight: 800; margin-bottom: 0.25rem; }
        .modern-header .header-subtitle { font-size: 0.95rem; opacity: 0.95; font-weight: 400; }

        .branch-toolbar {
            position: sticky; top: 0; z-index: 5;
            padding: .5rem; margin: .5rem 0 1rem 0; border-radius: 14px;
            background: rgba(255,255,255,.8); backdrop-filter: blur(8px);
            border: 1px solid rgba(0,0,0,.05);
        }

        .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.25rem; margin: 1.25rem 0; }
        .kpi-card {
            background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
            border-radius: 20px; padding: 1.5rem; text-align: center; border: 1px solid rgba(0,0,0,.05);
            transition: transform .2s ease, box-shadow .2s ease;
        }
        .kpi-card:hover { transform: translateY(-4px); box-shadow: 0 15px 35px rgba(0,0,0,0.12); }
        .kpi-value { font-size: 1.9rem; font-weight: 800; margin-bottom: .35rem; }
        .kpi-label { font-size: .95rem; color: #64748b; font-weight: 600; margin-bottom: .5rem; }
        .kpi-change { display: inline-block; padding: .4rem .8rem; border-radius: 999px; font-size: .85rem; font-weight: 700; }
        .kpi-change.positive { background: #ecfdf5; color: #065f46; }
        .kpi-change.negative { background: #fef2f2; color: #991b1b; }

        .performance-card {
            background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
            border-radius: 22px; padding: 1.25rem; border-left: 6px solid var(--performance-color, #6366f1);
            border: 1px solid rgba(0,0,0,.05); margin-bottom: 1rem;
        }
        .performance-title { font-weight: 800; display: flex; align-items: center; gap:.6rem; }
        .performance-score { font-weight: 800; color: white; background: var(--performance-color, #6366f1); padding:.35rem .7rem; border-radius: 999px; }

        .insights-container { background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border-radius: 20px; padding: 1.25rem; border: 1px solid rgba(99,102,241,0.2); }
        .insight-item { padding: .6rem .8rem; margin: .35rem 0; background: rgba(255,255,255,.8); border-radius: 12px; }

        .stTabs [data-baseweb="tab-list"] { background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); border-radius: 14px; padding: .25rem; gap: .25rem; }
        .stTabs [data-baseweb="tab"] { border-radius: 10px !important; padding: .75rem 1.1rem !important; font-weight: 700 !important; }
        .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important; color: white !important; box-shadow: 0 4px 12px rgba(99,102,241,.35) !important; }

        .refresh-fab {
            position: fixed; top: 18px; right: 18px; z-index: 1000;
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); color: white;
            border: none; border-radius: 50px; padding: .7rem 1rem; font-size: 1rem; cursor: pointer;
            box-shadow: 0 10px 20px rgba(99,102,241,0.3);
        }
        .auto-refresh { position: fixed; top: 18px; left: 18px; background: rgba(16,185,129,0.9); color: white; padding: .5rem .9rem; border-radius: 20px; font-size: .85rem; font-weight: 700; z-index: 1000; }

        #MainMenu {visibility: hidden;} footer {visibility: hidden;} .stDeployButton {display: none;}
        </style>
        """,
        unsafe_allow_html=True,
    )

# =========================================
# DATA FUNCTIONS
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

@st.cache_data(ttl=config.REFRESH_INTERVAL, show_spinner=False)
def auto_load_data() -> Dict[str, pd.DataFrame]:
    """Automatically load data with caching and auto-refresh"""
    url = _pubhtml_to_xlsx(config.AUTO_REFRESH_URL)
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        xls = pd.ExcelFile(io.BytesIO(r.content))
        sheets: Dict[str, pd.DataFrame] = {}
        for sn in xls.sheet_names:
            df = xls.parse(sn)
            df = df.dropna(how="all").dropna(axis=1, how="all")
            if not df.empty:
                sheets[sn] = df
        return sheets
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return {}

def process_branch_data(excel_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    combined: List[pd.DataFrame] = []
    mapping = {
        "Date": "Date", "Day": "Day",
        "Sales Target": "SalesTarget", "SalesTarget": "SalesTarget",
        "Sales Achivement": "SalesActual", "Sales Achievement": "SalesActual", "SalesActual": "SalesActual",
        "Sales %": "SalesPercent", " Sales %": "SalesPercent", "SalesPercent": "SalesPercent",
        "NOB Target": "NOBTarget", "NOBTarget": "NOBTarget",
        "NOB Achievemnet": "NOBActual", "NOB Achievement": "NOBActual", "NOBActual": "NOBActual",
        "NOB %": "NOBPercent", " NOB %": "NOBPercent", "NOBPercent": "NOBPercent",
        "ABV Target": "ABVTarget", "ABVTarget": "ABVTarget",
        "ABV Achievement": "ABVActual", " ABV Achievement": "ABVActual", "ABVActual": "ABVActual",
        "ABV %": "ABVPercent", "ABVPercent": "ABVPercent",
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
        d["BranchIcon"] = meta.get("icon", "🏪")
        d["BranchColor"] = meta.get("color", "#6366f1")

        if "Date" in d.columns:
            d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        for col in [
            "SalesTarget", "SalesActual", "SalesPercent",
            "NOBTarget", "NOBActual", "NOBPercent",
            "ABVTarget", "ABVActual", "ABVPercent",
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
            .agg({
                "SalesTarget": "sum", "SalesActual": "sum", "SalesPercent": "mean",
                "NOBTarget": "sum", "NOBActual": "sum", "NOBPercent": "mean",
                "ABVTarget": "mean", "ABVActual": "mean", "ABVPercent": "mean",
            })
            .round(2)
        )

    if "Date" in df.columns and df["Date"].notna().any():
        k["date_range"] = {
            "start": df["Date"].min(),
            "end": df["Date"].max(),
            "days": int(df["Date"].dt.date.nunique())
        }

    score = w = 0.0
    if k.get("overall_sales_percent", 0) > 0:
        score += min(k["overall_sales_percent"] / 100, 1.2) * 40; w += 40
    if k.get("overall_nob_percent", 0) > 0:
        score += min(k["overall_nob_percent"] / 100, 1.2) * 35; w += 35
    if k.get("overall_abv_percent", 0) > 0:
        score += min(k["overall_abv_percent"] / 100, 1.2) * 25; w += 25
    k["performance_score"] = (score / w * 100) if w > 0 else 0.0
    return k

# =========================================
# CHART FUNCTIONS
# =========================================
def _hex_to_rgb_tuple(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def create_modern_trend_chart(df: pd.DataFrame, metric_col: str, target_col: str, title: str) -> go.Figure:
    if df.empty or "Date" not in df.columns:
        return go.Figure()

    agg_func = "mean" if "ABV" in metric_col else "sum"
    cols = {metric_col: agg_func}
    if target_col in df.columns:
        cols[target_col] = agg_func if "ABV" in metric_col else "sum"

    daily_data = df.groupby(["Date", "BranchName", "BranchColor"]).agg(cols).reset_index()
    fig = go.Figure()

    for branch in sorted(daily_data["BranchName"].unique()):
        branch_data = daily_data[daily_data["BranchName"] == branch].sort_values("Date")
        if branch_data.empty:
            continue

        color_hex = branch_data["BranchColor"].iloc[0]
        r, g, b = _hex_to_rgb_tuple(color_hex)

        # Actual
        fig.add_trace(
            go.Scatter(
                x=branch_data["Date"],
                y=branch_data.get(metric_col, 0),
                name=f"{branch}",
                mode="lines+markers",
                line=dict(width=4, color=color_hex),
                marker=dict(size=7, color=color_hex),
                fill="tozeroy",
                fillcolor=f"rgba({r},{g},{b},0.10)",
                hovertemplate=f"<b>{branch}</b><br>Date: %{{x}}<br>Actual: %{{y:,.0f}}<extra></extra>",
            )
        )

        # Target
        if target_col in branch_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=branch_data["Date"],
                    y=branch_data[target_col],
                    name=f"{branch} Target",
                    mode="lines",
                    line=dict(width=2, color=color_hex, dash="dot"),
                    showlegend=False,
                    hovertemplate=f"<b>{branch} Target</b><br>Date: %{{x}}<br>Target: %{{y:,.0f}}<extra></extra>",
                )
            )

    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        height=500,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)"),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)"),
    )
    return fig

def create_performance_chart(df: pd.DataFrame, metric_col: str, target_col: str, title: str) -> go.Figure:
    if df.empty or "Date" not in df.columns or target_col not in df.columns:
        return go.Figure()

    agg_func = "mean" if "ABV" in metric_col else "sum"
    daily_data = df.groupby(["Date", "BranchName", "BranchColor"]).agg({
        metric_col: agg_func, target_col: agg_func
    }).reset_index()
    daily_data["Performance"] = (daily_data[metric_col] / daily_data[target_col] * 100).replace([np.inf, -np.inf], np.nan).fillna(0)

    fig = go.Figure()
    for branch in sorted(daily_data["BranchName"].unique()):
        branch_data = daily_data[daily_data["BranchName"] == branch].sort_values("Date")
        if branch_data.empty:
            continue
        color = branch_data["BranchColor"].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=branch_data["Date"],
                y=branch_data["Performance"],
                name=f"{branch}",
                mode="lines+markers",
                line=dict(width=3, color=color),
                marker=dict(size=6, color=color),
                hovertemplate=f"<b>{branch}</b><br>Date: %{{x}}<br>Performance: %{{y:.1f}}%<extra></extra>",
            )
        )

    fig.add_hline(y=100, line_dash="dash", line_color="rgba(239,68,68,0.8)", line_width=2,
                  annotation_text="Target (100%)", annotation_position="bottom right")
    fig.update_layout(
        title=dict(text=f"{title} Performance %", font=dict(size=18)),
        height=300, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)"),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)", title="Performance %"),
    )
    return fig

def create_radar_chart(bp: pd.DataFrame) -> go.Figure:
    if bp.empty:
        return go.Figure()
    fig = go.Figure()
    metrics = ["SalesPercent", "NOBPercent", "ABVPercent"]
    metric_labels = ["Sales %", "NOB %", "ABV %"]
    palette = ["#6366f1", "#10b981", "#ef4444", "#8b5cf6", "#f59e0b", "#14b8a6"]

    for i, branch_name in enumerate(bp.index):
        vals = [float(bp.loc[branch_name, m]) if m in bp.columns else 0 for m in metrics]
        vals.append(vals[0])
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=metric_labels + [metric_labels[0]], fill='toself',
            name=str(branch_name), line_color=palette[i % len(palette)],
            hovertemplate=f"<b>{branch_name}</b><br>%{{theta}}: %{{r:.1f}}%<extra></extra>"
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 120], showgrid=True, gridcolor="rgba(0,0,0,0.1)")),
        showlegend=True, title=dict(text="Branch Performance Comparison", font=dict(size=18)),
        height=400, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig

# =========================================
# UI COMPONENTS
# =========================================
def render_header():
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f"""
        <div class="modern-header">
            <div class="header-title">🏢 Al Khair Business Performance</div>
            <div class="header-subtitle">Real-time Business Analytics • Last Updated: {current_time}</div>
        </div>
        <div class="auto-refresh">🔄 Auto-refresh: {config.REFRESH_INTERVAL//60} min</div>
    """, unsafe_allow_html=True)

def render_branch_filter(df: pd.DataFrame) -> List[str]:
    if "BranchName" not in df.columns:
        return []
    st.markdown('<div class="branch-toolbar">', unsafe_allow_html=True)

    all_branches = list(df["BranchName"].dropna().unique())
    if "selected_branches" not in st.session_state:
        st.session_state.selected_branches = all_branches.copy()

    col_a, col_b, col_c = st.columns([2, 5, 3])
    with col_a:
        st.write("### 🏪 Branches")
    with col_b:
        btn_cols = st.columns(len(all_branches))
        for i, b in enumerate(all_branches):
            is_on = b in st.session_state.selected_branches
            if btn_cols[i].button(("✅ " if is_on else "➕ ") + b, key=f"branch_btn_{i}"):
                if is_on:
                    st.session_state.selected_branches.remove(b)
                else:
                    st.session_state.selected_branches.append(b)
                st.rerun()
    with col_c:
        c1, c2 = st.columns(2)
        if c1.button("Select all"):
            st.session_state.selected_branches = all_branches.copy()
            st.rerun()
        if c2.button("Clear"):
            st.session_state.selected_branches = []
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    return st.session_state.selected_branches

def render_kpi_cards(kpis: Dict[str, Any]):
    st.markdown("### 📊 Key Performance Indicators")
    kpi_data = [
        {
            "icon": "💰",
            "value": f"SAR {kpis.get('total_sales_actual', 0):,.0f}",
            "label": "Total Sales",
            "change": f"{kpis.get('overall_sales_percent', 0):.1f}% of Target",
            "positive": kpis.get("total_sales_variance", 0) >= 0
        },
        {
            "icon": "🛍️",
            "value": f"{kpis.get('total_nob_actual', 0):,.0f}",
            "label": "Total Baskets",
            "change": f"{kpis.get('overall_nob_percent', 0):.1f}% Achievement",
            "positive": kpis.get("overall_nob_percent", 0) >= 90
        },
        {
            "icon": "💎",
            "value": f"SAR {kpis.get('avg_abv_actual', 0):,.2f}",
            "label": "Avg Basket Value",
            "change": f"{kpis.get('overall_abv_percent', 0):.1f}% vs Target",
            "positive": kpis.get("overall_abv_percent", 0) >= 90
        },
        {
            "icon": "⭐",
            "value": f"{kpis.get('performance_score', 0):.0f}/100",
            "label": "Performance Score",
            "change": "Weighted Average",
            "positive": kpis.get("performance_score", 0) >= 80
        }
    ]
    cols = st.columns(len(kpi_data))
    colors = ["#10b981", "#6366f1", "#8b5cf6", "#ef4444"]
    for i, (col, kpi) in enumerate(zip(cols, kpi_data)):
        with col:
            change_class = "positive" if kpi["positive"] else "negative"
            st.markdown(f"""
                <div class="kpi-card">
                    <div style="font-size:1.6rem; opacity:.9;">{kpi['icon']}</div>
                    <div class="kpi-value" style="color:{colors[i]}">{kpi['value']}</div>
                    <div class="kpi-label">{kpi['label']}</div>
                    <div class="kpi-change {change_class}">{kpi['change']}</div>
                </div>
            """, unsafe_allow_html=True)

def render_performance_cards(bp: pd.DataFrame):
    if bp.empty: return
    st.markdown("### 🏪 Branch Performance")
    branch_colors = {"Al Khair": "#6366f1", "Noora": "#10b981", "Hamra": "#ef4444", "Magnus": "#8b5cf6"}
    branch_icons = {"Al Khair": "🏢", "Noora": "🌟", "Hamra": "🏪", "Magnus": "⭐"}

    cols = st.columns(min(len(bp), 2))
    for i, branch_name in enumerate(bp.index):
        with cols[i % 2]:
            bd = bp.loc[branch_name]
            icon = branch_icons.get(branch_name, "🏪")
            color = branch_colors.get(branch_name, "#6366f1")
            avg_perf = float(np.mean([bd.get("SalesPercent", 0), bd.get("NOBPercent", 0), bd.get("ABVPercent", 0)]))
            if avg_perf >= 95: status_color = "#10b981"
            elif avg_perf >= 85: status_color = "#6366f1"
            elif avg_perf >= 75: status_color = "#f59e0b"
            else: status_color = "#ef4444"

            st.markdown(f"""
                <div class="performance-card" style="--performance-color:{status_color}">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div class="performance-title"><span>{icon}</span>{branch_name}</div>
                        <div class="performance-score">{avg_perf:.1f}%</div>
                    </div>
                    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:.6rem;margin-top:.8rem;">
                        <div><div style="font-weight:800;color:{color}">{bd.get('SalesPercent', 0):.1f}%</div><div style="color:#64748b">Sales</div></div>
                        <div><div style="font-weight:800;color:{color}">{bd.get('NOBPercent', 0):.1f}%</div><div style="color:#64748b">Baskets</div></div>
                        <div><div style="font-weight:800;color:{color}">{bd.get('ABVPercent', 0):.1f}%</div><div style="color:#64748b">Basket Value</div></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

def render_insights(kpis: Dict[str, Any]):
    insights = []
    bp = kpis.get("branch_performance")
    if isinstance(bp, pd.DataFrame) and not bp.empty:
        best_sales = bp["SalesPercent"].idxmax()
        worst_sales = bp["SalesPercent"].idxmin()
        insights += [
            f"🥇 **Top Performer**: {best_sales} with {bp.loc[best_sales, 'SalesPercent']:.1f}% sales achievement",
            f"📉 **Needs Attention**: {worst_sales} with {bp.loc[worst_sales, 'SalesPercent']:.1f}% sales achievement",
            f"💰 **Total Revenue**: SAR {kpis.get('total_sales_actual', 0):,.0f} ({kpis.get('overall_sales_percent', 0):.1f}% of target)",
            f"🛍️ **Customer Traffic**: {kpis.get('total_nob_actual', 0):,.0f} baskets processed",
        ]
        if kpis.get("total_sales_variance", 0) < 0:
            insights.append(f"⚠️ **Revenue Gap**: SAR {abs(kpis['total_sales_variance']):,.0f} below target")

    if insights:
        st.markdown('<div class="insights-container"><h3 style="margin:0 0 .6rem 0;">💡 Key Insights</h3>', unsafe_allow_html=True)
        for s in insights:
            st.markdown(f'<div class="insight-item">{s}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# =========================================
# MAIN APPLICATION
# =========================================
def main():
    apply_modern_css()

    # Auto refresh every REFRESH_INTERVAL seconds
    st.autorefresh(interval=config.REFRESH_INTERVAL * 1000, key="auto_refresh_tick")

    st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)

    # Manual refresh
    if st.button("🔄 Refresh", key="refresh", type="primary"):
        auto_load_data.clear()
        st.rerun()

    render_header()

    with st.spinner("🔄 Loading latest data..."):
        sheets_map = auto_load_data()
    if not sheets_map:
        st.error("❌ Unable to load data. Please check your data source.")
        st.stop()

    df = process_branch_data(sheets_map)
    if df.empty:
        st.error("❌ No valid data found.")
        st.stop()

    selected_branches = render_branch_filter(df)
    if not selected_branches:
        st.warning("⚠️ Please select at least one branch.")
        st.stop()

    df_filtered = df[df["BranchName"].isin(selected_branches)].copy()

    # Date filter
    if "Date" in df_filtered.columns and df_filtered["Date"].notna().any():
        st.markdown("### 📅 Date Range")
        c1, c2 = st.columns(2)
        with c1:
            start_date = st.date_input("Start Date", value=df_filtered["Date"].min().date())
        with c2:
            end_date = st.date_input("End Date", value=df_filtered["Date"].max().date())
        mask = (df_filtered["Date"].dt.date >= start_date) & (df_filtered["Date"].dt.date <= end_date)
        df_filtered = df_filtered.loc[mask].copy()

    if df_filtered.empty:
        st.warning("⚠️ No data available for selected filters.")
        st.stop()

    kpis = calc_kpis(df_filtered)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Trends", "🎯 Analysis", "📤 Export"])

    with tab1:
        render_kpi_cards(kpis)
        st.markdown("---")
        if "branch_performance" in kpis and isinstance(kpis["branch_performance"], pd.DataFrame) and not kpis["branch_performance"].empty:
            render_performance_cards(kpis["branch_performance"])
        render_insights(kpis)

    with tab2:
        st.markdown("### 📈 Daily Trends")
        period_options = {"Last 7 Days": 7, "Last 14 Days": 14, "Last 30 Days": 30, "Last 90 Days": 90, "All Time": None}
        selected_period = st.selectbox("📅 Time Period", options=list(period_options.keys()), index=2)
        if period_options[selected_period] and "Date" in df_filtered.columns:
            cutoff_date = df_filtered["Date"].max() - timedelta(days=period_options[selected_period])
            trend_df = df_filtered[df_filtered["Date"] >= cutoff_date].copy()
        else:
            trend_df = df_filtered.copy()

        sales_tab, nob_tab, abv_tab = st.tabs(["💰 Sales", "🛍️ Baskets", "💎 Value"])

        with sales_tab:
            if "SalesActual" in trend_df:
                st.plotly_chart(create_modern_trend_chart(trend_df, "SalesActual", "SalesTarget", "Daily Sales Performance"),
                                use_container_width=True, config={"displayModeBar": False})
                if "SalesTarget" in trend_df:
                    st.plotly_chart(create_performance_chart(trend_df, "SalesActual", "SalesTarget", "Sales"),
                                    use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("No sales data available.")

        with nob_tab:
            if "NOBActual" in trend_df:
                st.plotly_chart(create_modern_trend_chart(trend_df, "NOBActual", "NOBTarget", "Daily Basket Count"),
                                use_container_width=True, config={"displayModeBar": False})
                if "NOBTarget" in trend_df:
                    st.plotly_chart(create_performance_chart(trend_df, "NOBActual", "NOBTarget", "NOB"),
                                    use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("No basket data available.")

        with abv_tab:
            if "ABVActual" in trend_df:
                st.plotly_chart(create_modern_trend_chart(trend_df, "ABVActual", "ABVTarget", "Average Basket Value"),
                                use_container_width=True, config={"displayModeBar": False})
                if "ABVTarget" in trend_df:
                    st.plotly_chart(create_performance_chart(trend_df, "ABVActual", "ABVTarget", "ABV"),
                                    use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("No basket value data available.")

    with tab3:
        st.markdown("### 🎯 Performance Analysis")
        c1, c2 = st.columns(2)
        with c1:
            bp = kpis.get("branch_performance")
            if isinstance(bp, pd.DataFrame) and not bp.empty:
                st.plotly_chart(create_radar_chart(bp), use_container_width=True, config={"displayModeBar": False})
        with c2:
            st.markdown("#### Performance Matrix")
            bp = kpis.get("branch_performance")
            if isinstance(bp, pd.DataFrame) and not bp.empty:
                matrix_data = []
                for branch in bp.index:
                    matrix_data.append({
                        "Branch": str(branch),
                        "Sales %": float(bp.loc[branch, "SalesPercent"]),
                        "NOB %": float(bp.loc[branch, "NOBPercent"]),
                        "ABV %": float(bp.loc[branch, "ABVPercent"]),
                        "Overall": float(np.mean([bp.loc[branch, "SalesPercent"], bp.loc[branch, "NOBPercent"], bp.loc[branch, "ABVPercent"]]))
                    })
                matrix_df = pd.DataFrame(matrix_data).round(1)
                st.dataframe(
                    matrix_df.style.background_gradient(
                        subset=["Sales %", "NOB %", "ABV %", "Overall"],
                        cmap="RdYlGn", vmin=70, vmax=110
                    ),
                    use_container_width=True
                )

    with tab4:
        st.markdown("### 📤 Export Data")
        c1, c2 = st.columns(2)
        with c1:
            if not df_filtered.empty:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                    df_filtered.to_excel(writer, sheet_name="Raw Data", index=False)
                    if isinstance(kpis.get("branch_performance"), pd.DataFrame):
                        kpis["branch_performance"].to_excel(writer, sheet_name="Summary")
                st.download_button(
                    "📊 Download Excel Report",
                    data=buffer.getvalue(),
                    file_name=f"AlKhair_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        with c2:
            if not df_filtered.empty:
                csv_data = df_filtered.to_csv(index=False)
                st.download_button(
                    "📄 Download CSV Data",
                    data=csv_data,
                    file_name=f"AlKhair_Data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.info("Please refresh the page if the issue persists.")
