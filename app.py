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
    SHOW_SUBTITLE = False

config = Config()
st.set_page_config(page_title=config.PAGE_TITLE, layout=config.LAYOUT, initial_sidebar_state="expanded")


# =========================================
# ENHANCED CSS WITH THEMES
# =========================================
def apply_enhanced_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800;900&display=swap');
        
        /* Root Variables */
        :root {
            --primary-color: #3b82f6;
            --secondary-color: #10b981;
            --bg-primary: #f8fafc;
            --bg-secondary: #ffffff;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --border-radius: 20px;
            --shadow: 0 10px 40px rgba(0,0,0,0.1);
            --gradient-primary: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
            --gradient-secondary: linear-gradient(135deg, #10b981 0%, #059669 100%);
        }

        html, body, [data-testid="stAppViewContainer"] { 
            font-family: 'Inter', sans-serif; 
            background: var(--bg-primary);
            color: var(--text-primary);
        }

        .main .block-container { 
            padding: 1.5rem 2rem; 
            max-width: 1500px; 
        }

        /* Enhanced Hero Section */
        .hero-enhanced {
            text-align: center;
            margin: 0 0 2rem 0;
            padding: 3rem 2rem;
            background: var(--gradient-primary);
            border-radius: 24px;
            color: white;
            position: relative;
            overflow: hidden;
            animation: slideInDown 0.8s ease-out;
        }

        .hero-enhanced::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.1'%3E%3Ccircle cx='30' cy='30' r='2'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
            opacity: 0.5;
        }

        .hero-title-enhanced {
            font-family: 'Poppins', sans-serif;
            font-weight: 800;
            font-size: clamp(28px, 4vw, 48px);
            letter-spacing: -0.02em;
            margin-bottom: 15px;
            text-shadow: 0 4px 20px rgba(0,0,0,0.3);
            position: relative;
            z-index: 1;
        }

        .hero-subtitle {
            font-size: 1.1rem;
            opacity: 0.9;
            font-weight: 400;
            position: relative;
            z-index: 1;
        }

        /* Enhanced Metrics */
        [data-testid="metric-container"] {
            background: var(--bg-secondary) !important;
            border-radius: var(--border-radius) !important;
            border: none !important;
            box-shadow: var(--shadow) !important;
            padding: 28px !important;
            position: relative !important;
            overflow: hidden !important;
            transition: all 0.3s ease !important;
            animation: slideInUp 0.6s ease-out;
        }

        [data-testid="metric-container"]:hover {
            transform: translateY(-8px);
            box-shadow: 0 20px 60px rgba(0,0,0,0.15) !important;
        }

        [data-testid="metric-container"]::before {
            content: '';
            position: absolute;
            top: -50%; right: -50%;
            width: 100px; height: 100px;
            background: linear-gradient(45deg, rgba(59,130,246,0.1), rgba(16,185,129,0.1));
            border-radius: 50%;
        }

        [data-testid="metric-container"] > div > div:first-child {
            font-family: 'Poppins', sans-serif !important;
            font-weight: 700 !important;
            font-size: 2rem !important;
            color: var(--text-primary) !important;
        }

        [data-testid="metric-container"] > div > div:nth-child(2) {
            color: var(--text-secondary) !important;
            font-weight: 500 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            font-size: 0.85rem !important;
        }

        /* Enhanced Cards */
        .enhanced-card {
            background: var(--bg-secondary);
            border-radius: var(--border-radius);
            padding: 24px;
            border: none;
            box-shadow: var(--shadow);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
            animation: slideInUp 0.8s ease-out;
        }

        .enhanced-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 4px;
            background: var(--gradient-primary);
            transform: translateX(-100%);
            transition: transform 0.3s ease;
        }

        .enhanced-card:hover::before {
            transform: translateX(0);
        }

        .enhanced-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 20px 60px rgba(0,0,0,0.15);
        }

        /* Enhanced Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
            border-right: none;
            min-width: 300px; 
            max-width: 350px;
        }

        [data-testid="stSidebar"] * {
            color: white !important;
        }

        [data-testid="stSidebar"] .block-container {
            padding: 20px 18px 22px;
        }

        .sidebar-title {
            display: flex;
            gap: 12px;
            align-items: center;
            font-weight: 900;
            font-size: 1.2rem;
            margin-bottom: 8px;
            color: white;
        }

        .sidebar-subtitle {
            color: #cbd5e1 !important;
            font-size: 0.9rem;
            margin-bottom: 20px;
            opacity: 0.8;
        }

        .sidebar-section {
            background: rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 16px;
            margin: 16px 0;
            backdrop-filter: blur(10px);
        }

        .sidebar-section-title {
            font-size: 0.85rem;
            font-weight: 800;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            color: #e2e8f0;
            margin-bottom: 12px;
        }

        [data-testid="stSidebar"] .stButton > button {
            background: rgba(59, 130, 246, 0.2) !important;
            color: white !important;
            border: 1px solid rgba(59, 130, 246, 0.3) !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            background: rgba(59, 130, 246, 0.4) !important;
            transform: translateY(-2px) !important;
        }

        [data-testid="stSidebar"] .stCheckbox {
            margin: 8px 0;
        }

        [data-testid="stSidebar"] .stSelectbox > div > div {
            background: rgba(255,255,255,0.1) !important;
            border: 1px solid rgba(255,255,255,0.2) !important;
            border-radius: 8px !important;
        }

        /* Enhanced Tabs */
        .stTabs [data-baseweb="tab-list"] { 
            gap: 8px; 
            border-bottom: none; 
            margin-bottom: 24px;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(59, 130, 246, 0.1) !important;
            border: 2px solid transparent !important;
            border-radius: 16px !important;
            padding: 12px 24px !important;
            font-weight: 600 !important;
            color: var(--text-primary) !important;
            transition: all 0.3s ease !important;
            margin-right: 8px !important;
        }

        .stTabs [data-baseweb="tab"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }

        .stTabs [aria-selected="true"] { 
            background: var(--gradient-primary) !important;
            color: white !important; 
            box-shadow: 0 8px 25px rgba(59,130,246,0.4) !important;
        }

        /* Chart Containers */
        .chart-container {
            background: var(--bg-secondary);
            border-radius: var(--border-radius);
            padding: 32px;
            box-shadow: var(--shadow);
            margin: 24px 0;
            animation: fadeIn 1s ease-out;
        }

        .chart-title {
            font-family: 'Poppins', sans-serif;
            font-weight: 700;
            font-size: 1.4rem;
            color: var(--text-primary);
            margin-bottom: 24px;
            text-align: center;
        }

        /* Liquidity Panel */
        .liquidity-panel-enhanced {
            background: var(--bg-secondary);
            border: none;
            border-radius: 24px;
            padding: 32px;
            box-shadow: var(--shadow);
            position: relative;
            overflow: hidden;
            animation: slideInRight 0.8s ease-out;
        }

        .liquidity-panel-enhanced::before {
            content: '💧';
            position: absolute;
            top: -20px; right: -20px;
            font-size: 120px;
            opacity: 0.05;
            transform: rotate(15deg);
        }

        .liquidity-title-enhanced {
            font-weight: 900;
            font-size: 1.3rem;
            color: var(--text-primary);
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .liquidity-metric-enhanced {
            margin: 20px 0;
            padding: 20px;
            background: rgba(59, 130, 246, 0.05);
            border-radius: 12px;
            border-left: 4px solid var(--primary-color);
        }

        .liquidity-label-enhanced {
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-bottom: 8px;
            font-weight: 500;
        }

        .liquidity-value-enhanced {
            font-family: 'Poppins', sans-serif;
            font-weight: 800;
            font-size: 1.8rem;
            color: var(--text-primary);
        }

        /* Status Pills */
        .status-pill {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 0.85rem;
        }

        .status-pill.excellent { background: #ecfdf5; color: #059669; }
        .status-pill.good { background: #eff6ff; color: #2563eb; }
        .status-pill.warning { background: #fffbeb; color: #d97706; }
        .status-pill.danger { background: #fef2f2; color: #dc2626; }

        /* Animations */
        @keyframes slideInDown {
            from { opacity: 0; transform: translateY(-30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes slideInUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes slideInRight {
            from { opacity: 0; transform: translateX(30px); }
            to { opacity: 1; transform: translateX(0); }
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .main .block-container { padding: 1rem; }
            .hero-enhanced { padding: 2rem 1rem; }
            [data-testid="metric-container"] { padding: 20px !important; }
            .enhanced-card { padding: 16px; }
            [data-testid="stSidebar"] { min-width: 280px; }
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { 
            background: rgba(59, 130, 246, 0.1); 
            border-radius: 4px; 
        }
        ::-webkit-scrollbar-thumb { 
            background: var(--gradient-primary); 
            border-radius: 4px; 
        }
        ::-webkit-scrollbar-thumb:hover { 
            background: var(--gradient-secondary); 
        }

        /* Data Tables */
        .stDataFrame {
            background: var(--bg-secondary);
            border-radius: var(--border-radius);
            padding: 20px;
            box-shadow: var(--shadow);
            overflow: auto;
        }

        /* Expander */
        [data-testid="stExpander"] {
            background: var(--bg-secondary);
            border-radius: var(--border-radius);
            box-shadow: var(--shadow);
            border: none;
            margin: 20px 0;
        }

        [data-testid="stExpander"] > div:first-child {
            background: var(--gradient-primary);
            color: white;
            font-weight: 600;
            border-radius: var(--border-radius) var(--border-radius) 0 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================
# LOADERS / HELPERS (SAME AS ORIGINAL)
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
        "BANK": "Bank", "Bank": "Bank", "CASH": "Cash", "Cash": "Cash",
        "TOTAL LIQUIDITY": "TotalLiquidity", "Total Liquidity": "TotalLiquidity",
        "Change in Liquidity": "ChangeLiquidity", "% of Change": "PctChange",
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
            "SalesTarget", "SalesActual", "SalesPercent", "NOBTarget", "NOBActual", "NOBPercent",
            "ABVTarget", "ABVActual", "ABVPercent", "Bank", "Cash", "TotalLiquidity",
            "ChangeLiquidity", "PctChange",
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
        k["avg_abv_actual"] / k["avg_abv_target"] * 100 if "ABVTarget" in df and k["avg_abv_target"] > 0 else 0.0
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
# ENHANCED CHARTS
# =========================================
def _enhanced_metric_area(df: pd.DataFrame, y_col: str, title: str, *, show_target: bool = True) -> go.Figure:
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
        fig.add_trace(
            go.Scatter(
                x=d_actual["Date"], y=d_actual[y_col],
                name=f"{br} - Actual",
                mode="lines+markers",
                line=dict(width=3, color=color),
                fill="tonexty" if i > 0 else "tozeroy",
                fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)",
                hovertemplate=f"<b>{br}</b><br>Date: %{{x|%Y-%m-%d}}<br>Actual: %{{y:,.0f}}<extra></extra>",
            )
        )
        if daily_target is not None:
            d_target = daily_target[daily_target["BranchName"] == br]
            fig.add_trace(
                go.Scatter(
                    x=d_target["Date"], y=d_target[target_col],
                    name=f"{br} - Target",
                    mode="lines",
                    line=dict(width=2, color=color, dash="dash"),
                    hovertemplate=f"<b>{br}</b><br>Date: %{{x|%Y-%m-%d}}<br>Target: %{{y:,.0f}}<extra></extra>",
                    showlegend=False,
                )
            )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, family="Poppins, sans-serif", color="#1e293b")),
        height=450,
        showlegend=True,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        xaxis=dict(title="Date", showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
        yaxis=dict(title="Value", showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
        autosize=True,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def _enhanced_branch_comparison_chart(bp: pd.DataFrame) -> go.Figure:
    if bp.empty:
        return go.Figure()
    
    metrics = {"SalesPercent": "Sales %", "NOBPercent": "NOB %", "ABVPercent": "ABV %"}
    fig = go.Figure()
    x = bp.index.tolist()
    colors = ["#3b82f6", "#10b981", "#f59e0b"]
    
    for i, (col, label) in enumerate(metrics.items()):
        y = bp[col].tolist()
        fig.add_trace(
            go.Bar(
                x=x, y=y,
                name=label,
                marker=dict(
                    color=colors[i % len(colors)],
                    line=dict(width=0)
                ),
                hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:.1f}}%<extra></extra>"
            )
        )
    
    fig.update_layout(
        barmode="group",
        title=dict(text="Branch Performance Comparison", font=dict(size=20, family="Poppins, sans-serif")),
        xaxis=dict(title="Branch", showgrid=False),
        yaxis=dict(title="Percent", showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
        height=450,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        autosize=True,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def _enhanced_liquidity_trend_fig(daily: pd.DataFrame, title: str = "Total Liquidity Trend") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily["Date"],
            y=daily["TotalLiquidity"],
            mode="lines+markers",
            line=dict(width=4, color="#3b82f6", shape="spline"),
            marker=dict(size=8, color="#1d4ed8"),
            fill="tozeroy",
            fillcolor="rgba(59, 130, 246, 0.1)",
            hovertemplate="Date: %{x|%b %d, %Y}<br>Liquidity: SAR %{y:,.0f}<extra></extra>",
            name="Total Liquidity",
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, family="Poppins, sans-serif", color="#1e293b")),
        height=450,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Date", showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
        yaxis=dict(title="Liquidity (SAR)", showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
        autosize=True,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


# =========================================
# ENHANCED RENDERING FUNCTIONS
# =========================================
def render_enhanced_hero():
    """Enhanced hero section with modern styling"""
    st.markdown(
        """
        <div class="hero-enhanced">
            <div class="hero-title-enhanced">
                📊 Al Khair Business Performance
            </div>
            <div class="hero-subtitle">
                Real-time insights • Advanced analytics • Performance tracking
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_enhanced_branch_cards(bp: pd.DataFrame):
    """Enhanced branch cards with modern styling and animations"""
    st.markdown("### 🏪 Branch Overview")
    if bp.empty:
        st.info("No branch summary available.")
        return
    
    cols_per_row = 3
    items = list(bp.index)
    for i in range(0, len(items), cols_per_row):
        row = st.columns(cols_per_row, gap="large")
        for j, br in enumerate(items[i : i + cols_per_row]):
            if j < len(row):
                with row[j]:
                    r = bp.loc[br]
                    avg_pct = float(np.nanmean([r.get("SalesPercent", 0), r.get("NOBPercent", 0), r.get("ABVPercent", 0)]) or 0)
                    
                    # Determine status class
                    if avg_pct >= 95:
                        status_class = "excellent"
                    elif avg_pct >= 85:
                        status_class = "good"
                    elif avg_pct >= 75:
                        status_class = "warning"
                    else:
                        status_class = "danger"
                    
                    color = _branch_color_by_name(br)
                    
                    st.markdown(
                        f"""
                        <div class="enhanced-card">
                            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px">
                                <h3 style="margin:0;font-weight:700;font-size:1.2rem;color:var(--text-primary)">{br}</h3>
                                <div class="status-pill {status_class}">{avg_pct:.1f}%</div>
                            </div>
                            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px">
                                <div style="text-align:center;padding:16px 12px;background:rgba(59,130,246,0.05);border-radius:12px;border-left:4px solid {color}">
                                    <div style="font-size:0.8rem;color:var(--text-secondary);margin-bottom:6px;font-weight:500">Sales %</div>
                                    <div style="font-weight:700;color:var(--text-primary);font-size:1.1rem">{r.get('SalesPercent',0):.1f}%</div>
                                </div>
                                <div style="text-align:center;padding:16px 12px;background:rgba(16,185,129,0.05);border-radius:12px;border-left:4px solid {color}">
                                    <div style="font-size:0.8rem;color:var(--text-secondary);margin-bottom:6px;font-weight:500">NOB %</div>
                                    <div style="font-weight:700;color:var(--text-primary);font-size:1.1rem">{r.get('NOBPercent',0):.1f}%</div>
                                </div>
                                <div style="text-align:center;padding:16px 12px;background:rgba(245,158,11,0.05);border-radius:12px;border-left:4px solid {color}">
                                    <div style="font-size:0.8rem;color:var(--text-secondary);margin-bottom:6px;font-weight:500">ABV %</div>
                                    <div style="font-weight:700;color:var(--text-primary);font-size:1.1rem">{r.get('ABVPercent',0):.1f}%</div>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


def render_enhanced_liquidity_tab(df: pd.DataFrame):
    """Enhanced liquidity tab with modern styling"""
    if df.empty or "Date" not in df or "TotalLiquidity" not in df:
        st.info("Liquidity columns not found. Include 'TOTAL LIQUIDITY' in your sheet.")
        return

    d = df.dropna(subset=["Date", "TotalLiquidity"]).copy()
    if d.empty:
        st.info("No liquidity rows in the selected range.")
        return

    daily = d.groupby("Date", as_index=False)["TotalLiquidity"].sum().sort_values("Date")
    last30 = daily.tail(30).copy()

    current = float(last30["TotalLiquidity"].iloc[-1]) if len(last30) >= 1 else np.nan
    prev = float(last30["TotalLiquidity"].iloc[-2]) if len(last30) >= 2 else np.nan
    trend_pct = ((current - prev) / prev * 100.0) if (len(last30) >= 2 and not np.isnan(prev) and prev != 0) else np.nan

    max_30 = float(last30["TotalLiquidity"].max()) if not last30.empty else np.nan
    min_30 = float(last30["TotalLiquidity"].min()) if not last30.empty else np.nan
    avg_30 = float(last30["TotalLiquidity"].mean()) if not last30.empty else np.nan

    col_chart, col_metrics = st.columns([3, 1], gap="large")

    with col_chart:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">💧 Liquidity Trend Analysis</div>', unsafe_allow_html=True)
        st.plotly_chart(_enhanced_liquidity_trend_fig(last30), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_metrics:
        def _fmt(n: Optional[float]) -> str:
            if n is None or np.isnan(n):
                return "—"
            return f"{n:,.0f}"

        trend_display = ("+" if (not np.isnan(trend_pct) and trend_pct >= 0) else "") + (f"{trend_pct:.1f}%" if not np.isnan(trend_pct) else "—")
        
        st.markdown(
            f"""
            <div class="liquidity-panel-enhanced">
                <div class="liquidity-title-enhanced">📊 Liquidity Metrics</div>
                
                <div class="liquidity-metric-enhanced">
                    <div class="liquidity-label-enhanced">Current Liquidity</div>
                    <div class="liquidity-value-enhanced">SAR {_fmt(current)}</div>
                </div>
                
                <div class="liquidity-metric-enhanced">
                    <div class="liquidity-label-enhanced">Trend (Last Period)</div>
                    <div class="liquidity-value-enhanced" style="color: {'#10b981' if not np.isnan(trend_pct) and trend_pct >= 0 else '#ef4444'}">{trend_display}</div>
                </div>
                
                <div class="liquidity-metric-enhanced">
                    <div class="liquidity-label-enhanced">Statistics (30 days)</div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px">
                        <div>
                            <div style="font-size:0.8rem;color:var(--text-secondary);margin-bottom:4px">Max</div>
                            <div style="font-weight:700;color:var(--text-primary)">SAR {_fmt(max_30)}</div>
                        </div>
                        <div>
                            <div style="font-size:0.8rem;color:var(--text-secondary);margin-bottom:4px">Min</div>
                            <div style="font-weight:700;color:var(--text-primary)">SAR {_fmt(min_30)}</div>
                        </div>
                        <div style="grid-column:1/-1">
                            <div style="font-size:0.8rem;color:var(--text-secondary);margin-bottom:4px">Average</div>
                            <div style="font-weight:700;color:var(--text-primary)">SAR {_fmt(avg_30)}</div>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_enhanced_overview(df: pd.DataFrame, k: Dict[str, Any]):
    """Enhanced overview section with modern metrics"""
    st.markdown("### 🏆 Overall Performance")
    
    c1, c2, c3, c4, c5 = st.columns(5, gap="medium")
    
    # Enhanced metrics with better formatting
    with c1:
        variance_color = "normal" if k.get("overall_sales_percent", 0) >= config.TARGETS["sales_achievement"] else "inverse"
        st.metric(
            "💰 Total Sales", 
            f"SAR {k.get('total_sales_actual',0):,.0f}", 
            delta=f"{k.get('overall_sales_percent',0):.1f}% of target",
            delta_color=variance_color
        )
    
    with c2:
        variance_color = "normal" if k.get("total_sales_variance", 0) >= 0 else "inverse"
        st.metric(
            "📊 Sales Variance",
            f"SAR {k.get('total_sales_variance',0):,.0f}",
            delta=f"{k.get('overall_sales_percent',0)-100:+.1f}% vs target",
            delta_color=variance_color,
        )
    
    with c3:
        nob_color = "normal" if k.get("overall_nob_percent", 0) >= config.TARGETS["nob_achievement"] else "inverse"
        st.metric(
            "🛍️ Total Baskets", 
            f"{k.get('total_nob_actual',0):,.0f}", 
            delta=f"{k.get('overall_nob_percent',0):.1f}% achievement", 
            delta_color=nob_color
        )
    
    with c4:
        abv_color = "normal" if k.get("overall_abv_percent", 0) >= config.TARGETS["abv_achievement"] else "inverse"
        st.metric(
            "💎 Avg Basket Value", 
            f"SAR {k.get('avg_abv_actual',0):,.2f}", 
            delta=f"{k.get('overall_abv_percent',0):.1f}% vs target", 
            delta_color=abv_color
        )
    
    with c5:
        score_color = "normal" if k.get("performance_score", 0) >= 80 else "off"
        st.metric(
            "⭐ Performance Score", 
            f"{k.get('performance_score',0):.0f}/100", 
            delta="Weighted Score", 
            delta_color=score_color
        )

    # Enhanced branch cards
    if "branch_performance" in k and not k["branch_performance"].empty:
        render_enhanced_branch_cards(k["branch_performance"])

    # Enhanced comparison table
    if "branch_performance" in k and not k["branch_performance"].empty:
        st.markdown("### 📊 Performance Analysis")
        
        bp = k["branch_performance"].copy()
        df_table = (
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
            .round(1)
        )
        
        # Enhanced styling for the dataframe
        st.markdown('<div style="margin: 20px 0;">', unsafe_allow_html=True)
        st.dataframe(
            df_table.style.format({
                "Sales %": "{:.1f}%",
                "NOB %": "{:.1f}%", 
                "ABV %": "{:.1f}%",
                "Sales (Actual)": "SAR {:,.0f}",
                "Sales (Target)": "SAR {:,.0f}",
            }).background_gradient(subset=["Sales %", "NOB %", "ABV %"], cmap="RdYlGn"),
            use_container_width=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Enhanced comparison chart
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(_enhanced_branch_comparison_chart(bp), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)


def _branch_color_by_name(name: str) -> str:
    """Get branch color by name"""
    for _, meta in config.BRANCHES.items():
        if meta.get("name") == name:
            return meta.get("color", "#6b7280")
    return "#6b7280"


def render_enhanced_sidebar(df_all: pd.DataFrame, all_branches: List[str]):
    """Enhanced sidebar with modern styling"""
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-title">
                📊 <span>AL KHAIR DASHBOARD</span>
            </div>
            <div class="sidebar-subtitle">Filters affect all tabs and visualizations</div>
            """,
            unsafe_allow_html=True,
        )

        # Actions section
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-title">⚡ Quick Actions</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", use_container_width=True, help="Reload data from Google Sheets"):
                load_workbook_from_gsheet.clear()
                st.rerun()
        with col2:
            if st.button("📊 Export", use_container_width=True, help="Go to Export tab"):
                st.session_state.active_tab = "Export"
        st.markdown('</div>', unsafe_allow_html=True)

        # Branch selection
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-title">🏪 Branch Selection</div>', unsafe_allow_html=True)
        
        ca, cb = st.columns(2)
        if ca.button("All", use_container_width=True):
            st.session_state.selected_branches = list(all_branches)
            st.rerun()
        if cb.button("None", use_container_width=True):
            st.session_state.selected_branches = []
            st.rerun()

        sel: List[str] = []
        for i, b in enumerate(all_branches):
            checked = st.checkbox(
                b, 
                value=(b in st.session_state.selected_branches), 
                key=f"sb_br_{i}",
                help=f"Include {b} in analysis"
            )
            if checked:
                sel.append(b)
        st.session_state.selected_branches = sel
        st.markdown('</div>', unsafe_allow_html=True)

        # Date range selection
        df_for_bounds = df_all[df_all["BranchName"].isin(st.session_state.selected_branches)].copy() if st.session_state.selected_branches else df_all.copy()
        if "Date" in df_for_bounds.columns and df_for_bounds["Date"].notna().any():
            dmin_sb = df_for_bounds["Date"].min().date()
            dmax_sb = df_for_bounds["Date"].max().date()
        else:
            dmin_sb = dmax_sb = datetime.today().date()

        if "start_date" not in st.session_state:
            st.session_state.start_date = dmin_sb
        if "end_date" not in st.session_state:
            st.session_state.end_date = dmax_sb

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-title">📅 Date Range</div>', unsafe_allow_html=True)
        
        # Quick date range buttons
        col1, col2, col3 = st.columns(3)
        today = datetime.today().date()
        
        with col1:
            if st.button("7D", use_container_width=True, help="Last 7 days"):
                st.session_state.start_date = max(dmin_sb, today - timedelta(days=7))
                st.session_state.end_date = min(dmax_sb, today)
                st.rerun()
        with col2:
            if st.button("30D", use_container_width=True, help="Last 30 days"):
                st.session_state.start_date = max(dmin_sb, today - timedelta(days=30))
                st.session_state.end_date = min(dmax_sb, today)
                st.rerun()
        with col3:
            if st.button("All", use_container_width=True, help="All available data"):
                st.session_state.start_date = dmin_sb
                st.session_state.end_date = dmax_sb
                st.rerun()

        _sd = st.date_input("Start Date:", value=st.session_state.start_date, min_value=dmin_sb, max_value=dmax_sb, key="sb_sd")
        _ed = st.date_input("End Date:", value=st.session_state.end_date, min_value=dmin_sb, max_value=dmax_sb, key="sb_ed")
        
        st.session_state.start_date = max(dmin_sb, min(_sd, dmax_sb))
        st.session_state.end_date = max(dmin_sb, min(_ed, dmax_sb))
        if st.session_state.start_date > st.session_state.end_date:
            st.session_state.start_date, st.session_state.end_date = dmin_sb, dmax_sb
        st.markdown('</div>', unsafe_allow_html=True)

        # Data summary
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-title">📈 Data Summary</div>', unsafe_allow_html=True)
        
        filtered_df = df_all[df_all["BranchName"].isin(st.session_state.selected_branches)].copy() if st.session_state.selected_branches else df_all.copy()
        if "Date" in filtered_df.columns and filtered_df["Date"].notna().any():
            date_mask = (filtered_df["Date"].dt.date >= st.session_state.start_date) & (filtered_df["Date"].dt.date <= st.session_state.end_date)
            filtered_df = filtered_df.loc[date_mask]
        
        st.markdown(
            f"""
            <div style="color: #cbd5e1; font-size: 0.85rem; line-height: 1.6;">
                • <strong>{len(st.session_state.selected_branches)}</strong> branches selected<br>
                • <strong>{len(filtered_df):,}</strong> total records<br>
                • <strong>{(st.session_state.end_date - st.session_state.start_date).days + 1}</strong> days in range<br>
                • Updated: <strong>{datetime.now().strftime("%H:%M")}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)


# =========================================
# MAIN FUNCTION
# =========================================
def main():
    # Apply enhanced CSS
    apply_enhanced_css()

    # Initialize session state
    if "selected_branches" not in st.session_state:
        st.session_state.selected_branches = []
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Overview"

    # Load data
    try:
        sheets_map = load_workbook_from_gsheet(config.DEFAULT_PUBLISHED_URL)
        if not sheets_map:
            st.error("❌ No non-empty sheets found. Please check your Google Sheets URL.")
            st.stop()
            
        df_all = process_branch_data(sheets_map)
        if df_all.empty:
            st.error("❌ Could not process data. Check column names and sheet structure.")
            st.stop()

    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

    # Get all branches
    all_branches = sorted(df_all["BranchName"].dropna().unique()) if "BranchName" in df_all else []
    if not st.session_state.selected_branches:
        st.session_state.selected_branches = list(all_branches)

    # Render enhanced sidebar
    render_enhanced_sidebar(df_all, all_branches)

    # Render enhanced hero
    render_enhanced_hero()

    # Apply filters
    df = df_all.copy()
    if "BranchName" in df.columns and st.session_state.selected_branches:
        df = df[df["BranchName"].isin(st.session_state.selected_branches)].copy()
    if "Date" in df.columns and df["Date"].notna().any():
        mask = (df["Date"].dt.date >= st.session_state.start_date) & (df["Date"].dt.date <= st.session_state.end_date)
        df = df.loc[mask].copy()
        if df.empty:
            st.warning("⚠️ No rows in selected date range. Please adjust your filters.")
            st.stop()

    # Calculate KPIs
    k = calc_kpis(df)

    # Quick insights expander with enhanced styling
    with st.expander("⚡ Quick Insights & Alerts", expanded=True):
        insights: List[str] = []
        
        # Performance insights
        if "branch_performance" in k and isinstance(k["branch_performance"], pd.DataFrame) and not k["branch_performance"].empty:
            bp = k["branch_performance"]
            if not bp["SalesPercent"].isna().all():
                try:
                    best_branch = bp["SalesPercent"].idxmax()
                    worst_branch = bp["SalesPercent"].idxmin()
                    insights.append(f"🥇 **Best Performer**: {best_branch} with {bp['SalesPercent'].max():.1f}% sales achievement")
                    insights.append(f"🔻 **Needs Attention**: {worst_branch} with {bp['SalesPercent'].min():.1f}% sales achievement")
                except Exception:
                    pass
            
            # Target analysis
            below_target = bp[bp["SalesPercent"] < config.TARGETS["sales_achievement"]].index.tolist()
            if below_target:
                insights.append(f"⚠️ **Below 95% Target**: {', '.join(below_target)}")
            
            # NOB analysis
            low_nob = bp[bp["NOBPercent"] < config.TARGETS["nob_achievement"]].index.tolist()
            if low_nob:
                insights.append(f"📦 **Low Basket Count**: {', '.join(low_nob)} below 90% NOB target")

        # Overall performance insights
        if k.get("total_sales_variance", 0) < 0:
            insights.append(f"🟥 **Revenue Gap**: SAR {abs(k['total_sales_variance']):,.0f} below target ({k.get('overall_sales_percent', 0):.1f}%)")
        elif k.get("total_sales_variance", 0) > 0:
            insights.append(f"🟢 **Revenue Surplus**: SAR {k['total_sales_variance']:,.0f} above target ({k.get('overall_sales_percent', 0):.1f}%)")
        
        # Performance score insights
        score = k.get("performance_score", 0)
        if score >= 90:
            insights.append(f"⭐ **Excellent Performance**: Overall score of {score:.0f}/100")
        elif score >= 75:
            insights.append(f"👍 **Good Performance**: Overall score of {score:.0f}/100")
        elif score >= 60:
            insights.append(f"⚠️ **Average Performance**: Score of {score:.0f}/100 - room for improvement")
        else:
            insights.append(f"🚨 **Performance Alert**: Score of {score:.0f}/100 requires immediate attention")

        if insights:
            for insight in insights:
                st.markdown(f"- {insight}")
        else:
            st.success("✅ All metrics are performing well within expected ranges!")

    # Enhanced Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 Branch Overview", 
        "📈 Daily Trends", 
        "💧 Liquidity Analysis", 
        "📥 Export & Reports"
    ])

    with tab1:
        render_enhanced_overview(df, k)

    with tab2:
        st.markdown("#### 📊 Performance Trends Analysis")
        
        # Time period selector
        period_col, metric_col = st.columns([1, 3])
        
        with period_col:
            if "Date" in df.columns and df["Date"].notna().any():
                opts = ["Last 7 Days", "Last 30 Days", "Last 3 Months", "All Time"]
                choice = st.selectbox("📅 Time Period", opts, index=1, key="trend_window")
                
                today = (df["Date"].max() if df["Date"].notna().any() else pd.Timestamp.today()).date()
                start_date = {
                    "Last 7 Days": today - timedelta(days=7),
                    "Last 30 Days": today - timedelta(days=30),
                    "Last 3 Months": today - timedelta(days=90),
                }.get(choice, df["Date"].min().date())
                
                filtered_df = df[(df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= today)].copy()
            else:
                filtered_df = df.copy()
        
        # Metrics tabs
        m1, m2, m3 = st.tabs(["💰 Sales Performance", "🛍️ Basket Analysis", "💎 Value Analysis"])
        
        with m1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(
                _enhanced_metric_area(filtered_df, "ABVActual", "Average Basket Value (Actual vs Target)"), 
                use_container_width=True, 
                config={"displayModeBar": False}
            )
            st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        render_enhanced_liquidity_tab(df)

    with tab4:
        st.markdown("### 📥 Export & Reports")
        
        if df.empty:
            st.info("No data to export.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Data Export")
                
                # Excel export with multiple sheets
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                    # Main data
                    df.to_excel(writer, sheet_name="Raw Data", index=False)
                    
                    # Branch performance summary
                    if "branch_performance" in k and isinstance(k["branch_performance"], pd.DataFrame):
                        k["branch_performance"].to_excel(writer, sheet_name="Branch Summary")
                    
                    # KPIs summary
                    kpi_data = pd.DataFrame([{
                        "Metric": "Total Sales Actual",
                        "Value": k.get('total_sales_actual', 0),
                        "Target": k.get('total_sales_target', 0),
                        "Achievement %": k.get('overall_sales_percent', 0)
                    }, {
                        "Metric": "Total NOB Actual", 
                        "Value": k.get('total_nob_actual', 0),
                        "Target": k.get('total_nob_target', 0),
                        "Achievement %": k.get('overall_nob_percent', 0)
                    }, {
                        "Metric": "Avg ABV Actual",
                        "Value": k.get('avg_abv_actual', 0),
                        "Target": k.get('avg_abv_target', 0), 
                        "Achievement %": k.get('overall_abv_percent', 0)
                    }])
                    kpi_data.to_excel(writer, sheet_name="KPI Summary", index=False)

                st.download_button(
                    "📊 Download Excel Report",
                    buf.getvalue(),
                    f"AlKhair_Analytics_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    help="Complete Excel report with multiple sheets"
                )
                
                st.download_button(
                    "📄 Download CSV Data",
                    df.to_csv(index=False).encode("utf-8"),
                    f"AlKhair_Data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    help="Raw data in CSV format"
                )
            
            with col2:
                st.markdown("#### 📋 Report Summary")
                
                # Generate summary report
                report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
                date_range = f"{st.session_state.start_date} to {st.session_state.end_date}"
                selected_branches_text = ", ".join(st.session_state.selected_branches) if st.session_state.selected_branches else "All branches"
                
                summary_text = f"""
**Al Khair Performance Report**
Generated: {report_date}
Date Range: {date_range}
Branches: {selected_branches_text}

**Key Metrics:**
• Total Sales: SAR {k.get('total_sales_actual', 0):,.0f} ({k.get('overall_sales_percent', 0):.1f}% of target)
• Total Baskets: {k.get('total_nob_actual', 0):,.0f} ({k.get('overall_nob_percent', 0):.1f}% achievement)
• Avg Basket Value: SAR {k.get('avg_abv_actual', 0):.2f} ({k.get('overall_abv_percent', 0):.1f}% vs target)
• Performance Score: {k.get('performance_score', 0):.0f}/100

**Data Summary:**
• Total Records: {len(df):,}
• Date Range: {(st.session_state.end_date - st.session_state.start_date).days + 1} days
• Branches Analyzed: {len(st.session_state.selected_branches)}
                """
                
                st.download_button(
                    "📋 Download Summary Report",
                    summary_text,
                    f"AlKhair_Summary_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                    help="Text summary of key insights"
                )
                
                # Show preview of summary
                with st.expander("📖 Preview Summary Report", expanded=False):
                    st.text(summary_text)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: var(--text-secondary); font-size: 0.85rem; padding: 20px 0;">
            <strong>Al Khair Business Intelligence Dashboard</strong><br>
            Last updated: {timestamp} • Data refreshed automatically from Google Sheets<br>
            <em>Built with ❤️ using Streamlit & Plotly</em>
        </div>
        """.format(timestamp=datetime.now().strftime("%B %d, %Y at %I:%M %p")),
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.exception(e)
        
        # Show helpful error information
        st.markdown("### 🔧 Troubleshooting Tips")
        st.markdown("""
        - **Data Loading Issues**: Check your Google Sheets URL and ensure it's publicly accessible
        - **Column Mapping**: Verify column names match expected format (Sales Target, Sales Achievement, etc.)
        - **Date Format**: Ensure dates are in a recognizable format (YYYY-MM-DD, MM/DD/YYYY, etc.)
        - **Permission Issues**: Make sure the Google Sheet is published and accessible
        
        **Need Help?** Check the sidebar for refresh options or contact your system administrator.
        """)
(
                _enhanced_metric_area(filtered_df, "SalesActual", "Daily Sales Performance (Actual vs Target)"), 
                use_container_width=True, 
                config={"displayModeBar": False}
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
        with m2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(
                _enhanced_metric_area(filtered_df, "NOBActual", "Number of Baskets (Actual vs Target)"), 
                use_container_width=True, 
                config={"displayModeBar": False}
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
        with m3:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart
