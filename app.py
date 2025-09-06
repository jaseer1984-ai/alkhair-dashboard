from __future__ import annotations

import io
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
        "Al khair - 102": {"name": "Al Khair", "code": "102", "color": "#6366f1", "gradient": "linear-gradient(135deg, #6366f1, #8b5cf6)"},
        "Noora - 104": {"name": "Noora", "code": "104", "color": "#10b981", "gradient": "linear-gradient(135deg, #10b981, #059669)"},
        "Hamra - 109": {"name": "Hamra", "code": "109", "color": "#f59e0b", "gradient": "linear-gradient(135deg, #f59e0b, #d97706)"},
        "Magnus - 107": {"name": "Magnus", "code": "107", "color": "#8b5cf6", "gradient": "linear-gradient(135deg, #8b5cf6, #a855f7)"},
    }
    TARGETS = {"sales_achievement": 95.0, "nob_achievement": 90.0, "abv_achievement": 90.0}
    SHOW_SUBTITLE = False
    CARDS_PER_ROW = 3  # branch cards per row


config = Config()
st.set_page_config(page_title=config.PAGE_TITLE, layout=config.LAYOUT, initial_sidebar_state="expanded")


# =========================================
# ENHANCED CSS WITH MODERN DESIGN
# =========================================
def apply_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
        
        html, body, [data-testid="stAppViewContainer"] { 
            font-family: 'Inter', sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #1f2937;
        }

        /* Main container with glassmorphism */
        .main .block-container { 
            padding: 1.5rem 2rem; 
            max-width: 1900px; 
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            margin: 1rem auto;
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.1);
        }

        /* ---- ENHANCED HERO HEADER ---- */
        .hero { 
            text-align: center; 
            margin: 0 0 2rem 0; 
            position: relative;
            padding: 2rem 0;
        }
        
        .hero::before {
            content: '';
            position: absolute;
            top: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 100px;
            height: 4px;
            background: linear-gradient(90deg, #6366f1, #8b5cf6, #ec4899);
            border-radius: 2px;
            animation: shimmer 2s ease-in-out infinite;
        }
        
        @keyframes shimmer {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
        
        .hero-title { 
            font-weight: 900; 
            font-size: clamp(32px, 4vw, 48px); 
            letter-spacing: -0.5px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            display: inline-flex; 
            align-items: center; 
            gap: 1rem;
            margin-top: 1rem;
        }
        
        .hero-emoji { 
            font-size: 1.2em; 
            filter: drop-shadow(0 4px 8px rgba(0,0,0,0.1));
        }
        
        .subtitle { 
            color: #6b7280; 
            font-size: 1.1rem; 
            margin: 1rem 0 2rem; 
            text-align: center;
            font-weight: 400;
        }

        /* Enhanced Metric Containers */
        [data-testid="metric-container"] {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%) !important;
            border-radius: 20px !important;
            border: 1px solid rgba(226, 232, 240, 0.8) !important;
            padding: 24px !important;
            text-align: center !important;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08) !important;
            transition: all 0.3s ease !important;
            position: relative !important;
            overflow: hidden !important;
        }
        
        [data-testid="metric-container"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.12) !important;
        }
        
        [data-testid="metric-container"]::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #6366f1, #8b5cf6, #ec4899, #f59e0b);
            background-size: 200% 100%;
            animation: gradient-shift 3s ease infinite;
        }
        
        @keyframes gradient-shift {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
        
        [data-testid="stMetricValue"] { 
            font-size: clamp(18px, 2.5vw, 28px) !important; 
            line-height: 1.3 !important; 
            font-weight: 800 !important;
            color: #1f2937 !important;
        }
        
        [data-testid="stMetricLabel"] { 
            font-size: 0.9rem !important;
            color: #6b7280 !important; 
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Enhanced metric row layout */
        .metrics-row [data-testid="stHorizontalBlock"] { 
            display: flex; 
            flex-wrap: nowrap; 
            gap: 16px; 
        }
        .metrics-row [data-testid="stHorizontalBlock"] > div { 
            flex: 1 1 0 !important; 
            min-width: 200px; 
        }
        
        @media (max-width: 1024px) { 
            .metrics-row [data-testid="stHorizontalBlock"] { 
                flex-wrap: wrap; 
            } 
        }

        /* Enhanced Branch Cards */
        .card { 
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%); 
            border: 1px solid rgba(226, 232, 240, 0.6); 
            border-radius: 24px; 
            padding: 24px; 
            height: 100%; 
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        
        .card:hover {
            transform: translateY(-4px);
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
        }
        
        .branch-card { 
            display: flex; 
            flex-direction: column; 
            height: 100%; 
            position: relative; 
        }
        
        .branch-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }
        
        .branch-name {
            font-weight: 800;
            font-size: 1.25rem;
            color: #1f2937;
            letter-spacing: -0.25px;
        }
        
        .branch-metrics {
            display: grid; 
            grid-template-columns: repeat(3, 1fr);
            gap: 12px 16px; 
            margin-top: 16px; 
            text-align: center; 
            align-items: center;
        }
        
        .branch-metrics .label { 
            font-size: 0.75rem; 
            color: #64748b; 
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .branch-metrics .value { 
            font-weight: 800; 
            font-size: 1.1rem;
            color: #1f2937;
        }

        /* Enhanced Status Pills */
        .pill { 
            display: inline-flex;
            align-items: center;
            padding: 8px 16px; 
            border-radius: 50px; 
            font-weight: 700; 
            font-size: 0.8rem;
            letter-spacing: 0.25px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            transition: all 0.2s ease;
        }
        
        .pill:hover {
            transform: scale(1.05);
        }
        
        .pill.excellent { 
            background: linear-gradient(135deg, #10b981, #059669); 
            color: white;
        }
        .pill.good { 
            background: linear-gradient(135deg, #3b82f6, #2563eb); 
            color: white;
        }
        .pill.warn { 
            background: linear-gradient(135deg, #f59e0b, #d97706); 
            color: white;
        }
        .pill.danger { 
            background: linear-gradient(135deg, #ef4444, #dc2626); 
            color: white;
        }

        /* Enhanced Tabs */
        .stTabs [data-baseweb="tab-list"] { 
            gap: 4px; 
            border-bottom: none; 
            background: rgba(248, 250, 252, 0.8);
            padding: 8px;
            border-radius: 16px;
            backdrop-filter: blur(10px);
        }
        
        .stTabs [data-baseweb="tab"] {
            position: relative; 
            border-radius: 12px !important; 
            padding: 12px 24px !important;
            font-weight: 600 !important; 
            background: transparent !important; 
            color: #64748b !important;
            border: none !important;
            transition: all 0.3s ease !important;
        }
        
        .stTabs [data-baseweb="tab"] p { 
            color: inherit !important; 
        }
        
        .stTabs [data-baseweb="tab"]:hover { 
            background: rgba(255, 255, 255, 0.7) !important; 
            color: #374151 !important;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #6366f1, #8b5cf6) !important; 
            color: white !important;
            box-shadow: 0 8px 20px rgba(99, 102, 241, 0.3) !important;
        }
        
        .stTabs [aria-selected="true"] p { 
            color: white !important; 
        }

        /* Enhanced Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
            border-right: 1px solid rgba(226, 232, 240, 0.8);
            width: 280px; 
            min-width: 260px; 
            max-width: 300px;
            backdrop-filter: blur(20px);
        }
        
        [data-testid="stSidebar"] .block-container { 
            padding: 24px 16px; 
        }
        
        .sb-title { 
            display: flex; 
            gap: 12px; 
            align-items: center; 
            font-weight: 900; 
            font-size: 1.1rem;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .sb-subtle { 
            color: #6b7280; 
            font-size: 0.85rem; 
            margin: 8px 0 16px; 
            font-style: italic;
        }
        
        .sb-section { 
            font-size: 0.8rem; 
            font-weight: 800; 
            letter-spacing: 0.5px; 
            text-transform: uppercase; 
            color: #64748b; 
            margin: 20px 4px 8px; 
            position: relative;
        }
        
        .sb-section::after {
            content: '';
            position: absolute;
            bottom: -4px;
            left: 0;
            width: 24px;
            height: 2px;
            background: linear-gradient(90deg, #6366f1, #8b5cf6);
            border-radius: 1px;
        }
        
        .sb-hr { 
            height: 1px; 
            background: linear-gradient(90deg, transparent, #e2e8f0, transparent); 
            margin: 16px 0; 
            border-radius: 1px; 
        }

        /* Enhanced Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 12px 24px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 20px rgba(99, 102, 241, 0.4) !important;
        }

        /* Enhanced Checkboxes */
        .stCheckbox > label {
            background: rgba(255, 255, 255, 0.6) !important;
            padding: 8px 12px !important;
            border-radius: 10px !important;
            margin: 2px 0 !important;
            transition: all 0.2s ease !important;
        }
        
        .stCheckbox > label:hover {
            background: rgba(255, 255, 255, 0.9) !important;
        }

        /* Loading States */
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        .loading {
            animation: pulse 2s ease-in-out infinite;
        }

        /* Responsive improvements */
        @media (max-width: 768px) {
            .main .block-container {
                margin: 0.5rem;
                padding: 1rem;
                border-radius: 12px;
            }
            
            .hero {
                padding: 1rem 0;
            }
            
            .hero-title {
                font-size: clamp(24px, 6vw, 32px);
            }
            
            .branch-metrics {
                gap: 8px 12px;
            }
        }

        /* Chart enhancements */
        .js-plotly-plot, .plotly, .js-plotly-plot .plotly, .js-plotly-plot .main-svg { 
            max-width: 100% !important; 
            border-radius: 16px;
            overflow: hidden;
        }
        
        /* Data table enhancements */
        .stDataFrame {
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        }
        
        .stDataFrame table {
            border-radius: 16px;
            overflow: hidden;
        }
        
        .stDataFrame table thead tr th { 
            text-align: center !important;
            background: linear-gradient(135deg, #f8fafc, #e2e8f0) !important;
            font-weight: 700 !important;
            color: #374151 !important;
        }

        /* Insights section */
        .insights-container {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1));
            border: 1px solid rgba(99, 102, 241, 0.2);
            border-radius: 16px;
            padding: 20px;
            margin: 16px 0;
            backdrop-filter: blur(10px);
        }
        
        .insights-title {
            font-weight: 700;
            color: #374151;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        /* Success/Error states */
        .success-indicator {
            color: #10b981;
            background: rgba(16, 185, 129, 0.1);
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        
        .error-indicator {
            color: #ef4444;
            background: rgba(239, 68, 68, 0.1);
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================
# LOADERS / HELPERS (unchanged)
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
        "BANK": "Bank", "Bank": "Bank",
        "CASH": "Cash", "Cash": "Cash",
        "TOTAL LIQUIDITY": "TotalLiquidity", "Total Liquidity": "TotalLiquidity",
        "Change in Liquidity": "ChangeLiquidity", "% of Change": "PctChange",
    }
    for sheet_name, df in excel_data.items():
        if df.empty: continue
        d = df.copy()
        d.columns = d.columns.astype(str).str.strip()
        for old, new in mapping.items():
            if old in d.columns and new not in d.columns:
                d.rename(columns={old: new}, inplace=True)
        if "Date" not in d.columns:
            maybe = [c for c in d.columns if "date" in c.lower()]
            if maybe: d.rename(columns={maybe[0]: "Date"}, inplace=True)
        d["Branch"] = sheet_name
        meta = config.BRANCHES.get(sheet_name, {})
        d["BranchName"] = meta.get("name", sheet_name)
        d["BranchCode"] = meta.get("code", "000")
        if "Date" in d.columns: d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        for col in ["SalesTarget","SalesActual","SalesPercent","NOBTarget","NOBActual","NOBPercent",
                    "ABVTarget","ABVActual","ABVPercent","Bank","Cash","TotalLiquidity","ChangeLiquidity","PctChange"]:
            if col in d.columns: d[col] = _parse_numeric(d[col])
        combined.append(d)
    return pd.concat(combined, ignore_index=True) if combined else pd.DataFrame()


def calc_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty: return {}
    k: Dict[str, Any] = {}
    k["total_sales_target"] = float(df.get("SalesTarget", pd.Series(dtype=float)).sum())
    k["total_sales_actual"] = float(df.get("SalesActual", pd.Series(dtype=float)).sum())
    k["total_sales_variance"] = k["total_sales_actual"] - k["total_sales_target"]
    k["overall_sales_percent"] = (k["total_sales_actual"]/k["total_sales_target"]*100) if k["total_sales_target"] > 0 else 0.0
    k["total_nob_target"] = float(df.get("NOBTarget", pd.Series(dtype=float)).sum())
    k["total_nob_actual"] = float(df.get("NOBActual", pd.Series(dtype=float)).sum())
    k["overall_nob_percent"] = (k["total_nob_actual"]/k["total_nob_target"]*100) if k["total_nob_target"] > 0 else 0.0
    k["avg_abv_target"] = float(df.get("ABVTarget", pd.Series(dtype=float)).mean()) if "ABVTarget" in df else 0.0
    k["avg_abv_actual"] = float(df.get("ABVActual", pd.Series(dtype=float)).mean()) if "ABVActual" in df else 0.0
    k["overall_abv_percent"] = (k["avg_abv_actual"]/k["avg_abv_target"]*100) if ("ABVTarget" in df and k["avg_abv_target"]>0) else 0.0
    if "BranchName" in df.columns:
        k["branch_performance"] = (
            df.groupby("BranchName")
              .agg({"SalesTarget":"sum","SalesActual":"sum","SalesPercent":"mean",
                    "NOBTarget":"sum","NOBActual":"sum","NOBPercent":"mean",
                    "ABVTarget":"mean","ABVActual":"mean","ABVPercent":"mean"}).round(2)
        )
    if "Date" in df.columns and df["Date"].notna().any():
        k["date_range"] = {"start": df["Date"].min(), "end": df["Date"].max(), "days": int(df["Date"].dt.date.nunique())}
    score=w=0.0
    if k.get("overall_sales_percent",0)>0: score += min(k["overall_sales_percent"]/100,1.2)*40; w+=40
    if k.get("overall_nob_percent",0)>0:   score += min(k["overall_nob_percent"]/100,1.2)*35; w+=35
    if k.get("overall_abv_percent",0)>0:   score += min(k["overall_abv_percent"]/100,1.2)*25; w+=25
    k["performance_score"] = (score/w*100) if w>0 else 0.0
    return k


# =========================================
# ENHANCED CHARTS
# =========================================
def _metric_area(df: pd.DataFrame, y_col: str, title: str, *, show_target: bool = True) -> go.Figure:
    if df.empty or "Date" not in df.columns or df["Date"].isna().all():
        return go.Figure()
    
    agg_func = "sum" if y_col != "ABVActual" else "mean"
    daily_actual = df.groupby(["Date","BranchName"]).agg({y_col: agg_func}).reset_index()
    target_col = {"SalesActual":"SalesTarget","NOBActual":"NOBTarget","ABVActual":"ABVTarget"}.get(y_col)
    daily_target: Optional[pd.DataFrame] = None
    
    if show_target and target_col and target_col in df.columns:
        agg_func_target = "sum" if y_col != "ABVActual" else "mean"
        daily_target = df.groupby(["Date","BranchName"]).agg({target_col: agg_func_target}).reset_index()
    
    # Enhanced color palette with gradients
    colors = ["#6366f1", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444", "#14b8a6"]
    
    fig = go.Figure()
    
    for i, br in enumerate(sorted(daily_actual["BranchName"].unique())):
        d_actual = daily_actual[daily_actual["BranchName"]==br]
        color = colors[i % len(colors)]
        
        # Add area fill with gradient effect
        fig.add_trace(go.Scatter(
            x=d_actual["Date"], 
            y=d_actual[y_col], 
            name=f"{br}",
            mode="lines+markers", 
            line=dict(width=3, color=color, shape='spline', smoothing=0.3), 
            fill="tonexty" if i > 0 else "tozeroy",
            fillcolor=f"rgba{tuple(list(bytes.fromhex(color[1:])) + [0.2])}",
            marker=dict(size=8, color=color, line=dict(width=2, color='white')),
            hovertemplate=f"<b>{br}</b><br>📅 %{{x|%Y-%m-%d}}<br>📊 %{{y:,.0f}}<extra></extra>",
        ))
        
        # Add target line if available
        if daily_target is not None:
            d_target = daily_target[daily_target["BranchName"]==br]
            fig.add_trace(go.Scatter(
                x=d_target["Date"], 
                y=d_target[target_col], 
                name=f"{br} Target",
                mode="lines", 
                line=dict(width=2, color=color, dash="dash", opacity=0.7),
                hovertemplate=f"<b>{br} Target</b><br>📅 %{{x|%Y-%m-%d}}<br>🎯 %{{y:,.0f}}<extra></extra>",
                showlegend=False,
            ))
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, weight=700, color="#1f2937"),
            x=0.5,
            xanchor='center'
        ),
        height=450, 
        showlegend=True,
        plot_bgcolor="rgba(0,0,0,0)", 
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h", 
            y=1.08, 
            x=0.5, 
            xanchor="center",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
            font=dict(size=12, color="#374151")
        ),
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.05)",
            gridwidth=1,
            tickfont=dict(color="#6b7280"),
            titlefont=dict(color="#374151", size=14, weight=600)
        ),
        yaxis=dict(
            title="Value",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.05)",
            gridwidth=1,
            tickfont=dict(color="#6b7280"),
            titlefont=dict(color="#374151", size=14, weight=600)
        ),
        autosize=True, 
        margin=dict(l=60,r=40,t=80,b=60),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="rgba(0,0,0,0.1)",
            font=dict(color="#374151")
        )
    )
    return fig


def _branch_comparison_chart(bp: pd.DataFrame) -> go.Figure:
    if bp.empty: 
        return go.Figure()
    
    metrics = {"SalesPercent":"Sales %","NOBPercent":"NOB %","ABVPercent":"ABV %"}
    fig = go.Figure()
    x = bp.index.tolist()
    
    # Enhanced gradient colors
    colors = [
        {"color": "#6366f1", "gradient": "rgba(99, 102, 241, 0.8)"},
        {"color": "#10b981", "gradient": "rgba(16, 185, 129, 0.8)"},
        {"color": "#f59e0b", "gradient": "rgba(245, 158, 11, 0.8)"}
    ]
    
    for i, (col, label) in enumerate(metrics.items()):
        fig.add_trace(go.Bar(
            x=x, 
            y=bp[col].tolist(), 
            name=label,
            marker=dict(
                color=colors[i]["gradient"],
                line=dict(color=colors[i]["color"], width=2),
                opacity=0.9
            ),
            hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:.1f}}%<extra></extra>",
            text=[f"{val:.1f}%" for val in bp[col].tolist()],
            textposition="auto",
            textfont=dict(color="white", weight=700)
        ))
    
    fig.update_layout(
        barmode="group", 
        title=dict(
            text="Branch Performance Comparison",
            font=dict(size=20, weight=700, color="#1f2937"),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title="Branch",
            tickfont=dict(color="#6b7280", size=12),
            titlefont=dict(color="#374151", size=14, weight=600)
        ),
        yaxis=dict(
            title="Achievement (%)",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.05)",
            tickfont=dict(color="#6b7280"),
            titlefont=dict(color="#374151", size=14, weight=600)
        ),
        height=420,
        plot_bgcolor="rgba(0,0,0,0)", 
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h", 
            y=1.08, 
            x=0.5, 
            xanchor="center",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1
        ),
        autosize=True, 
        margin=dict(l=60,r=40,t=80,b=60),
        hovermode='x unified'
    )
    return fig


def _liquidity_total_trend_fig(daily: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    
    # Create smooth area chart with gradient
    fig.add_trace(go.Scatter(
        x=daily["Date"], 
        y=daily["TotalLiquidity"], 
        mode="lines+markers",
        line=dict(width=4, color="#6366f1", shape='spline', smoothing=0.3), 
        marker=dict(size=8, color="#6366f1", line=dict(width=2, color='white')),
        fill="tozeroy",
        fillcolor="rgba(99, 102, 241, 0.1)",
        hovertemplate="📅 %{x|%b %d, %Y}<br>💰 SAR %{y:,.0f}<extra></extra>",
        name="Total Liquidity",
    ))
    
    fig.update_layout(
        title=dict(
            text=title or "Liquidity Trend",
            font=dict(size=20, weight=700, color="#1f2937"),
            x=0.5,
            xanchor='center'
        ),
        height=450, 
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)", 
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.05)",
            tickfont=dict(color="#6b7280"),
            title_font=dict(color="#374151", size=14)
        ),
        yaxis=dict(
            title="Liquidity (SAR)",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.05)",
            tickfont=dict(color="#6b7280"),
            title_font=dict(color="#374151", size=14)
        ),
        autosize=True, 
        margin=dict(l=60,r=40,t=80,b=60),
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="rgba(0,0,0,0.1)",
            font=dict(color="#374151")
        )
    )
    return fig


def _fmt(n: Optional[float], decimals: int = 0) -> str:
    if n is None or (isinstance(n, float) and np.isnan(n)): 
        return "—"
    fmt = f"{{:,.{decimals}f}}" if decimals > 0 else "{:,.0f}"
    return fmt.format(n)


# =============== LIQUIDITY ANALYTICS (unchanged) ===============
def _daily_series(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["Date","TotalLiquidity"]).copy()
    d = d.groupby("Date", as_index=False)["TotalLiquidity"].sum().sort_values("Date")
    return d

def _max_drawdown(values: pd.Series) -> float:
    if values.empty: return float("nan")
    run_max = values.cummax()
    dd = values - run_max
    return float(dd.min())

def _project_eom(current_date: pd.Timestamp, last_value: float, avg_daily_delta: float) -> float:
    if pd.isna(avg_daily_delta) or pd.isna(last_value): return float("nan")
    end_of_month = (current_date + pd.offsets.MonthEnd(0)).normalize()
    remaining_days = max(0, (end_of_month.date() - current_date.date()).days)
    return float(last_value + remaining_days * avg_daily_delta)

def _liquidity_report(daily: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if daily.empty:
        return out
    last_date = daily["Date"].max()
    mtd_start = pd.Timestamp(last_date.year, last_date.month, 1)
    mtd = daily[(daily["Date"] >= mtd_start) & (daily["Date"] <= last_date)].copy()
    if mtd.empty:
        return out

    mtd["Delta"] = mtd["TotalLiquidity"].diff()

    out["opening"] = float(mtd["TotalLiquidity"].iloc[0])
    out["current"] = float(mtd["TotalLiquidity"].iloc[-1])
    out["change_since_open"] = out["current"] - out["opening"]
    out["change_since_open_pct"] = (out["change_since_open"] / out["opening"] * 100.0) if out["opening"] else float("nan")
    out["avg_daily_delta"] = float(mtd["Delta"].dropna().mean()) if mtd["Delta"].notna().any() else float("nan")
    out["volatility"] = float(mtd["Delta"].dropna().std(ddof=0)) if mtd["Delta"].notna().any() else float("nan")
    out["proj_eom"] = _project_eom(last_date, out["current"], out["avg_daily_delta"])

    if mtd["Delta"].notna().any():
        best_idx = mtd["Delta"].idxmax()
        worst_idx = mtd["Delta"].idxmin()
        out["best_day"] = (mtd.loc[best_idx, "Date"].date(), float(mtd.loc[best_idx, "Delta"]))
        out["worst_day"] = (mtd.loc[worst_idx, "Date"].date(), float(mtd.loc[worst_idx, "Delta"]))
    else:
        out["best_day"] = out["worst_day"] = (None, float("nan"))

    out["max_drawdown"] = _max_drawdown(mtd["TotalLiquidity"])
    out["daily_mtd"] = mtd[["Date","TotalLiquidity","Delta"]].copy()
    return out


# =========================================
# ENHANCED RENDER HELPERS
# =========================================
def branch_status_class(pct: float) -> str:
    if pct >= 95: return "excellent"
    if pct >= 85: return "good"
    if pct >= 75: return "warn"
    return "danger"


def _branch_color_by_name(name: str) -> str:
    for _, meta in config.BRANCHES.items():
        if meta.get("name") == name: 
            return meta.get("color", "#6b7280")
    return "#6b7280"


def _branch_gradient_by_name(name: str) -> str:
    for _, meta in config.BRANCHES.items():
        if meta.get("name") == name: 
            return meta.get("gradient", "linear-gradient(135deg, #6b7280, #4b5563)")
    return "linear-gradient(135deg, #6b7280, #4b5563)"


def render_enhanced_branch_cards(bp: pd.DataFrame):
    st.markdown("### 🏪 Branch Performance Overview")
    if bp.empty:
        st.info("No branch summary available.")
        return

    items = list(bp.index)
    per_row = max(1, int(getattr(config, "CARDS_PER_ROW", 3)))

    for i in range(0, len(items), per_row):
        row = st.columns(per_row, gap="large")
        for j in range(per_row):
            col_idx = i + j
            with row[j]:
                if col_idx >= len(items):
                    st.empty()
                    continue
                    
                br = items[col_idx]
                r = bp.loc[br]
                avg_pct = float(np.nanmean([r.get("SalesPercent",0), r.get("NOBPercent",0), r.get("ABVPercent",0)]) or 0)
                cls = branch_status_class(avg_pct)
                gradient = _branch_gradient_by_name(br)
                
                # Performance indicators
                sales_indicator = "🟢" if r.get('SalesPercent', 0) >= 95 else "🟡" if r.get('SalesPercent', 0) >= 85 else "🔴"
                nob_indicator = "🟢" if r.get('NOBPercent', 0) >= 90 else "🟡" if r.get('NOBPercent', 0) >= 80 else "🔴"
                abv_indicator = "🟢" if r.get('ABVPercent', 0) >= 90 else "🟡" if r.get('ABVPercent', 0) >= 80 else "🔴"

                st.markdown(
                    f"""
                    <div class="card branch-card" style="background: {gradient}; background-size: 100% 30%, 100% 100%; background-repeat: no-repeat;">
                      <div class="branch-header">
                        <div class="branch-name">{br}</div>
                        <span class="pill {cls}">{avg_pct:.1f}%</span>
                      </div>
                      
                      <div style="margin: 8px 0; font-size: 0.9rem; color: #64748b; font-weight: 500;">
                        Overall Performance Score
                      </div>

                      <div class="branch-metrics">
                        <div>
                          <div class="label">{sales_indicator} Sales</div>
                          <div class="value">{r.get('SalesPercent',0):.1f}%</div>
                        </div>
                        <div>
                          <div class="label">{nob_indicator} NOB</div>
                          <div class="value">{r.get('NOBPercent',0):.1f}%</div>
                        </div>
                        <div>
                          <div class="label">{abv_indicator} ABV</div>
                          <div class="value">{r.get('ABVPercent',0):.1f}%</div>
                        </div>
                      </div>
                      
                      <div style="margin-top: 16px; padding-top: 16px; border-top: 1px solid rgba(0,0,0,0.1);">
                        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #6b7280;">
                          <span>Target: SAR {r.get('SalesTarget',0):,.0f}</span>
                          <span>Actual: SAR {r.get('SalesActual',0):,.0f}</span>
                        </div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_enhanced_overview(df: pd.DataFrame, k: Dict[str, Any]):
    st.markdown("### 🏆 Overall Performance Dashboard")

    # Enhanced metrics row with icons and better formatting
    st.markdown('<div class="metrics-row">', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5, gap="medium")

    with c1:
        st.metric(
            "💰 Total Sales", 
            f"SAR {k.get('total_sales_actual',0):,.0f}",
            delta=f"Target Achievement: {k.get('overall_sales_percent',0):.1f}%",
            delta_color="normal" if k.get('overall_sales_percent',0) >= 95 else "inverse"
        )

    with c2:
        variance_color = "normal" if k.get("total_sales_variance",0) >= 0 else "inverse"
        st.metric(
            "📊 Sales Variance", 
            f"SAR {k.get('total_sales_variance',0):,.0f}",
            delta=f"{k.get('overall_sales_percent',0)-100:+.1f}% vs Target", 
            delta_color=variance_color
        )

    with c3:
        nob_color = "normal" if k.get("overall_nob_percent",0) >= config.TARGETS['nob_achievement'] else "inverse"
        st.metric(
            "🛍️ Total Baskets", 
            f"{k.get('total_nob_actual',0):,.0f}",
            delta=f"Achievement: {k.get('overall_nob_percent',0):.1f}%", 
            delta_color=nob_color
        )

    with c4:
        abv_color = "normal" if k.get("overall_abv_percent",0) >= config.TARGETS['abv_achievement'] else "inverse"
        st.metric(
            "💎 Avg Basket Value", 
            f"SAR {k.get('avg_abv_actual',0):,.2f}",
            delta=f"vs Target: {k.get('overall_abv_percent',0):.1f}%", 
            delta_color=abv_color
        )

    with c5:
        score_color = "normal" if k.get("performance_score",0) >= 80 else "off"
        score_emoji = "🌟" if k.get("performance_score",0) >= 90 else "⭐" if k.get("performance_score",0) >= 80 else "📊"
        st.metric(
            f"{score_emoji} Performance Score", 
            f"{k.get('performance_score',0):.0f}/100",
            delta="Weighted Average", 
            delta_color=score_color
        )
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Enhanced branch cards
    if "branch_performance" in k and not k["branch_performance"].empty:
        render_enhanced_branch_cards(k["branch_performance"])

    # Enhanced comparison table
    if "branch_performance" in k and not k["branch_performance"].empty:
        st.markdown("### 📊 Detailed Performance Analysis")
        
        bp = k["branch_performance"].copy()
        df_table = (
            bp[["SalesPercent","NOBPercent","ABVPercent","SalesActual","SalesTarget"]]
            .rename(columns={
                "SalesPercent":"Sales Achievement %",
                "NOBPercent":"NOB Achievement %",
                "ABVPercent":"ABV Achievement %",
                "SalesActual":"Sales (Actual)",
                "SalesTarget":"Sales (Target)"
            }).round(1)
        )
        
        # Style the dataframe with conditional formatting
        def style_percentage(val):
            if val >= 95:
                return 'background-color: #dcfce7; color: #166534; font-weight: bold'
            elif val >= 85:
                return 'background-color: #dbeafe; color: #1e40af; font-weight: bold'
            elif val >= 75:
                return 'background-color: #fef3c7; color: #92400e; font-weight: bold'
            else:
                return 'background-color: #fee2e2; color: #991b1b; font-weight: bold'
        
        styler = (
            df_table.style
            .format({
                "Sales Achievement %": "{:.1f}%",
                "NOB Achievement %": "{:.1f}%",
                "ABV Achievement %": "{:.1f}%",
                "Sales (Actual)": "SAR {:,.0f}",
                "Sales (Target)": "SAR {:,.0f}"
            })
            .applymap(style_percentage, subset=["Sales Achievement %", "NOB Achievement %", "ABV Achievement %"])
            .set_table_styles([
                {"selector": "th", "props": [("text-align", "center"), ("font-weight", "bold")]},
                {"selector": "td", "props": [("text-align", "center")]}
            ])
        )
        
        st.dataframe(styler, use_container_width=True)

        st.markdown("### 📈 Branch Performance Comparison")
        st.plotly_chart(_branch_comparison_chart(bp), use_container_width=True, config={"displayModeBar": False})


def render_enhanced_liquidity_tab(df: pd.DataFrame):
    if df.empty or "Date" not in df or "TotalLiquidity" not in df:
        st.info("💧 Liquidity columns not found. Include 'TOTAL LIQUIDITY' in your sheet.")
        return

    daily = _daily_series(df)
    if daily.empty:
        st.info("💧 No liquidity data available in the selected range.")
        return

    rep = _liquidity_report(daily)
    if not rep:
        st.info("💧 No MTD liquidity data available.")
        return

    # Enhanced header with gradient background
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1)); 
                    padding: 20px; border-radius: 16px; margin-bottom: 20px; text-align: center;">
            <h3 style="margin: 0; color: #374151;">💧 Liquidity Analytics Dashboard</h3>
            <p style="margin: 8px 0 0 0; color: #6b7280; font-size: 0.9rem;">Month-to-Date Performance & Projections</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Chart with last 30 days
    chart_last = daily.tail(30).copy()
    st.plotly_chart(
        _liquidity_total_trend_fig(chart_last, "Liquidity Trend (Last 30 Days)"), 
        use_container_width=True, 
        config={"displayModeBar": False}
    )

    # Enhanced KPI Row 1
    st.markdown("#### 💰 Key Liquidity Metrics")
    c1, c2, c3, c4 = st.columns(4, gap="large")
    
    with c1:
        st.metric(
            "🏁 Opening (MTD)", 
            f"SAR {_fmt(rep['opening'])}"
        )
    
    with c2:
        delta_badge = f"SAR {_fmt(rep['change_since_open'], 0)} ({rep['change_since_open_pct']:.1f}%)" if not np.isnan(rep["change_since_open_pct"]) else "—"
        delta_color = "normal" if rep.get('change_since_open', 0) >= 0 else "inverse"
        st.metric(
            "📊 Current", 
            f"SAR {_fmt(rep['current'])}", 
            delta=delta_badge,
            delta_color=delta_color
        )
    
    with c3:
        avg_delta_color = "normal" if rep.get('avg_daily_delta', 0) >= 0 else "inverse"
        st.metric(
            "📈 Avg Daily Δ", 
            f"SAR {_fmt(rep['avg_daily_delta'])}",
            delta_color=avg_delta_color
        )
    
    with c4:
        st.metric(
            "🎯 Projected EOM", 
            f"SAR {_fmt(rep['proj_eom'])}"
        )

    # Enhanced Daily Dynamics
    st.markdown("#### 📊 Daily Performance Analysis")
    c5, c6, c7, c8 = st.columns(4, gap="large")
    
    best_day, best_val = rep["best_day"]
    worst_day, worst_val = rep["worst_day"]
    
    with c5:
        st.metric(
            "🏆 Best Day", 
            f"{best_day if best_day else '—'}", 
            delta=f"SAR {_fmt(best_val)}" if not np.isnan(best_val) else "—",
            delta_color="normal"
        )
    
    with c6:
        st.metric(
            "📉 Worst Day", 
            f"{worst_day if worst_day else '—'}", 
            delta=f"SAR {_fmt(worst_val)}" if not np.isnan(worst_val) else "—",
            delta_color="inverse"
        )
    
    with c7:
        st.metric(
            "📊 Volatility (σ)", 
            f"SAR {_fmt(rep['volatility'])}"
        )
    
    with c8:
        st.metric(
            "⬇️ Max Drawdown", 
            f"SAR {_fmt(rep['max_drawdown'])}",
            delta_color="inverse" if rep.get('max_drawdown', 0) < 0 else "off"
        )

    # Enhanced MTD table
    with st.expander("📋 Show MTD Daily Breakdown", expanded=False):
        mtd_df = rep["daily_mtd"].copy()
        mtd_df["Date"] = mtd_df["Date"].dt.date
        mtd_df["Status"] = mtd_df["Delta"].apply(
            lambda x: "🟢 Positive" if x > 0 else "🔴 Negative" if x < 0 else "⚪ Neutral"
        )
        
        st.dataframe(
            mtd_df.rename(columns={
                "TotalLiquidity": "💰 Liquidity (SAR)", 
                "Delta": "📈 Daily Change (SAR)",
                "Status": "📊 Status"
            }),
            use_container_width=True
        )


# =========================================
# ENHANCED MAIN FUNCTION
# =========================================
def main():
    apply_css()

    # Loading state
    with st.spinner('🔄 Loading Al Khair dashboard...'):
        sheets_map = load_workbook_from_gsheet(config.DEFAULT_PUBLISHED_URL)
    
    if not sheets_map: 
        st.error("❌ No data found. Please check your Google Sheets connection.")
        st.stop()
    
    df_all = process_branch_data(sheets_map)
    if df_all.empty: 
        st.error("❌ Could not process data. Check column names and sheet structure.")
        st.stop()

    all_branches = sorted(df_all["BranchName"].dropna().unique()) if "BranchName" in df_all else []
    if "selected_branches" not in st.session_state:
        st.session_state.selected_branches = list(all_branches)

    # Enhanced Sidebar
    with st.sidebar:
        st.markdown(
            '''
            <div class="sb-title">
                🏪 <span>AL KHAIR ANALYTICS</span>
            </div>
            <div class="sb-subtle">Real-time business intelligence</div>
            ''', 
            unsafe_allow_html=True
        )

        st.markdown('<div class="sb-section">🔧 Controls</div>', unsafe_allow_html=True)
        
        if st.button("🔄 Refresh Data", use_container_width=True, type="primary"):
            load_workbook_from_gsheet.clear()
            st.rerun()
        
        # Enhanced branch selection with select all/none
        st.markdown('<div class="sb-section">🏪 Branch Selection</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ All", use_container_width=True):
                st.session_state.selected_branches = list(all_branches)
                st.rerun()
        with col2:
            if st.button("❌ None", use_container_width=True):
                st.session_state.selected_branches = []
                st.rerun()
        
        sel: List[str] = []
        for i, b in enumerate(all_branches):
            is_selected = b in st.session_state.selected_branches
            if st.checkbox(b, value=is_selected, key=f"sb_br_{i}"):
                sel.append(b)
        st.session_state.selected_branches = sel

        # Date selection with better bounds handling
        df_for_bounds = df_all[df_all["BranchName"].isin(st.session_state.selected_branches)].copy() if st.session_state.selected_branches else df_all.copy()
        if "Date" in df_for_bounds.columns and df_for_bounds["Date"].notna().any():
            dmin_sb, dmax_sb = df_for_bounds["Date"].min().date(), df_for_bounds["Date"].max().date()
        else:
            dmin_sb = dmax_sb = datetime.today().date()

        sd_val = st.session_state.get("start_date", dmin_sb)
        ed_val = st.session_state.get("end_date", dmax_sb)
        sd_val = min(max(sd_val, dmin_sb), dmax_sb)
        ed_val = min(max(ed_val, dmin_sb), dmax_sb)
        if sd_val > ed_val:
            sd_val, ed_val = dmin_sb, dmax_sb

        st.markdown('<div class="sb-section">📅 Date Range</div>', unsafe_allow_html=True)
        
        # Quick date presets
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📅 This Month", use_container_width=True):
                today = datetime.today().date()
                st.session_state.start_date = today.replace(day=1)
                st.session_state.end_date = today
                st.rerun()
        with col2:
            if st.button("📅 Last 30d", use_container_width=True):
                today = datetime.today().date()
                st.session_state.start_date = today - timedelta(days=30)
                st.session_state.end_date = today
                st.rerun()
        
        _sd = st.date_input("Start Date:", value=sd_val, min_value=dmin_sb, max_value=dmax_sb, key="sb_sd")
        _ed = st.date_input("End Date:", value=ed_val, min_value=dmin_sb, max_value=dmax_sb, key="sb_ed")

        st.session_state.start_date = min(max(_sd, dmin_sb), dmax_sb)
        st.session_state.end_date = min(max(_ed, dmin_sb), dmax_sb)
        if st.session_state.start_date > st.session_state.end_date:
            st.session_state.start_date, st.session_state.end_date = dmin_sb, dmax_sb

        # Data summary in sidebar
        st.markdown('<div class="sb-hr"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-section">📊 Data Summary</div>', unsafe_allow_html=True)
        
        filtered_data = df_all.copy()
        if st.session_state.selected_branches:
            filtered_data = filtered_data[filtered_data["BranchName"].isin(st.session_state.selected_branches)]
        
        st.markdown(f"""
        <div style="font-size: 0.8rem; color: #6b7280; line-height: 1.4;">
        📍 Branches: {len(st.session_state.selected_branches)}<br>
        📅 Date Range: {(st.session_state.end_date - st.session_state.start_date).days + 1} days<br>
        📊 Total Records: {len(filtered_data):,}
        </div>
        """, unsafe_allow_html=True)

    # Enhanced Hero Header
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-title">
            <span class="hero-emoji">🏪</span>
            {config.PAGE_TITLE}
          </div>
          <div class="subtitle">Advanced Business Intelligence & Performance Analytics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Apply filters
    df = df_all.copy()
    if "BranchName" in df.columns and st.session_state.selected_branches:
        df = df[df["BranchName"].isin(st.session_state.selected_branches)].copy()
    if "Date" in df.columns and df["Date"].notna().any():
        mask = (df["Date"].dt.date >= st.session_state.start_date) & (df["Date"].dt.date <= st.session_state.end_date)
        df = df.loc[mask].copy()
        if df.empty: 
            st.warning("⚠️ No data found in selected date range. Please adjust your filters.")
            st.stop()

    # Calculate KPIs
    k = calc_kpis(df)

    # Enhanced Quick Insights with better styling
    with st.expander("⚡ Smart Insights & Alerts", expanded=True):
        insights: List[str] = []
        alerts: List[str] = []

        if "branch_performance" in k and isinstance(k["branch_performance"], pd.DataFrame) and not k["branch_performance"].empty:
            bp = k["branch_performance"]
            try:
                best_branch = bp['SalesPercent'].idxmax()
                worst_branch = bp['SalesPercent'].idxmin()
                insights.append(f"🥇 **Top Performer**: {best_branch} with {bp['SalesPercent'].max():.1f}% sales achievement")
                insights.append(f"📈 **Needs Attention**: {worst_branch} with {bp['SalesPercent'].min():.1f}% sales achievement")
            except Exception:
                pass
            
            below_target = bp[bp["SalesPercent"] < config.TARGETS["sales_achievement"]].index.tolist()
            if below_target: 
                alerts.append(f"🚨 **Below Target**: {', '.join(below_target)} need immediate attention")

        if k.get("total_sales_variance", 0) < 0:
            alerts.append(f"📉 **Revenue Gap**: SAR {abs(k['total_sales_variance']):,.0f} below target")
        elif k.get("total_sales_variance", 0) > 0:
            insights.append(f"💰 **Revenue Surplus**: SAR {k['total_sales_variance']:,.0f} above target")

        # Liquidity insights
        if {"TotalLiquidity"}.issubset(df.columns):
            dliq = df.dropna(subset=["Date","TotalLiquidity"]).copy()
            if not dliq.empty:
                by_date = dliq.groupby("Date", as_index=False)["TotalLiquidity"].sum().sort_values("Date")
                if len(by_date) >= 2:
                    last_val = float(by_date["TotalLiquidity"].iloc[-1])
                    prev_val = float(by_date["TotalLiquidity"].iloc[-2])
                    change = ((last_val - prev_val) / prev_val * 100) if prev_val else 0
                    
                    if change > 5:
                        insights.append(f"💧 **Liquidity Boost**: {change:+.1f}% increase (SAR {last_val:,.0f})")
                    elif change < -5:
                        alerts.append(f"💧 **Liquidity Drop**: {change:.1f}% decrease (SAR {last_val:,.0f})")

        # Performance score insights
        score = k.get("performance_score", 0)
        if score >= 90:
            insights.append(f"🌟 **Excellent Performance**: {score:.0f}/100 overall score")
        elif score < 70:
            alerts.append(f"⚠️ **Performance Alert**: {score:.0f}/100 overall score needs improvement")

        # Display insights with better formatting
        if insights or alerts:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if insights:
                    st.markdown("**✅ Positive Insights**")
                    for insight in insights:
                        st.markdown(f"• {insight}")
            
            with col2:
                if alerts:
                    st.markdown("**🚨 Action Required**")
                    for alert in alerts:
                        st.markdown(f"• {alert}")
        else:
            st.markdown("📊 All metrics are within normal ranges for the selected period.")

    # Enhanced Tabs with better icons
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 Executive Dashboard", 
        "📈 Trends & Analytics", 
        "💧 Liquidity Management", 
        "📊 Data Export"
    ])

    with tab1:
        render_enhanced_overview(df, k)

    with tab2:
        st.markdown("### 📈 Performance Trends Analysis")
        
        # Time period selector with better UI
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("#### 📊 Select Metric to Analyze")
        with col2:
            time_options = ["Last 7 Days", "Last 30 Days", "Last 3 Months", "All Available Data"]
            selected_period = st.selectbox("📅 Time Period", time_options, index=1, key="trend_window")
        
        # Filter data based on selected period
        if "Date" in df.columns and df["Date"].notna().any():
            today = (df["Date"].max() if df["Date"].notna().any() else pd.Timestamp.today()).date()
            period_map = {
                "Last 7 Days": today - timedelta(days=7),
                "Last 30 Days": today - timedelta(days=30), 
                "Last 3 Months": today - timedelta(days=90),
                "All Available Data": df["Date"].min().date()
            }
            start_date = period_map[selected_period]
            trend_df = df[(df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= today)].copy()
        else:
            trend_df = df.copy()
        
        # Metric tabs with enhanced charts
        m1, m2, m3 = st.tabs(["💰 Sales Performance", "🛍️ Basket Analytics", "💎 Value Analysis"])
        
        with m1:
            st.plotly_chart(
                _metric_area(trend_df, "SalesActual", f"Sales Performance - {selected_period}"), 
                use_container_width=True, 
                config={"displayModeBar": False}
            )
            
            # Additional sales insights
            if not trend_df.empty and "SalesActual" in trend_df.columns:
                total_sales = trend_df["SalesActual"].sum()
                avg_daily = trend_df.groupby("Date")["SalesActual"].sum().mean()
                st.markdown(f"""
                **📊 Sales Summary for {selected_period}:**
                - Total Sales: SAR {total_sales:,.0f}
                - Average Daily Sales: SAR {avg_daily:,.0f}
                """)
        
        with m2:
            st.plotly_chart(
                _metric_area(trend_df, "NOBActual", f"Number of Baskets - {selected_period}"), 
                use_container_width=True, 
                config={"displayModeBar": False}
            )
            
            if not trend_df.empty and "NOBActual" in trend_df.columns:
                total_baskets = trend_df["NOBActual"].sum()
                avg_daily_baskets = trend_df.groupby("Date")["NOBActual"].sum().mean()
                st.markdown(f"""
                **🛍️ Basket Summary for {selected_period}:**
                - Total Baskets: {total_baskets:,.0f}
                - Average Daily Baskets: {avg_daily_baskets:,.0f}
                """)
        
        with m3:
            st.plotly_chart(
                _metric_area(trend_df, "ABVActual", f"Average Basket Value - {selected_period}"), 
                use_container_width=True, 
                config={"displayModeBar": False}
            )
            
            if not trend_df.empty and "ABVActual" in trend_df.columns:
                avg_abv = trend_df["ABVActual"].mean()
                max_abv = trend_df["ABVActual"].max()
                min_abv = trend_df["ABVActual"].min()
                st.markdown(f"""
                **💎 Basket Value Summary for {selected_period}:**
                - Average ABV: SAR {avg_abv:,.2f}
                - Highest ABV: SAR {max_abv:,.2f}
                - Lowest ABV: SAR {min_abv:,.2f}
                """)

    with tab3:
        render_enhanced_liquidity_tab(df)

    with tab4:
        st.markdown("### 📊 Export & Download Center")
        
        if df.empty:
            st.info("📭 No data available for export with current filters.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 Excel Report")
                st.markdown("Complete analytics report with all sheets and summaries")
                
                # Create enhanced Excel export
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                    # Main data
                    df.to_excel(writer, sheet_name="Raw Data", index=False)
                    
                    # Branch performance summary
                    if "branch_performance" in k and isinstance(k["branch_performance"], pd.DataFrame):
                        k["branch_performance"].to_excel(writer, sheet_name="Branch Summary")
                    
                    # Daily aggregates
                    if "Date" in df.columns:
                        daily_agg = df.groupby("Date").agg({
                            "SalesActual": "sum",
                            "SalesTarget": "sum", 
                            "NOBActual": "sum",
                            "ABVActual": "mean"
                        }).round(2)
                        daily_agg.to_excel(writer, sheet_name="Daily Summary")
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                st.download_button(
                    "📊 Download Excel Report",
                    buf.getvalue(),
                    f"AlKhair_Analytics_Report_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    type="primary"
                )
            
            with col2:
                st.markdown("#### 📄 CSV Data")
                st.markdown("Raw data export for further analysis")
                
                csv_data = df.to_csv(index=False).encode("utf-8")
                timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                st.download_button(
                    "📄 Download CSV",
                    csv_data,
                    f"AlKhair_Data_Export_{timestamp}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # Export summary
            st.markdown("---")
            st.markdown("#### 📋 Export Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📊 Total Records", f"{len(df):,}")
            with col2:
                st.metric("🏪 Branches", f"{df['BranchName'].nunique() if 'BranchName' in df else 0}")
            with col3:
                date_range = (df["Date"].max() - df["Date"].min()).days + 1 if "Date" in df and df["Date"].notna().any() else 0
                st.metric("📅 Date Range", f"{date_range} days")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.exception(e)
