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
import plotly.express as px
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
        "Al khair - 102": {"name": "Al Khair", "code": "102", "color": "#3b82f6", "target_color": "#93c5fd"},
        "Noora - 104": {"name": "Noora", "code": "104", "color": "#10b981", "target_color": "#86efac"},
        "Hamra - 109": {"name": "Hamra", "code": "109", "color": "#f59e0b", "target_color": "#fbbf24"},
        "Magnus - 107": {"name": "Magnus", "code": "107", "color": "#8b5cf6", "target_color": "#c4b5fd"},
    }
    TARGETS = {"sales_achievement": 95.0, "nob_achievement": 90.0, "abv_achievement": 90.0}
    SHOW_SUBTITLE = False
    CARDS_PER_ROW = 2  # More compact branch cards


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
        }

        /* Main container with glassmorphism */
        .main .block-container { 
            padding: 1rem 1.5rem; 
            max-width: 1900px;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
            border: 1px solid rgba(255, 255, 255, 0.18);
            margin: 1rem auto;
        }

        /* ---- HERO HEADER WITH ANIMATION ---- */
        .hero { 
            text-align: center; 
            margin: 0 0 2rem 0; 
            position: relative;
        }
        
        .hero-bar { 
            width: min(1200px, 95%); 
            height: 6px; 
            margin: 8px auto 25px;
            border-radius: 8px; 
            background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1, #f9ca24, #f0932b);
            background-size: 300% 100%;
            animation: gradientShift 4s ease infinite;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .hero-title { 
            font-weight: 800; 
            font-size: clamp(28px, 4vw, 48px); 
            letter-spacing: -0.5px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: inline-flex; 
            align-items: center; 
            gap: 0.75rem;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .hero-emoji { 
            font-size: 1.2em; 
            filter: drop-shadow(2px 2px 4px rgba(0, 0, 0, 0.3));
        }

        /* Enhanced metric tiles with hover effects */
        [data-testid="metric-container"] {
            background: rgba(255, 255, 255, 0.95) !important;
            border-radius: 20px !important;
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
            padding: 20px !important;
            text-align: center !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            backdrop-filter: blur(10px) !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1) !important;
        }
        
        [data-testid="metric-container"]:hover {
            transform: translateY(-5px) !important;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2) !important;
            border-color: rgba(103, 126, 234, 0.5) !important;
        }
        
        [data-testid="stMetricValue"] { 
            font-size: clamp(18px, 2.5vw, 26px) !important; 
            line-height: 1.2 !important;
            font-weight: 700 !important;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        [data-testid="stMetricLabel"] { 
            font-size: 0.9rem !important;
            color: #6b7280 !important;
            font-weight: 500 !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Enhanced branch cards */
        .card { 
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 20px;
            padding: 24px;
            height: 100%;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            backdrop-filter: blur(15px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        }
        
        .card:hover {
            transform: translateY(-8px) scale(1.02);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
        }
        
        .branch-card { 
            display: flex; 
            flex-direction: column; 
            height: 100%; 
            position: relative;
        }
        
        .branch-metrics {
            display: grid; 
            grid-template-columns: repeat(3, 1fr);
            gap: 12px 16px; 
            margin-top: 20px; 
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
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Enhanced status pills */
        .pill { 
            display: inline-block; 
            padding: 6px 14px; 
            border-radius: 50px; 
            font-weight: 700; 
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }
        
        .pill:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }
        
        .pill.excellent { background: linear-gradient(135deg, #10b981, #059669); color: white; }
        .pill.good { background: linear-gradient(135deg, #3b82f6, #1d4ed8); color: white; }
        .pill.warn { background: linear-gradient(135deg, #f59e0b, #d97706); color: white; }
        .pill.danger { background: linear-gradient(135deg, #ef4444, #dc2626); color: white; }

        /* Modern tab styling */
        .stTabs [data-baseweb="tab-list"] { 
            gap: 4px; 
            border-bottom: none;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 4px;
            backdrop-filter: blur(10px);
        }
        
        .stTabs [data-baseweb="tab"] {
            position: relative; 
            border-radius: 12px !important; 
            padding: 12px 24px !important;
            font-weight: 600 !important; 
            background: transparent !important; 
            color: rgba(255, 255, 255, 0.7) !important;
            border: none !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        
        .stTabs [data-baseweb="tab"]:hover { 
            background: rgba(255, 255, 255, 0.1) !important;
            color: white !important;
        }
        
        .stTabs [aria-selected="true"] {
            background: rgba(255, 255, 255, 0.9) !important; 
            color: #1f2937 !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1) !important;
            transform: translateY(-2px);
        }

        /* Enhanced sidebar */
        [data-testid="stSidebar"] {
            background: rgba(247, 249, 252, 0.95);
            backdrop-filter: blur(20px);
            border-right: 1px solid rgba(255, 255, 255, 0.3);
            width: 280px;
            min-width: 260px;
            max-width: 300px;
        }

        /* Performance indicators */
        .performance-indicator {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            border-radius: 25px;
            font-weight: 600;
            font-size: 0.85rem;
            margin: 4px 0;
        }
        
        .indicator-excellent {
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
        }
        
        .indicator-good {
            background: linear-gradient(135deg, #3b82f6, #1d4ed8);
            color: white;
        }
        
        .indicator-warning {
            background: linear-gradient(135deg, #f59e0b, #d97706);
            color: white;
        }
        
        .indicator-danger {
            background: linear-gradient(135deg, #ef4444, #dc2626);
            color: white;
        }

        /* Responsive design improvements */
        @media (max-width: 768px) {
            .main .block-container {
                padding: 1rem;
                margin: 0.5rem;
            }
            
            .hero-title {
                font-size: 24px;
            }
            
            .branch-metrics {
                grid-template-columns: 1fr;
                gap: 8px;
            }
        }

        /* Loading animations */
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        .loading {
            animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #667eea, #764ba2);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #5a67d8, #6b46c1);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================
# ENHANCED DATA PROCESSING
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
            
        for col in ["SalesTarget","SalesActual","SalesPercent","NOBTarget","NOBActual","NOBPercent",
                    "ABVTarget","ABVActual","ABVPercent","Bank","Cash","TotalLiquidity","ChangeLiquidity","PctChange"]:
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
        k["date_range"] = {
            "start": df["Date"].min(), 
            "end": df["Date"].max(), 
            "days": int(df["Date"].dt.date.nunique())
        }
    
    # Enhanced performance scoring
    score = w = 0.0
    if k.get("overall_sales_percent", 0) > 0: 
        score += min(k["overall_sales_percent"]/100, 1.2) * 40
        w += 40
    if k.get("overall_nob_percent", 0) > 0:   
        score += min(k["overall_nob_percent"]/100, 1.2) * 35
        w += 35
    if k.get("overall_abv_percent", 0) > 0:   
        score += min(k["overall_abv_percent"]/100, 1.2) * 25
        w += 25
    
    k["performance_score"] = (score/w*100) if w > 0 else 0.0
    return k


# =========================================
# ENHANCED CHARTS WITH MODERN STYLING
# =========================================
def create_modern_chart_theme():
    """Create a modern chart theme with consistent styling"""
    return {
        'layout': {
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'font': {'family': 'Inter', 'size': 12, 'color': '#374151'},
            'margin': {'l': 60, 'r': 40, 't': 60, 'b': 50},
            'showlegend': True,
            'legend': {
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': 1.02,
                'xanchor': 'center',
                'x': 0.5,
                'bgcolor': 'rgba(255,255,255,0.8)',
                'bordercolor': 'rgba(0,0,0,0.1)',
                'borderwidth': 1
            },
            'xaxis': {
                'gridcolor': 'rgba(0,0,0,0.1)',
                'linecolor': 'rgba(0,0,0,0.2)',
                'tickfont': {'size': 10}
            },
            'yaxis': {
                'gridcolor': 'rgba(0,0,0,0.1)',
                'linecolor': 'rgba(0,0,0,0.2)',
                'tickfont': {'size': 10}
            }
        }
    }


def create_enhanced_area_chart(df: pd.DataFrame, y_col: str, title: str, *, show_target: bool = True) -> go.Figure:
    """Enhanced area chart with modern styling and animations"""
    if df.empty or "Date" not in df.columns or df["Date"].isna().all():
        return go.Figure()
    
    agg_func = "sum" if y_col != "ABVActual" else "mean"
    daily_actual = df.groupby(["Date", "BranchName"]).agg({y_col: agg_func}).reset_index()
    
    target_col = {"SalesActual": "SalesTarget", "NOBActual": "NOBTarget", "ABVActual": "ABVTarget"}.get(y_col)
    daily_target: Optional[pd.DataFrame] = None
    
    if show_target and target_col and target_col in df.columns:
        agg_func_target = "sum" if y_col != "ABVActual" else "mean"
        daily_target = df.groupby(["Date", "BranchName"]).agg({target_col: agg_func_target}).reset_index()
    
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    
    branches = sorted(daily_actual["BranchName"].unique())
    colors = [config.BRANCHES.get(f"{br} - {code}", {}).get("color", "#6b7280") 
              for br in branches for code in ["102", "104", "109", "107"] 
              if config.BRANCHES.get(f"{br} - {code}", {}).get("name") == br]
    
    for i, br in enumerate(branches):
        d_actual = daily_actual[daily_actual["BranchName"] == br]
        color = colors[i] if i < len(colors) else f"hsl({i * 60}, 70%, 50%)"
        
        # Add actual values with enhanced styling
        fig.add_trace(go.Scatter(
            x=d_actual["Date"], 
            y=d_actual[y_col], 
            name=f"{br}",
            mode="lines+markers", 
            line=dict(width=3, color=color, shape='spline'),
            marker=dict(size=6, color=color, line=dict(width=1, color='white')),
            fill="tonexty" if i > 0 else "tozeroy",
            fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(color)) + [0.2])}",
            hovertemplate=f"<b>{br}</b><br>Date: %{{x|%Y-%m-%d}}<br>Actual: %{{y:,.0f}}<extra></extra>",
        ))
        
        # Add target line if available
        if daily_target is not None:
            d_target = daily_target[daily_target["BranchName"] == br]
            if not d_target.empty:
                fig.add_trace(go.Scatter(
                    x=d_target["Date"], 
                    y=d_target[target_col], 
                    name=f"{br} Target",
                    mode="lines", 
                    line=dict(width=2, color=color, dash="dot"),
                    hovertemplate=f"<b>{br} Target</b><br>Date: %{{x|%Y-%m-%d}}<br>Target: %{{y:,.0f}}<extra></extra>",
                    showlegend=False,
                ))
    
    # Apply modern theme
    theme = create_modern_chart_theme()
    fig.update_layout(**theme['layout'])
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16, weight="bold")),
        height=450,
        hovermode='x unified'
    )
    
    return fig


def create_performance_comparison_chart(bp: pd.DataFrame) -> go.Figure:
    """Enhanced comparison chart with better visual design"""
    if bp.empty: 
        return go.Figure()
    
    fig = go.Figure()
    
    branches = bp.index.tolist()
    metrics = {
        "SalesPercent": {"name": "Sales Achievement", "color": "#3b82f6"},
        "NOBPercent": {"name": "Baskets Achievement", "color": "#10b981"}, 
        "ABVPercent": {"name": "Basket Value Achievement", "color": "#f59e0b"}
    }
    
    for metric, config_item in metrics.items():
        fig.add_trace(go.Bar(
            name=config_item["name"],
            x=branches,
            y=bp[metric].tolist(),
            marker=dict(
                color=config_item["color"],
                line=dict(color='rgba(255,255,255,0.8)', width=1)
            ),
            hovertemplate=f"<b>%{{x}}</b><br>{config_item['name']}: %{{y:.1f}}%<extra></extra>",
        ))
    
    # Add target reference lines
    target_lines = [95, 90, 90]  # Sales, NOB, ABV targets
    for i, (target, color) in enumerate(zip(target_lines, ["#3b82f6", "#10b981", "#f59e0b"])):
        fig.add_hline(
            y=target,
            line_dash="dash",
            line_color=color,
            opacity=0.5,
            annotation_text=f"Target: {target}%",
            annotation_position="right"
        )
    
    theme = create_modern_chart_theme()
    fig.update_layout(**theme['layout'])
    fig.update_layout(
        title=dict(text="Branch Performance vs Targets", x=0.5, font=dict(size=16, weight="bold")),
        barmode="group",
        height=400,
        yaxis_title="Achievement (%)",
        xaxis_title="Branch"
    )
    
    return fig


def create_liquidity_trend_chart(daily: pd.DataFrame) -> go.Figure:
    """Enhanced liquidity chart with trend analysis"""
    if daily.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    # Main liquidity line
    fig.add_trace(go.Scatter(
        x=daily["Date"], 
        y=daily["TotalLiquidity"], 
        mode="lines+markers",
        name="Total Liquidity",
        line=dict(width=4, color="#3b82f6", shape='spline'),
        marker=dict(size=8, color="#3b82f6", line=dict(width=2, color='white')),
        fill="tozeroy",
        fillcolor="rgba(59, 130, 246, 0.1)",
        hovertemplate="Date: %{x|%b %d, %Y}<br>Liquidity: SAR %{y:,.0f}<extra></extra>",
    ))
    
    # Add trend line if we have enough data
    if len(daily) > 7:
        # Simple linear trend
        from sklearn.linear_model import LinearRegression
        import numpy as np
        
        X = np.array(range(len(daily))).reshape(-1, 1)
        y = daily["TotalLiquidity"].values
        
        model = LinearRegression()
        model.fit(X, y)
        trend_line = model.predict(X)
        
        fig.add_trace(go.Scatter(
            x=daily["Date"],
            y=trend_line,
            mode="lines",
            name="Trend",
            line=dict(width=2, color="#ef4444", dash="dot"),
            hovertemplate="Trend: SAR %{y:,.0f}<extra></extra>",
        ))
    
    theme = create_modern_chart_theme()
    fig.update_layout(**theme['layout'])
    fig.update_layout(
        title=dict(text="Liquidity Trend Analysis", x=0.5, font=dict(size=16, weight="bold")),
        height=450,
        yaxis_title="Liquidity (SAR)",
        xaxis_title="Date"
    )
    
    return fig


def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create correlation heatmap for key metrics"""
    if df.empty:
        return go.Figure()
    
    # Select numeric columns for correlation
    numeric_cols = ["SalesActual", "NOBActual", "ABVActual", "SalesPercent", "NOBPercent", "ABVPercent"]
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if len(available_cols) < 2:
        return go.Figure()
    
    corr_data = df[available_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_data.values,
        x=corr_data.columns,
        y=corr_data.columns,
        colorscale="RdYlBu_r",
        zmid=0,
        colorbar=dict(title="Correlation"),
        hovertemplate="X: %{x}<br>Y: %{y}<br>Correlation: %{z:.2f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(text="Metrics Correlation Matrix", x=0.5, font=dict(size=16, weight="bold")),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


# =========================================
# ENHANCED RENDERING FUNCTIONS
# =========================================
def get_performance_status(pct: float) -> tuple:
    """Get status class and emoji for performance percentage"""
    if pct >= 95:
        return "excellent", "🚀", "Excellent"
    elif pct >= 85:
        return "good", "✅", "Good"
    elif pct >= 75:
        return "warn", "⚠️", "Warning"
    else:
        return "danger", "🔴", "Critical"


def render_enhanced_branch_cards(bp: pd.DataFrame):
    """Render enhanced branch cards with better visual design"""
    st.markdown("### 🏪 Branch Performance Overview")
    if bp.empty:
        st.info("No branch summary available.")
        return

    items = list(bp.index)
    per_row = 2  # Two cards per row for better visibility

    for i in range(0, len(items), per_row):
        cols = st.columns(per_row, gap="large")
        for j in range(per_row):
            col_idx = i + j
            with cols[j]:
                if col_idx >= len(items):
                    st.empty()
                    continue
                    
                br = items[col_idx]
                r = bp.loc[br]
                
                # Calculate weighted average performance
                weights = {"SalesPercent": 0.4, "NOBPercent": 0.35, "ABVPercent": 0.25}
                avg_pct = sum(r.get(metric, 0) * weight for metric, weight in weights.items())
                
                cls, emoji, status_text = get_performance_status(avg_pct)
                
                # Get branch color
                branch_key = next((k for k, v in config.BRANCHES.items() if v.get("name") == br), "")
                color = config.BRANCHES.get(branch_key, {}).get("color", "#6b7280")
                
                st.markdown(
                    f"""
                    <div class="card branch-card" style="background: linear-gradient(135deg, {color}15 0%, rgba(255,255,255,0.95) 50%);">
                      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
                        <div style="font-weight:800;font-size:1.2rem;color:#1f2937;">{emoji} {br}</div>
                        <span class="pill {cls}">{avg_pct:.1f}%</span>
                      </div>
                      
                      <div class="performance-indicator indicator-{cls}" style="margin-bottom:20px;">
                        <span>{status_text} Performance</span>
                      </div>

                      <div class="branch-metrics">
                        <div>
                          <div class="label">Sales Achievement</div>
                          <div class="value">{r.get('SalesPercent',0):.1f}%</div>
                        </div>
                        <div>
                          <div class="label">Baskets Achievement</div>
                          <div class="value">{r.get('NOBPercent',0):.1f}%</div>
                        </div>
                        <div>
                          <div class="label">Basket Value</div>
                          <div class="value">{r.get('ABVPercent',0):.1f}%</div>
                        </div>
                        <div>
                          <div class="label">Total Sales</div>
                          <div class="value">SAR {r.get('SalesActual',0):,.0f}</div>
                        </div>
                        <div>
                          <div class="label">Total Baskets</div>
                          <div class="value">{r.get('NOBActual',0):,.0f}</div>
                        </div>
                        <div>
                          <div class="label">Avg Basket Value</div>
                          <div class="value">SAR {r.get('ABVActual',0):,.0f}</div>
                        </div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_enhanced_overview(df: pd.DataFrame, k: Dict[str, Any]):
    """Render enhanced overview with better KPIs and insights"""
    st.markdown("### 🏆 Performance Dashboard")

    # Enhanced metrics row
    c1, c2, c3, c4, c5 = st.columns(5, gap="medium")

    with c1:
        sales_status = "🚀" if k.get('overall_sales_percent', 0) >= 95 else "📈"
        st.metric(
            f"{sales_status} Total Sales", 
            f"SAR {k.get('total_sales_actual', 0):,.0f}",
            delta=f"{k.get('overall_sales_percent', 0) - 100:+.1f}% vs Target"
        )

    with c2:
        variance = k.get('total_sales_variance', 0)
        variance_emoji = "💚" if variance >= 0 else "📉"
        st.metric(
            f"{variance_emoji} Sales Variance", 
            f"SAR {variance:,.0f}",
            delta=f"{k.get('overall_sales_percent', 0):.1f}% Achievement"
        )

    with c3:
        nob_emoji = "🛍️" if k.get('overall_nob_percent', 0) >= 90 else "🔄"
        st.metric(
            f"{nob_emoji} Total Baskets", 
            f"{k.get('total_nob_actual', 0):,.0f}",
            delta=f"{k.get('overall_nob_percent', 0):.1f}% vs Target"
        )

    with c4:
        abv_emoji = "💎" if k.get('overall_abv_percent', 0) >= 90 else "💰"
        st.metric(
            f"{abv_emoji} Avg Basket Value", 
            f"SAR {k.get('avg_abv_actual', 0):,.2f}",
            delta=f"{k.get('overall_abv_percent', 0):.1f}% vs Target"
        )

    with c5:
        score = k.get('performance_score', 0)
        score_emoji = "⭐" if score >= 80 else "📊"
        st.metric(
            f"{score_emoji} Performance Score", 
            f"{score:.0f}/100",
            delta="Weighted Average"
        )

    # Enhanced branch cards
    if "branch_performance" in k and not k["branch_performance"].empty:
        render_enhanced_branch_cards(k["branch_performance"])

    # Performance comparison chart
    if "branch_performance" in k and not k["branch_performance"].empty:
        st.markdown("### 📊 Performance Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.plotly_chart(
                create_performance_comparison_chart(k["branch_performance"]), 
                use_container_width=True, 
                config={"displayModeBar": False}
            )
        
        with col2:
            st.plotly_chart(
                create_correlation_heatmap(df),
                use_container_width=True,
                config={"displayModeBar": False}
            )


def render_enhanced_trends_tab(df: pd.DataFrame):
    """Enhanced trends tab with multiple visualization options"""
    if df.empty or "Date" not in df.columns:
        st.info("No date data available for trends.")
        return
    
    st.markdown("### 📈 Performance Trends Analysis")
    
    # Time period selector
    col1, col2 = st.columns([3, 1])
    with col1:
        period_options = ["Last 7 Days", "Last 30 Days", "Last 3 Months", "All Time"]
        selected_period = st.selectbox("📅 Select Time Period", period_options, index=1)
    
    with col2:
        chart_type = st.selectbox("📊 Chart Type", ["Area Chart", "Line Chart", "Bar Chart"])
    
    # Filter data based on selected period
    today = df["Date"].max().date() if df["Date"].notna().any() else datetime.today().date()
    period_map = {
        "Last 7 Days": timedelta(days=7),
        "Last 30 Days": timedelta(days=30),
        "Last 3 Months": timedelta(days=90),
        "All Time": None
    }
    
    if period_map[selected_period]:
        start_date = today - period_map[selected_period]
        filtered_df = df[df["Date"].dt.date >= start_date].copy()
    else:
        filtered_df = df.copy()
    
    # Metrics tabs
    metric_tab1, metric_tab2, metric_tab3 = st.tabs(["💰 Sales Performance", "🛍️ Basket Analysis", "💎 Value Analysis"])
    
    with metric_tab1:
        fig = create_enhanced_area_chart(filtered_df, "SalesActual", f"Sales Trends - {selected_period}")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        
        # Sales insights
        if not filtered_df.empty:
            daily_sales = filtered_df.groupby("Date")["SalesActual"].sum()
            best_day = daily_sales.idxmax()
            worst_day = daily_sales.idxmin()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("📈 Best Day", f"{best_day.strftime('%b %d')}", f"SAR {daily_sales.max():,.0f}")
            col2.metric("📉 Lowest Day", f"{worst_day.strftime('%b %d')}", f"SAR {daily_sales.min():,.0f}")
            col3.metric("📊 Daily Average", "Period Avg", f"SAR {daily_sales.mean():,.0f}")
    
    with metric_tab2:
        fig = create_enhanced_area_chart(filtered_df, "NOBActual", f"Basket Count Trends - {selected_period}")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    
    with metric_tab3:
        fig = create_enhanced_area_chart(filtered_df, "ABVActual", f"Basket Value Trends - {selected_period}")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_enhanced_liquidity_tab(df: pd.DataFrame):
    """Enhanced liquidity analysis with more insights"""
    if df.empty or "TotalLiquidity" not in df.columns:
        st.info("💧 Liquidity data not available. Please ensure 'TOTAL LIQUIDITY' column exists.")
        return
    
    st.markdown("### 💧 Liquidity Management Dashboard")
    
    # Process liquidity data
    liquidity_df = df.dropna(subset=["Date", "TotalLiquidity"]).copy()
    if liquidity_df.empty:
        st.warning("No valid liquidity data found.")
        return
    
    daily_liquidity = liquidity_df.groupby("Date", as_index=False)["TotalLiquidity"].sum().sort_values("Date")
    
    # Liquidity metrics
    current_liquidity = daily_liquidity["TotalLiquidity"].iloc[-1]
    prev_liquidity = daily_liquidity["TotalLiquidity"].iloc[-2] if len(daily_liquidity) > 1 else current_liquidity
    liquidity_change = current_liquidity - prev_liquidity
    liquidity_change_pct = (liquidity_change / prev_liquidity * 100) if prev_liquidity != 0 else 0
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric(
        "💰 Current Liquidity", 
        f"SAR {current_liquidity:,.0f}",
        delta=f"{liquidity_change:+,.0f} from yesterday"
    )
    
    col2.metric(
        "📈 Daily Change", 
        f"{liquidity_change_pct:+.1f}%",
        delta=f"SAR {liquidity_change:+,.0f}"
    )
    
    avg_liquidity = daily_liquidity["TotalLiquidity"].mean()
    col3.metric(
        "📊 Period Average", 
        f"SAR {avg_liquidity:,.0f}",
        delta=f"{(current_liquidity/avg_liquidity - 1)*100:+.1f}% vs avg"
    )
    
    max_liquidity = daily_liquidity["TotalLiquidity"].max()
    col4.metric(
        "🎯 Peak Liquidity", 
        f"SAR {max_liquidity:,.0f}",
        delta=f"{(current_liquidity/max_liquidity)*100:.1f}% of peak"
    )
    
    # Enhanced liquidity chart
    st.plotly_chart(
        create_liquidity_trend_chart(daily_liquidity.tail(30)), 
        use_container_width=True, 
        config={"displayModeBar": False}
    )
    
    # Liquidity insights
    with st.expander("📊 Detailed Liquidity Analysis"):
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            st.markdown("**📈 Liquidity Statistics**")
            volatility = daily_liquidity["TotalLiquidity"].std()
            st.write(f"• **Volatility:** SAR {volatility:,.0f}")
            st.write(f"• **Min Liquidity:** SAR {daily_liquidity['TotalLiquidity'].min():,.0f}")
            st.write(f"• **Max Liquidity:** SAR {max_liquidity:,.0f}")
            st.write(f"• **Range:** SAR {max_liquidity - daily_liquidity['TotalLiquidity'].min():,.0f}")
        
        with insight_col2:
            st.markdown("**🎯 Performance Indicators**")
            trend_days = min(7, len(daily_liquidity))
            recent_trend = daily_liquidity.tail(trend_days)["TotalLiquidity"].pct_change().mean() * 100
            
            if recent_trend > 1:
                trend_status = "🚀 Improving"
            elif recent_trend < -1:
                trend_status = "📉 Declining"
            else:
                trend_status = "➡️ Stable"
            
            st.write(f"• **Recent Trend:** {trend_status}")
            st.write(f"• **7-Day Avg Change:** {recent_trend:+.1f}% daily")
            
            # Risk assessment
            risk_level = "🟢 Low" if volatility < avg_liquidity * 0.1 else "🟡 Medium" if volatility < avg_liquidity * 0.2 else "🔴 High"
            st.write(f"• **Volatility Risk:** {risk_level}")


def main():
    apply_css()

    # Load and process data
    sheets_map = load_workbook_from_gsheet(config.DEFAULT_PUBLISHED_URL)
    if not sheets_map: 
        st.warning("⚠️ No data found. Please check your data source.")
        st.stop()
        
    df_all = process_branch_data(sheets_map)
    if df_all.empty: 
        st.error("❌ Could not process data. Please verify column names and sheet structure.")
        st.stop()

    # Initialize session state
    all_branches = sorted(df_all["BranchName"].dropna().unique()) if "BranchName" in df_all else []
    if "selected_branches" not in st.session_state:
        st.session_state.selected_branches = list(all_branches)

    # Enhanced Sidebar
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center;padding:20px 0;">
                <div style="font-size:1.5rem;font-weight:800;margin-bottom:8px;">
                    📊 AL KHAIR ANALYTICS
                </div>
                <div style="color:#6b7280;font-size:0.85rem;">
                    Business Intelligence Dashboard
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Quick refresh button
        if st.button("🔄 Refresh Data", use_container_width=True, type="primary"):
            load_workbook_from_gsheet.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Branch selection
        st.markdown("**🏪 Branch Selection**")
        
        # Select all/none buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Select All", use_container_width=True):
                st.session_state.selected_branches = list(all_branches)
                st.rerun()
        with col2:
            if st.button("Clear All", use_container_width=True):
                st.session_state.selected_branches = []
                st.rerun()
        
        # Branch checkboxes with performance indicators
        sel: List[str] = []
        for i, branch in enumerate(all_branches):
            # Get branch performance for indicator
            branch_data = df_all[df_all["BranchName"] == branch]
            if not branch_data.empty:
                avg_performance = branch_data["SalesPercent"].mean() if "SalesPercent" in branch_data else 0
                status_emoji = "🚀" if avg_performance >= 95 else "✅" if avg_performance >= 85 else "⚠️" if avg_performance >= 75 else "🔴"
            else:
                status_emoji = "📊"
            
            if st.checkbox(f"{status_emoji} {branch}", value=(branch in st.session_state.selected_branches), key=f"branch_{i}"):
                sel.append(branch)
        
        st.session_state.selected_branches = sel
        
        st.markdown("---")
        
        # Date range selection
        if "Date" in df_all.columns and df_all["Date"].notna().any():
            filtered_for_dates = df_all[df_all["BranchName"].isin(st.session_state.selected_branches)] if st.session_state.selected_branches else df_all
            date_min, date_max = filtered_for_dates["Date"].min().date(), filtered_for_dates["Date"].max().date()
        else:
            date_min = date_max = datetime.today().date()
        
        st.markdown("**📅 Date Range**")
        
        start_date = st.date_input("Start Date", value=date_min, min_value=date_min, max_value=date_max)
        end_date = st.date_input("End Date", value=date_max, min_value=date_min, max_value=date_max)
        
        if start_date > end_date:
            st.error("Start date cannot be after end date")
    
    # Hero Header
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-bar"></div>
          <div class="hero-title">
            <span class="hero-emoji">📊</span>
            {config.PAGE_TITLE}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Apply filters
    df = df_all.copy()
    if "BranchName" in df.columns and st.session_state.selected_branches:
        df = df[df["BranchName"].isin(st.session_state.selected_branches)].copy()
    if "Date" in df.columns and df["Date"].notna().any():
        mask = (df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)
        df = df.loc[mask].copy()
    
    if df.empty:
        st.warning("⚠️ No data available for the selected filters. Please adjust your selection.")
        st.stop()

    # Calculate KPIs
    k = calc_kpis(df)
    
    # Enhanced Quick Insights
    with st.expander("⚡ Business Insights", expanded=True):
        insights = []
        
        # Performance insights
        if k.get("performance_score", 0) >= 90:
            insights.append("🚀 **Excellent Performance**: All metrics exceeding expectations!")
        elif k.get("performance_score", 0) >= 80:
            insights.append("✅ **Good Performance**: Strong results across key metrics")
        elif k.get("performance_score", 0) >= 70:
            insights.append("⚠️ **Moderate Performance**: Some areas need attention")
        else:
            insights.append("🔴 **Performance Alert**: Multiple metrics below target")
        
        # Sales insights
        if k.get("overall_sales_percent", 0) > 100:
            insights.append(f"💰 **Sales Exceeding**: {k['overall_sales_percent']:.1f}% achievement (SAR {k.get('total_sales_variance', 0):,.0f} above target)")
        elif k.get("overall_sales_percent", 0) < 90:
            insights.append(f"📉 **Sales Gap**: {100 - k['overall_sales_percent']:.1f}% below target (SAR {abs(k.get('total_sales_variance', 0)):,.0f} shortfall)")
        
        # Branch insights
        if "branch_performance" in k and not k["branch_performance"].empty:
            bp = k["branch_performance"]
            top_performer = bp["SalesPercent"].idxmax()
            lowest_performer = bp["SalesPercent"].idxmin()
            insights.append(f"🏆 **Top Performer**: {top_performer} ({bp['SalesPercent'].max():.1f}% sales achievement)")
            if bp["SalesPercent"].max() - bp["SalesPercent"].min() > 20:
                insights.append(f"📊 **Performance Gap**: {bp['SalesPercent'].max() - bp['SalesPercent'].min():.1f}% difference between best and lowest performing branches")
        
        for insight in insights:
            st.markdown(insight)

    # Enhanced Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Overview", 
        "📈 Trends", 
        "💧 Liquidity", 
        "📊 Analytics",
        "📥 Export"
    ])

    with tab1:
        render_enhanced_overview(df, k)

    with tab2:
        render_enhanced_trends_tab(df)

    with tab3:
        render_enhanced_liquidity_tab(df)
    
    with tab4:
        st.markdown("### 📊 Advanced Analytics")
        
        if not df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Daily performance distribution
                st.markdown("#### 📈 Performance Distribution")
                if "SalesPercent" in df.columns:
                    fig = go.Figure()
                    for branch in st.session_state.selected_branches:
                        branch_data = df[df["BranchName"] == branch]["SalesPercent"].dropna()
                        if not branch_data.empty:
                            fig.add_trace(go.Box(
                                y=branch_data,
                                name=branch,
                                boxpoints='outliers'
                            ))
                    
                    fig.update_layout(
                        title="Sales Performance Distribution by Branch",
                        yaxis_title="Sales Achievement (%)",
                        height=400,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Monthly aggregation if we have enough data
                st.markdown("#### 📅 Monthly Trends")
                if "Date" in df.columns and len(df) > 30:
                    monthly_data = df.groupby([df["Date"].dt.to_period("M"), "BranchName"]).agg({
                        "SalesActual": "sum",
                        "NOBActual": "sum",
                        "ABVActual": "mean"
                    }).reset_index()
                    
                    if not monthly_data.empty:
                        fig = px.line(
                            monthly_data,
                            x="Date",
                            y="SalesActual",
                            color="BranchName",
                            title="Monthly Sales Trend"
                        )
                        fig.update_layout(
                            height=400,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.markdown("### 📥 Export & Reports")
        
        if df.empty:
            st.info("No data available for export.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                # Excel export
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                    df.to_excel(writer, sheet_name="Filtered_Data", index=False)
                    if "branch_performance" in k:
                        k["branch_performance"].to_excel(writer, sheet_name="Branch_Summary")
                
                st.download_button(
                    "📊 Download Excel Report",
                    buffer.getvalue(),
                    f"AlKhair_Analytics_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    type="primary"
                )
            
            with col2:
                # CSV export
                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📄 Download CSV Data",
                    csv_data,
                    f"AlKhair_Data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            st.markdown("---")
            
            # Summary statistics
            st.markdown("#### 📋 Export Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("📊 Total Records", f"{len(df):,}")
            col2.metric("🏪 Branches", f"{df['BranchName'].nunique()}")
            col3.metric("📅 Date Range", f"{(df['Date'].max() - df['Date'].min()).days} days")
            col4.metric("💰 Total Sales", f"SAR {df['SalesActual'].sum():,.0f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"⚠️ Application Error: {str(e)}")
        st.exception(e)
