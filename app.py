from __future__ import annotations

import io
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================================
# CONFIG
# =========================================
class Config:
    PAGE_TITLE = "Al Khair Business Performance"
    LAYOUT = "wide"
    DEFAULT_PUBLISHED_URL = (
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vROaePKJN3XhRor42BU4Kgd9fCAgW7W8vWJbwjveQWoJy8HCRBUAYh2s0AxGsBa3w/pub?output=csv"
    )
    BRANCHES = {
        "Al khair - 102": {"name": "Al Khair", "code": "102", "color": "#3b82f6"},
        "Noora - 104": {"name": "Noora", "code": "104", "color": "#10b981"},
        "Hamra - 109": {"name": "Hamra", "code": "109", "color": "#f59e0b"},
        "Magnus - 107": {"name": "Magnus", "code": "107", "color": "#8b5cf6"},
    }
    TARGETS = {
        "sales_achievement": 95.0, 
        "nob_achievement": 90.0, 
        "abv_achievement": 90.0
    }
    
    # New: Alert thresholds
    ALERT_THRESHOLDS = {
        "critical": 75.0,  # Below this = critical
        "warning": 85.0,   # Below this = warning
        "good": 95.0       # Above this = good
    }
    
    # New: Forecasting parameters
    FORECAST_DAYS = 7
    
    # New: Performance weights for scoring
    PERFORMANCE_WEIGHTS = {
        "sales": 0.4,
        "nob": 0.35,
        "abv": 0.25
    }

config = Config()
st.set_page_config(
    page_title=config.PAGE_TITLE, 
    layout=config.LAYOUT, 
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# =========================================
# ENHANCED CSS
# =========================================
def apply_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
        
        /* Base styles */
        html, body, [data-testid="stAppViewContainer"] { 
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; 
        }
        .main .block-container { 
            padding: 1.2rem 2rem; 
            max-width: 1500px; 
        }
        
        /* Typography */
        .title { 
            font-weight: 900; 
            font-size: 1.8rem; 
            margin-bottom: .25rem; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .subtitle { 
            color: #6b7280; 
            font-size: .95rem; 
            margin-bottom: 1rem; 
        }

        /* Enhanced metric containers */
        [data-testid="metric-container"] {
            background: #fff !important; 
            border-radius: 16px !important; 
            border: 1px solid rgba(0,0,0,.06) !important; 
            padding: 20px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
            transition: all 0.2s ease !important;
        }
        [data-testid="metric-container"]:hover {
            box-shadow: 0 4px 16px rgba(0,0,0,0.08) !important;
            transform: translateY(-1px) !important;
        }
        
        /* Enhanced cards */
        .card { 
            background: #fcfcfc; 
            border: 1px solid #f1f5f9; 
            border-radius: 16px; 
            padding: 20px; 
            height: 100%;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            transition: all 0.2s ease;
        }
        .card:hover {
            box-shadow: 0 4px 16px rgba(0,0,0,0.08);
            transform: translateY(-2px);
        }

        /* Status pills with animations */
        .pill { 
            display: inline-block; 
            padding: 6px 12px; 
            border-radius: 999px; 
            font-weight: 700; 
            font-size: .80rem;
            transition: all 0.2s ease;
        }
        .pill:hover {
            transform: scale(1.05);
        }
        .pill.excellent { 
            background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); 
            color: #059669; 
            border: 1px solid #a7f3d0;
        }
        .pill.good { 
            background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); 
            color: #2563eb; 
            border: 1px solid #93c5fd;
        }
        .pill.warn { 
            background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%); 
            color: #d97706; 
            border: 1px solid #fcd34d;
        }
        .pill.danger { 
            background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%); 
            color: #dc2626; 
            border: 1px solid #fca5a5;
        }
        .pill.critical {
            background: linear-gradient(135deg, #450a0a 0%, #7f1d1d 100%); 
            color: #ffffff; 
            border: 1px solid #991b1b;
            animation: pulse 2s infinite;
        }

        /* Pulse animation for critical alerts */
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }

        /* Enhanced tabs */
        .stTabs [data-baseweb="tab-list"] { 
            gap: 8px; 
            border-bottom: 2px solid #f1f5f9; 
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 12px 12px 0 0 !important;
            padding: 12px 20px !important; 
            font-weight: 700 !important;
            background: #f8fafc !important; 
            color: #64748b !important;
            border: 1px solid #e2e8f0 !important; 
            border-bottom: none !important;
            transition: all 0.2s ease !important;
        }
        .stTabs [data-baseweb="tab"]:hover { 
            background: #e2e8f0 !important; 
            transform: translateY(-1px) !important;
        }
        .stTabs [aria-selected="true"] { 
            color: #fff !important; 
            box-shadow: 0 6px 20px rgba(0,0,0,0.15) !important;
            transform: translateY(-2px) !important;
        }

        /* Gradient backgrounds for active tabs */
        .stTabs [data-baseweb="tab"]:nth-child(1)[aria-selected="true"] { 
            background: linear-gradient(135deg, #ef4444 0%, #f87171 100%) !important; 
        }
        .stTabs [data-baseweb="tab"]:nth-child(2)[aria-selected="true"] { 
            background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%) !important; 
        }
        .stTabs [data-baseweb="tab"]:nth-child(3)[aria-selected="true"] { 
            background: linear-gradient(135deg, #8b5cf6 0%, #a78bfa 100%) !important; 
        }
        .stTabs [data-baseweb="tab"]:nth-child(4)[aria-selected="true"] { 
            background: linear-gradient(135deg, #10b981 0%, #34d399 100%) !important; 
        }

        /* Enhanced sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%); 
            border-right: 1px solid #e2e8f0; 
            min-width: 300px; 
            max-width: 320px;
        }
        [data-testid="stSidebar"] .block-container { 
            padding: 20px 18px; 
        }
        .sb-title { 
            display: flex; 
            gap: 12px; 
            align-items: center; 
            font-weight: 900; 
            font-size: 1.1rem; 
            color: #1e293b;
        }
        .sb-subtle { 
            color: #64748b; 
            font-size: .85rem; 
            margin: 8px 0 16px; 
        }
        .sb-section { 
            font-size: .78rem; 
            font-weight: 800; 
            letter-spacing: .03em; 
            text-transform: uppercase; 
            color: #475569; 
            margin: 18px 4px 8px; 
        }
        .sb-card { 
            background: #ffffff; 
            border: 1px solid #e2e8f0; 
            border-radius: 16px; 
            padding: 16px; 
            box-shadow: 0 2px 4px rgba(0,0,0,.04);
            transition: all 0.2s ease;
        }
        .sb-card:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,.08);
        }
        .sb-hr { 
            height: 1px; 
            background: linear-gradient(90deg, transparent, #e2e8f0, transparent); 
            margin: 16px 0; 
        }
        .sb-foot { 
            margin-top: 12px; 
            font-size: .72rem; 
            color: #94a3b8; 
            text-align: center; 
        }

        /* Alert cards */
        .alert-card {
            border-radius: 12px;
            padding: 16px;
            margin: 8px 0;
            border-left: 4px solid;
        }
        .alert-critical {
            background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
            border-left-color: #dc2626;
            color: #7f1d1d;
        }
        .alert-warning {
            background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
            border-left-color: #f59e0b;
            color: #92400e;
        }
        .alert-info {
            background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
            border-left-color: #3b82f6;
            color: #1e40af;
        }

        /* Responsive design */
        @media (max-width: 768px) {
            .main .block-container { 
                padding: 0.8rem 1rem; 
            }
            .title { 
                font-size: 1.4rem; 
            }
            .subtitle { 
                font-size: .82rem; 
            }
            [data-testid="metric-container"] { 
                padding: 16px !important; 
            }
            [data-testid="stSidebar"] { 
                display: none; 
            }
            .mobile-only { 
                display: block !important; 
            }
            .desktop-only { 
                display: none !important; 
            }
        }
        .mobile-only { 
            display: none; 
        }
        
        /* Loading animations */
        .loading-spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #3b82f6;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# =========================================
# ENHANCED DATA PROCESSING
# =========================================
def _pubhtml_to_xlsx(url: str) -> str:
    """Convert Google Sheets pubhtml URL to xlsx download URL."""
    if "docs.google.com/spreadsheets/d/e/" in url:
        return re.sub(r"/pubhtml(.*)$", "/pub?output=xlsx", url.strip())
    return url

def _parse_numeric(s: pd.Series) -> pd.Series:
    """Enhanced numeric parsing with better error handling."""
    try:
        return pd.to_numeric(
            s.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.replace("SAR", "", regex=False)
            .str.strip(),
            errors="coerce",
        )
    except Exception as e:
        logger.warning(f"Error parsing numeric data: {e}")
        return pd.Series([0] * len(s))

@st.cache_data(show_spinner=False, ttl=300)  # 5-minute cache
def load_workbook_from_gsheet(published_url: str) -> Dict[str, pd.DataFrame]:
    """Load Excel workbook from Google Sheets with enhanced error handling."""
    try:
        url = _pubhtml_to_xlsx((published_url or config.DEFAULT_PUBLISHED_URL).strip())
        logger.info(f"Loading data from: {url}")
        
        with st.spinner("🔄 Loading data from Google Sheets..."):
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            xls = pd.ExcelFile(io.BytesIO(response.content))
            sheets: Dict[str, pd.DataFrame] = {}
            
            for sheet_name in xls.sheet_names:
                try:
                    df = xls.parse(sheet_name)
                    df = df.dropna(how="all").dropna(axis=1, how="all")
                    if not df.empty:
                        sheets[sheet_name] = df
                        logger.info(f"Loaded sheet '{sheet_name}' with {len(df)} rows")
                except Exception as e:
                    logger.warning(f"Error processing sheet '{sheet_name}': {e}")
                    
        return sheets
        
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Network error: {str(e)}")
        return {}
    except Exception as e:
        st.error(f"❌ Data loading error: {str(e)}")
        return {}

def process_branch_data(excel_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Enhanced branch data processing with better column mapping."""
    if not excel_data:
        return pd.DataFrame()
        
    combined: List[pd.DataFrame] = []
    
    # Enhanced column mapping
    mapping = {
        "Date": "Date",
        "Day": "Day", 
        "Sales Target": "SalesTarget",
        "SalesTarget": "SalesTarget",
        "Sales Achivement": "SalesActual",  # Keep typo for compatibility
        "Sales Achievement": "SalesActual",
        "SalesActual": "SalesActual",
        "Sales %": "SalesPercent",
        " Sales %": "SalesPercent",
        "SalesPercent": "SalesPercent",
        "NOB Target": "NOBTarget",
        "NOBTarget": "NOBTarget",
        "NOB Achievemnet": "NOBActual",  # Keep typo for compatibility
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
            
        try:
            processed_df = df.copy()
            processed_df.columns = processed_df.columns.astype(str).str.strip()
            
            # Apply column mapping
            for old_col, new_col in mapping.items():
                if old_col in processed_df.columns and new_col not in processed_df.columns:
                    processed_df.rename(columns={old_col: new_col}, inplace=True)
            
            # Handle date column detection
            if "Date" not in processed_df.columns:
                date_candidates = [c for c in processed_df.columns if "date" in c.lower()]
                if date_candidates:
                    processed_df.rename(columns={date_candidates[0]: "Date"}, inplace=True)
            
            # Add branch metadata
            processed_df["Branch"] = sheet_name
            branch_meta = config.BRANCHES.get(sheet_name, {})
            processed_df["BranchName"] = branch_meta.get("name", sheet_name)
            processed_df["BranchCode"] = branch_meta.get("code", "000")
            
            # Parse dates
            if "Date" in processed_df.columns:
                processed_df["Date"] = pd.to_datetime(processed_df["Date"], errors="coerce")
            
            # Parse numeric columns
            numeric_columns = [
                "SalesTarget", "SalesActual", "SalesPercent",
                "NOBTarget", "NOBActual", "NOBPercent", 
                "ABVTarget", "ABVActual", "ABVPercent"
            ]
            
            for col in numeric_columns:
                if col in processed_df.columns:
                    processed_df[col] = _parse_numeric(processed_df[col])
            
            combined.append(processed_df)
            
        except Exception as e:
            logger.error(f"Error processing sheet '{sheet_name}': {e}")
            continue
    
    if combined:
        result = pd.concat(combined, ignore_index=True)
        logger.info(f"Combined data: {len(result)} rows, {len(result.columns)} columns")
        return result
    
    return pd.DataFrame()

# =========================================
# ENHANCED KPI CALCULATIONS
# =========================================
def calc_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Enhanced KPI calculations with additional metrics."""
    if df.empty:
        return {}
    
    kpis: Dict[str, Any] = {}
    
    # Basic sales metrics
    kpis["total_sales_target"] = float(df.get("SalesTarget", pd.Series(dtype=float)).sum())
    kpis["total_sales_actual"] = float(df.get("SalesActual", pd.Series(dtype=float)).sum())
    kpis["total_sales_variance"] = kpis["total_sales_actual"] - kpis["total_sales_target"]
    kpis["overall_sales_percent"] = (
        kpis["total_sales_actual"] / kpis["total_sales_target"] * 100 
        if kpis["total_sales_target"] > 0 else 0.0
    )
    
    # NOB metrics
    kpis["total_nob_target"] = float(df.get("NOBTarget", pd.Series(dtype=float)).sum())
    kpis["total_nob_actual"] = float(df.get("NOBActual", pd.Series(dtype=float)).sum())
    kpis["overall_nob_percent"] = (
        kpis["total_nob_actual"] / kpis["total_nob_target"] * 100 
        if kpis["total_nob_target"] > 0 else 0.0
    )
    
    # ABV metrics
    kpis["avg_abv_target"] = float(df.get("ABVTarget", pd.Series(dtype=float)).mean()) if "ABVTarget" in df else 0.0
    kpis["avg_abv_actual"] = float(df.get("ABVActual", pd.Series(dtype=float)).mean()) if "ABVActual" in df else 0.0
    kpis["overall_abv_percent"] = (
        kpis["avg_abv_actual"] / kpis["avg_abv_target"] * 100 
        if "ABVTarget" in df and kpis["avg_abv_target"] > 0 else 0.0
    )
    
    # Enhanced performance score using configurable weights
    weights = config.PERFORMANCE_WEIGHTS
    score = 0.0
    total_weight = 0.0
    
    if kpis.get("overall_sales_percent", 0) > 0:
        score += min(kpis["overall_sales_percent"] / 100, 1.2) * weights["sales"] * 100
        total_weight += weights["sales"]
    
    if kpis.get("overall_nob_percent", 0) > 0:
        score += min(kpis["overall_nob_percent"] / 100, 1.2) * weights["nob"] * 100
        total_weight += weights["nob"]
    
    if kpis.get("overall_abv_percent", 0) > 0:
        score += min(kpis["overall_abv_percent"] / 100, 1.2) * weights["abv"] * 100
        total_weight += weights["abv"]
    
    kpis["performance_score"] = (score / total_weight) if total_weight > 0 else 0.0
    
    # Branch performance analysis
    if "BranchName" in df.columns:
        branch_agg = df.groupby("BranchName").agg({
            "SalesTarget": "sum",
            "SalesActual": "sum", 
            "SalesPercent": "mean",
            "NOBTarget": "sum",
            "NOBActual": "sum",
            "NOBPercent": "mean",
            "ABVTarget": "mean",
            "ABVActual": "mean", 
            "ABVPercent": "mean",
        }).round(2)
        
        kpis["branch_performance"] = branch_agg
    
    # Time-based analysis
    if "Date" in df.columns and df["Date"].notna().any():
        date_range = {
            "start": df["Date"].min(),
            "end": df["Date"].max(),
            "days": int(df["Date"].dt.date.nunique())
        }
        kpis["date_range"] = date_range
        
        # Daily trends
        daily_trends = df.groupby("Date").agg({
            "SalesActual": "sum",
            "SalesTarget": "sum", 
            "NOBActual": "sum",
            "ABVActual": "mean"
        }).round(2)
        kpis["daily_trends"] = daily_trends
    
    # Alert generation
    alerts = generate_alerts(kpis)
    kpis["alerts"] = alerts
    
    return kpis

def generate_alerts(kpis: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate performance alerts based on thresholds."""
    alerts = []
    thresholds = config.ALERT_THRESHOLDS
    
    # Sales alerts
    sales_pct = kpis.get("overall_sales_percent", 0)
    if sales_pct < thresholds["critical"]:
        alerts.append({
            "type": "critical",
            "title": "Critical Sales Performance", 
            "message": f"Overall sales achievement at {sales_pct:.1f}% - immediate attention required!"
        })
    elif sales_pct < thresholds["warning"]:
        alerts.append({
            "type": "warning",
            "title": "Sales Below Target",
            "message": f"Sales achievement at {sales_pct:.1f}% - below target threshold"
        })
    
    # NOB alerts 
    nob_pct = kpis.get("overall_nob_percent", 0)
    if nob_pct < thresholds["critical"]:
        alerts.append({
            "type": "critical",
            "title": "Critical Basket Count",
            "message": f"Number of baskets at {nob_pct:.1f}% - customer acquisition issue!"
        })
    elif nob_pct < thresholds["warning"]:
        alerts.append({
            "type": "warning", 
            "title": "Low Basket Count",
            "message": f"NOB achievement at {nob_pct:.1f}% - monitor customer traffic"
        })
    
    # Branch-specific alerts
    if "branch_performance" in kpis:
        bp = kpis["branch_performance"]
        for branch in bp.index:
            branch_sales = bp.loc[branch, "SalesPercent"]
            if branch_sales < thresholds["critical"]:
                alerts.append({
                    "type": "critical",
                    "title": f"{branch} Critical Performance",
                    "message": f"Branch sales at {branch_sales:.1f}% - urgent intervention needed!"
                })
    
    return alerts

# =========================================
# ENHANCED VISUALIZATIONS  
# =========================================
def create_trend_chart(df: pd.DataFrame, y_col: str, title: str, show_target: bool = True) -> go.Figure:
    """Create enhanced trend chart with better styling."""
    if df.empty or "Date" not in df.columns or df["Date"].isna().all():
        return go.Figure().add_annotation(
            text="No data available for the selected period",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="#64748b")
        )

    # Aggregate data by date and branch
    agg_func = "sum" if y_col != "ABVActual" else "mean"
    daily_data = df.groupby(["Date", "BranchName"]).agg({y_col: agg_func}).reset_index()
    
    # Target data if available
    target_mapping = {"SalesActual": "SalesTarget", "NOBActual": "NOBTarget", "ABVActual": "ABVTarget"}
    target_col = target_mapping.get(y_col)
    daily_target = None
    
    if show_target and target_col and target_col in df.columns:
        agg_func_target = "sum" if y_col != "ABVActual" else "mean"
        daily_target = df.groupby(["Date", "BranchName"]).agg({target_col: agg_func_target}).reset_index()

    # Create subplot
    fig = go.Figure()
    
    # Enhanced color palette
    colors = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444", "#14b8a6", "#f97316"]
    
    for i, branch in enumerate(sorted(daily_data["BranchName"].unique())):
        branch_data = daily_data[daily_data["BranchName"] == branch]
        color = colors[i % len(colors)]
        
        # Actual values with area fill
        fig.add_trace(go.Scatter(
            x=branch_data["Date"],
            y=branch_data[y_col],
            name=f"{branch}",
            mode="lines+markers",
            line=dict(width=3, color=color, shape="spline"),
            marker=dict(size=6, color=color, line=dict(width=2, color="white")),
            fill="tonexty" if i > 0 else "tozeroy",
            fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)",
            hovertemplate=f"<b>{branch}</b><br>Date: %{{x|%b %d, %Y}}<br>Value: %{{y:,.0f}}<extra></extra>",
        ))
        
        # Target line if available
        if daily_target is not None:
            target_data = daily_target[daily_target["BranchName"] == branch]
            if not target_data.empty:
                fig.add_trace(go.Scatter(
                    x=target_data["Date"],
                    y=target_data[target_col],
                    name=f"{branch} Target",
                    mode="lines",
                    line=dict(width=2, color=color, dash="dash"),
                    opacity=0.7,
                    showlegend=False,
                    hovertemplate=f"<b>{branch} Target</b><br>Date: %{{x|%b %d, %Y}}<br>Target: %{{y:,.0f}}<extra></extra>",
                ))

    # Enhanced layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, weight="bold"),
            x=0.02
        ),
        height=450,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right", 
            x=1
        ),
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(0,0,0,0.1)",
            showline=True,
            linewidth=1,
            linecolor="rgba(0,0,0,0.1)"
        ),
        yaxis=dict(
            title="Value",
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(0,0,0,0.1)",
            showline=True,
            linewidth=1,
            linecolor="rgba(0,0,0,0.1)"
        ),
        hovermode="x unified"
    )
    
    return fig

def create_performance_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create a performance heatmap for branches and metrics."""
    if df.empty or "BranchName" not in df.columns:
        return go.Figure()
    
    # Prepare data for heatmap
    metrics = ["SalesPercent", "NOBPercent", "ABVPercent"]
    metric_names = ["Sales %", "NOB %", "ABV %"]
    
    branch_performance = df.groupby("BranchName").agg({
        "SalesPercent": "mean",
        "NOBPercent": "mean", 
        "ABVPercent": "mean"
    }).round(1)
    
    if branch_performance.empty:
        return go.Figure()
    
    # Create heatmap data
    z_data = []
    branches = list(branch_performance.index)
    
    for metric in metrics:
        z_data.append(branch_performance[metric].tolist())
    
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=branches,
        y=metric_names,
        colorscale=[
            [0, "#fef2f2"],      # Light red
            [0.25, "#fef3c7"],   # Light yellow  
            [0.5, "#ecfdf5"],    # Light green
            [0.75, "#dbeafe"],   # Light blue
            [1, "#e0e7ff"]       # Light purple
        ],
        text=z_data,
        texttemplate="%{text:.1f}%",
        textfont={"size": 14, "color": "black"},
        hovertemplate="<b>%{y}</b><br>Branch: %{x}<br>Performance: %{z:.1f}%<extra></extra>",
        colorbar=dict(
            title="Performance %",
            titleside="right"
        )
    ))
    
    fig.update_layout(
        title="Branch Performance Heatmap",
        height=300,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Branches",
        yaxis_title="Metrics"
    )
    
    return fig

def create_comparison_chart(bp: pd.DataFrame) -> go.Figure:
    """Enhanced branch comparison chart."""
    if bp.empty:
        return go.Figure()
    
    metrics = {"SalesPercent": "Sales %", "NOBPercent": "NOB %", "ABVPercent": "ABV %"}
    colors = ["#3b82f6", "#10b981", "#f59e0b"]
    
    fig = go.Figure()
    
    branches = bp.index.tolist()
    
    for i, (col, label) in enumerate(metrics.items()):
        if col in bp.columns:
            values = bp[col].tolist()
            
            fig.add_trace(go.Bar(
                x=branches,
                y=values,
                name=label,
                marker=dict(
                    color=colors[i % len(colors)],
                    line=dict(color="white", width=2)
                ),
                text=[f"{v:.1f}%" for v in values],
                textposition="outside",
                hovertemplate=f"<b>{label}</b><br>Branch: %{{x}}<br>Performance: %{{y:.1f}}%<extra></extra>"
            ))
    
    # Add target lines
    fig.add_hline(
        y=config.TARGETS["sales_achievement"], 
        line_dash="dash", 
        line_color="red",
        annotation_text="Sales Target (95%)",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title="Branch Performance Comparison",
        barmode="group",
        height=400,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Branches"),
        yaxis=dict(title="Performance %", range=[0, max(110, bp[list(metrics.keys())].max().max() + 10)])
    )
    
    return fig

def create_gauge_chart(value: float, title: str, max_value: float = 120) -> go.Figure:
    """Create a gauge chart for KPI visualization."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16}},
        delta={'reference': 100, 'position': "top"},
        gauge={
            'axis': {'range': [None, max_value], 'tickwidth': 1},
            'bar': {'color': "#3b82f6"},
            'steps': [
                {'range': [0, 75], 'color': "#fef2f2"},
                {'range': [75, 90], 'color': "#fffbeb"}, 
                {'range': [90, 100], 'color': "#f0f9ff"},
                {'range': [100, max_value], 'color': "#ecfdf5"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 100
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig

# =========================================
# ENHANCED UI COMPONENTS
# =========================================
def render_alerts(alerts: List[Dict[str, str]]):
    """Render performance alerts with enhanced styling."""
    if not alerts:
        return
    
    st.markdown("### 🚨 Performance Alerts")
    
    for alert in alerts:
        alert_type = alert.get("type", "info")
        title = alert.get("title", "Alert")
        message = alert.get("message", "")
        
        icon_map = {
            "critical": "🔴",
            "warning": "🟡", 
            "info": "🔵"
        }
        
        icon = icon_map.get(alert_type, "ℹ️")
        
        st.markdown(f"""
            <div class="alert-card alert-{alert_type}">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 1.2em;">{icon}</span>
                    <div>
                        <div style="font-weight: 700; margin-bottom: 4px;">{title}</div>
                        <div style="font-size: 0.9em;">{message}</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

def render_enhanced_metrics(kpis: Dict[str, Any]):
    """Render enhanced KPI metrics with improved styling."""
    st.markdown("### 🏆 Overall Performance")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Sales metrics
    with col1:
        sales_actual = kpis.get('total_sales_actual', 0)
        sales_percent = kpis.get('overall_sales_percent', 0)
        delta_color = "normal" if sales_percent >= 95 else "inverse"
        
        st.metric(
            "💰 Total Sales",
            f"SAR {sales_actual:,.0f}",
            delta=f"{sales_percent:.1f}% of target",
            delta_color=delta_color
        )
    
    # Variance
    with col2:
        variance = kpis.get('total_sales_variance', 0)
        variance_pct = sales_percent - 100
        variance_color = "normal" if variance >= 0 else "inverse"
        
        st.metric(
            "📊 Sales Variance", 
            f"SAR {variance:,.0f}",
            delta=f"{variance_pct:+.1f}%",
            delta_color=variance_color
        )
    
    # NOB metrics
    with col3:
        nob_actual = kpis.get('total_nob_actual', 0)
        nob_percent = kpis.get('overall_nob_percent', 0)
        nob_color = "normal" if nob_percent >= config.TARGETS['nob_achievement'] else "inverse"
        
        st.metric(
            "🛍️ Total Baskets",
            f"{nob_actual:,.0f}",
            delta=f"{nob_percent:.1f}% achievement", 
            delta_color=nob_color
        )
    
    # ABV metrics
    with col4:
        abv_actual = kpis.get('avg_abv_actual', 0)
        abv_percent = kpis.get('overall_abv_percent', 0)
        abv_color = "normal" if abv_percent >= config.TARGETS['abv_achievement'] else "inverse"
        
        st.metric(
            "💎 Avg Basket Value",
            f"SAR {abv_actual:,.2f}",
            delta=f"{abv_percent:.1f}% vs target",
            delta_color=abv_color
        )
    
    # Performance score
    with col5:
        score = kpis.get('performance_score', 0)
        score_color = "normal" if score >= 80 else "off"
        
        st.metric(
            "⭐ Performance Score",
            f"{score:.0f}/100",
            delta="Weighted average",
            delta_color=score_color
        )

def render_branch_cards(bp: pd.DataFrame):
    """Enhanced branch cards with improved styling and functionality."""
    if bp.empty:
        st.info("No branch data available.")
        return
    
    st.markdown("### 🏪 Branch Performance Overview")
    
    # Sort branches by performance
    bp_sorted = bp.copy()
    bp_sorted['avg_performance'] = bp_sorted[['SalesPercent', 'NOBPercent', 'ABVPercent']].mean(axis=1)
    bp_sorted = bp_sorted.sort_values('avg_performance', ascending=False)
    
    # Create responsive grid
    branches = list(bp_sorted.index)
    cols_per_row = min(3, len(branches))
    
    for i in range(0, len(branches), cols_per_row):
        row_branches = branches[i:i + cols_per_row]
        cols = st.columns(len(row_branches))
        
        for j, branch in enumerate(row_branches):
            with cols[j]:
                branch_data = bp_sorted.loc[branch]
                avg_performance = branch_data['avg_performance']
                
                # Determine status and styling
                if avg_performance >= config.ALERT_THRESHOLDS["good"]:
                    status_class = "excellent"
                    status_emoji = "🟢"
                elif avg_performance >= config.ALERT_THRESHOLDS["warning"]:
                    status_class = "good"
                    status_emoji = "🟡"
                elif avg_performance >= config.ALERT_THRESHOLDS["critical"]:
                    status_class = "warn"
                    status_emoji = "🟠"
                else:
                    status_class = "danger"
                    status_emoji = "🔴"
                
                # Get branch color
                branch_color = "#6b7280"  # default
                for branch_key, meta in config.BRANCHES.items():
                    if meta.get("name") == branch:
                        branch_color = meta.get("color", "#6b7280")
                        break
                
                # Render enhanced card
                st.markdown(f"""
                    <div class="card" style="background: linear-gradient(135deg, {branch_color}15 0%, #ffffff 50%);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                            <div style="font-weight: 800; font-size: 1.1em; color: #1e293b;">{branch}</div>
                            <div style="display: flex; align-items: center; gap: 6px;">
                                <span>{status_emoji}</span>
                                <span class="pill {status_class}">{avg_performance:.1f}%</span>
                            </div>
                        </div>
                        
                        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 16px;">
                            <div style="text-align: center;">
                                <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 4px;">Sales</div>
                                <div style="font-weight: 700; font-size: 1.1em; color: #1e293b;">{branch_data.get('SalesPercent', 0):.1f}%</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 4px;">NOB</div>
                                <div style="font-weight: 700; font-size: 1.1em; color: #1e293b;">{branch_data.get('NOBPercent', 0):.1f}%</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 4px;">ABV</div>
                                <div style="font-weight: 700; font-size: 1.1em; color: #1e293b;">{branch_data.get('ABVPercent', 0):.1f}%</div>
                            </div>
                        </div>
                        
                        <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #e2e8f0; font-size: 0.8em; color: #64748b;">
                            Actual Sales: SAR {branch_data.get('SalesActual', 0):,.0f}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

def render_branch_filter_buttons(df: pd.DataFrame, key_prefix: str = "br_") -> List[str]:
    """Enhanced branch filter with better UX."""
    if "BranchName" not in df.columns:
        return []

    all_branches = sorted([b for b in df["BranchName"].dropna().unique()])
    
    if "selected_branches" not in st.session_state:
        st.session_state.selected_branches = all_branches.copy()

    st.markdown("### 🏪 Branch Filter")
    
    # Control buttons
    col_controls, col_info = st.columns([3, 1])
    with col_controls:
        btn_cols = st.columns(4)
        with btn_cols[0]:
            if st.button("✅ Select All", use_container_width=True):
                st.session_state.selected_branches = all_branches.copy()
                st.rerun()
        with btn_cols[1]:
            if st.button("❌ Clear All", use_container_width=True):
                st.session_state.selected_branches = []
                st.rerun()
        with btn_cols[2]:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.selected_branches = all_branches.copy()
                st.rerun()
    
    with col_info:
        st.info(f"Selected: {len(st.session_state.selected_branches)}/{len(all_branches)}")
    
    # Branch toggle buttons
    if all_branches:
        btn_cols = st.columns(len(all_branches))
        for i, branch in enumerate(all_branches):
            with btn_cols[i]:
                is_selected = branch in st.session_state.selected_branches
                button_style = "✅" if is_selected else "⬜"
                label = f"{button_style} {branch}"
                
                if st.button(label, key=f"{key_prefix}{i}", use_container_width=True):
                    if is_selected:
                        st.session_state.selected_branches.remove(branch)
                    else:
                        st.session_state.selected_branches.append(branch)
                    st.rerun()

    return st.session_state.selected_branches

# =========================================
# MAIN APPLICATION
# =========================================
def main():
    """Enhanced main application with improved error handling and features."""
    apply_css()
    
    # Initialize session state
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    try:
        # Load and process data
        with st.spinner("🔄 Loading dashboard data..."):
            sheets_data = load_workbook_from_gsheet(config.DEFAULT_PUBLISHED_URL)
            
        if not sheets_data:
            st.error("❌ Unable to load data from Google Sheets. Please check the URL and try again.")
            st.stop()

        df_all = process_branch_data(sheets_data)
        
        if df_all.empty:
            st.error("❌ No valid data found. Please check the sheet structure and column names.")
            st.stop()

        # Initialize filter state
        all_branches = sorted(df_all["BranchName"].dropna().unique()) if "BranchName" in df_all else []
        if "selected_branches" not in st.session_state:
            st.session_state.selected_branches = list(all_branches)

        # Calculate date bounds
        if "Date" in df_all.columns and df_all["Date"].notna().any():
            date_min = df_all["Date"].min().date()
            date_max = df_all["Date"].max().date()
        else:
            date_min = date_max = datetime.today().date()

        if "start_date" not in st.session_state:
            st.session_state.start_date = max(date_min, date_max - timedelta(days=30))
        if "end_date" not in st.session_state:
            st.session_state.end_date = date_max

        # Sidebar
        with st.sidebar:
            st.markdown('<div class="sb-title">📊 <span>AL KHAIR DASHBOARD</span></div>', unsafe_allow_html=True)
            st.markdown('<div class="sb-subtle">Real-time business performance analytics</div>', unsafe_allow_html=True)

            # Actions
            st.markdown('<div class="sb-section">Quick Actions</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            if col1.button("🔄 Refresh", use_container_width=True):
                load_workbook_from_gsheet.clear()
                st.session_state.last_refresh = datetime.now()
                st.rerun()
                
            if col2.button("🧹 Reset", use_container_width=True):
                st.session_state.selected_branches = list(all_branches)
                st.session_state.start_date = max(date_min, date_max - timedelta(days=30))
                st.session_state.end_date = date_max
                st.rerun()

            st.markdown('<div class="sb-hr"></div>', unsafe_allow_html=True)

            # Quick date ranges
            st.markdown('<div class="sb-section">Quick Date Ranges</div>', unsafe_allow_html=True)
            
            date_presets = {
                "Last 7 days": 7,
                "Last 30 days": 30,
                "Last 90 days": 90,
                "Year to date": None,
                "All time": 0
            }
            
            selected_preset = st.radio("", list(date_presets.keys()), index=1, key="date_preset")
            
            if date_presets[selected_preset] is not None:
                if date_presets[selected_preset] == 0:  # All time
                    st.session_state.start_date = date_min
                    st.session_state.end_date = date_max
                else:
                    days = date_presets[selected_preset]
                    if days is None:  # YTD
                        st.session_state.start_date = date_max.replace(month=1, day=1)
                        st.session_state.end_date = date_max
                    else:
                        st.session_state.start_date = max(date_min, date_max - timedelta(days=days))
                        st.session_state.end_date = date_max

            # Custom date range
            st.markdown('<div class="sb-section">Custom Date Range</div>', unsafe_allow_html=True)
            start_date = st.date_input("Start Date", value=st.session_state.start_date, min_value=date_min, max_value=date_max)
            end_date = st.date_input("End Date", value=st.session_state.end_date, min_value=date_min, max_value=date_max)
            
            st.session_state.start_date = start_date
            st.session_state.end_date = end_date

            st.markdown('<div class="sb-hr"></div>', unsafe_allow_html=True)

            # Dashboard info
            st.markdown('<div class="sb-section">Dashboard Info</div>', unsafe_allow_html=True)
            st.markdown(f"""
                <div class="sb-card">
                    <div style="font-size: 0.85em;">
                        <div><strong>Total Branches:</strong> {len(all_branches)}</div>
                        <div><strong>Data Points:</strong> {len(df_all):,}</div>
                        <div><strong>Date Range:</strong> {date_min} to {date_max}</div>
                        <div><strong>Last Refresh:</strong> {st.session_state.last_refresh.strftime('%H:%M:%S')}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="sb-foot">Enhanced Dashboard v3.0</div>', unsafe_allow_html=True)

        # Main header
        st.markdown(f"<div class='title'>📊 {config.PAGE_TITLE}</div>", unsafe_allow_html=True)
        
        date_range_text = f"{st.session_state.start_date} → {st.session_state.end_date}"
        branch_count = len(st.session_state.selected_branches)
        total_branches = len(all_branches)
        
        st.markdown(
            f"<div class='subtitle'>Showing {branch_count}/{total_branches} branches • {len(df_all):,} data points • {date_range_text}</div>",
            unsafe_allow_html=True,
        )

        # Apply filters
        df_filtered = df_all.copy()
        
        # Branch filter
        selected_branches = render_branch_filter_buttons(df_filtered)
        if selected_branches:
            df_filtered = df_filtered[df_filtered["BranchName"].isin(selected_branches)]
        
        # Date filter
        if "Date" in df_filtered.columns and df_filtered["Date"].notna().any():
            date_mask = (
                (df_filtered["Date"].dt.date >= st.session_state.start_date) & 
                (df_filtered["Date"].dt.date <= st.session_state.end_date)
            )
            df_filtered = df_filtered.loc[date_mask]

        if df_filtered.empty:
            st.warning("⚠️ No data matches the current filters. Please adjust your selection.")
            st.stop()

        # Calculate KPIs
        kpis = calc_kpis(df_filtered)
        
        # Show alerts
        if kpis.get("alerts"):
            render_alerts(kpis["alerts"])

        # Main tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🏠 Overview", "📈 Trends", "📊 Analytics", "📥 Export"])

        with tab1:
            render_enhanced_metrics(kpis)
            
            if "branch_performance" in kpis and not kpis["branch_performance"].empty:
                render_branch_cards(kpis["branch_performance"])
                
                # Performance comparison chart
                st.markdown("### 📊 Performance Comparison")
                comparison_chart = create_comparison_chart(kpis["branch_performance"])
                st.plotly_chart(comparison_chart, use_container_width=True, config={"displayModeBar": False})

        with tab2:
            st.markdown("### 📈 Daily Performance Trends")
            
            # Time period selector
            trend_options = ["Last 7 days", "Last 30 days", "Last 90 days", "All selected data"]
            selected_trend_period = st.selectbox("Time Period", trend_options, index=1)
            
            # Filter data based on selection
            trend_df = df_filtered.copy()
            if "Date" in trend_df.columns and selected_trend_period != "All selected data":
                days_map = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90}
                days = days_map.get(selected_trend_period, 30)
                latest_date = trend_df["Date"].max().date()
                start_date = latest_date - timedelta(days=days)
                trend_df = trend_df[trend_df["Date"].dt.date >= start_date]

            # Create trend charts
            metrics_tabs = st.tabs(["💰 Sales Trends", "🛍️ Basket Trends", "💎 Value Trends"])
            
            with metrics_tabs[0]:
                sales_chart = create_trend_chart(trend_df, "SalesActual", "Daily Sales Performance")
                st.plotly_chart(sales_chart, use_container_width=True, config={"displayModeBar": False})
                
            with metrics_tabs[1]:
                nob_chart = create_trend_chart(trend_df, "NOBActual", "Daily Number of Baskets")
                st.plotly_chart(nob_chart, use_container_width=True, config={"displayModeBar": False})
                
            with metrics_tabs[2]:
                abv_chart = create_trend_chart(trend_df, "ABVActual", "Daily Average Basket Value")
                st.plotly_chart(abv_chart, use_container_width=True, config={"displayModeBar": False})

        with tab3:
            st.markdown("### 📊 Advanced Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Performance heatmap
                st.markdown("#### Performance Heatmap")
                heatmap = create_performance_heatmap(df_filtered)
                st.plotly_chart(heatmap, use_container_width=True, config={"displayModeBar": False})
            
            with col2:
                # Gauge charts
                st.markdown("#### Key Performance Indicators")
                
                gauge_col1, gauge_col2 = st.columns(2)
                with gauge_col1:
                    sales_gauge = create_gauge_chart(kpis.get("overall_sales_percent", 0), "Sales Achievement %")
                    st.plotly_chart(sales_gauge, use_container_width=True, config={"displayModeBar": False})
                
                with gauge_col2:
                    nob_gauge = create_gauge_chart(kpis.get("overall_nob_percent", 0), "NOB Achievement %")
                    st.plotly_chart(nob_gauge, use_container_width=True, config={"displayModeBar": False})
            
            # Detailed performance table
            if "branch_performance" in kpis and not kpis["branch_performance"].empty:
                st.markdown("#### Detailed Performance Table")
                
                performance_df = kpis["branch_performance"].round(2)
                
                # Add calculated columns
                performance_df["Sales_Achievement_Status"] = performance_df["SalesPercent"].apply(
                    lambda x: "🟢 Excellent" if x >= 95 else "🟡 Good" if x >= 85 else "🟠 Warning" if x >= 75 else "🔴 Critical"
                )
                
                # Display with formatting
                st.dataframe(
                    performance_df.style.format({
                        'SalesTarget': '{:,.0f}',
                        'SalesActual': '{:,.0f}',
                        'SalesPercent': '{:.1f}%',
                        'NOBTarget': '{:,.0f}',
                        'NOBActual': '{:,.0f}',
                        'NOBPercent': '{:.1f}%',
                        'ABVTarget': '{:.2f}',
                        'ABVActual': '{:.2f}',
                        'ABVPercent': '{:.1f}%'
                    }),
                    use_container_width=True
                )

        with tab4:
            st.markdown("### 📥 Data Export")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Export Options")
                
                # Excel export with multiple sheets
                if not df_filtered.empty:
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                        # Main data
                        df_filtered.to_excel(writer, sheet_name="Raw_Data", index=False)
                        
                        # Branch performance summary
                        if "branch_performance" in kpis:
                            kpis["branch_performance"].to_excel(writer, sheet_name="Branch_Summary")
                        
                        # Daily trends if available
                        if "daily_trends" in kpis:
                            kpis["daily_trends"].to_excel(writer, sheet_name="Daily_Trends")
                        
                        # KPI summary
                        kpi_summary = pd.DataFrame([{
                            'Metric': 'Total Sales Actual',
                            'Value': kpis.get('total_sales_actual', 0),
                            'Unit': 'SAR'
                        }, {
                            'Metric': 'Total Sales Target', 
                            'Value': kpis.get('total_sales_target', 0),
                            'Unit': 'SAR'
                        }, {
                            'Metric': 'Sales Achievement',
                            'Value': kpis.get('overall_sales_percent', 0),
                            'Unit': '%'
                        }, {
                            'Metric': 'NOB Achievement',
                            'Value': kpis.get('overall_nob_percent', 0), 
                            'Unit': '%'
                        }, {
                            'Metric': 'ABV Achievement',
                            'Value': kpis.get('overall_abv_percent', 0),
                            'Unit': '%'
                        }, {
                            'Metric': 'Performance Score',
                            'Value': kpis.get('performance_score', 0),
                            'Unit': '/100'
                        }])
                        kpi_summary.to_excel(writer, sheet_name="KPI_Summary", index=False)
                    
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    st.download_button(
                        "📊 Download Excel Report",
                        excel_buffer.getvalue(),
                        f"Al_Khair_Analytics_{timestamp}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
                
                # CSV export
                csv_data = df_filtered.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📄 Download CSV Data",
                    csv_data,
                    f"Al_Khair_Data_{timestamp}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                
                # JSON export for API integration
                json_export = {
                    'metadata': {
                        'export_date': datetime.now().isoformat(),
                        'date_range': {
                            'start': st.session_state.start_date.isoformat(),
                            'end': st.session_state.end_date.isoformat()
                        },
                        'branches_included': selected_branches,
                        'total_records': len(df_filtered)
                    },
                    'kpis': {k: (v if not isinstance(v, (pd.DataFrame, pd.Series)) else v.to_dict()) 
                             for k, v in kpis.items() if k != 'alerts'},
                    'raw_data': df_filtered.to_dict('records')
                }
                
                json_data = pd.io.json.dumps(json_export, indent=2, default=str).encode('utf-8')
                st.download_button(
                    "🔗 Download JSON (API Format)",
                    json_data,
                    f"Al_Khair_API_{timestamp}.json",
                    mime="application/json",
                    use_container_width=True,
                )
            
            with col2:
                st.markdown("#### Export Summary")
                st.info(f"""
                **Data Range:** {st.session_state.start_date} to {st.session_state.end_date}  
                **Branches:** {len(selected_branches)} of {len(all_branches)}  
                **Records:** {len(df_filtered):,}  
                **Last Updated:** {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}
                """)
                
                if kpis.get("alerts"):
                    st.markdown("#### ⚠️ Active Alerts")
                    for alert in kpis["alerts"][:3]:  # Show max 3 alerts
                        alert_type = alert.get("type", "info")
                        emoji_map = {"critical": "🔴", "warning": "🟡", "info": "🔵"}
                        st.write(f"{emoji_map.get(alert_type, 'ℹ️')} {alert.get('title', 'Alert')}")
                
                # Data quality indicators
                st.markdown("#### 📊 Data Quality")
                
                quality_metrics = []
                if "Date" in df_filtered.columns:
                    missing_dates = df_filtered["Date"].isna().sum()
                    quality_metrics.append(f"Missing dates: {missing_dates}")
                
                for col in ["SalesActual", "NOBActual", "ABVActual"]:
                    if col in df_filtered.columns:
                        missing_values = df_filtered[col].isna().sum()
                        quality_metrics.append(f"Missing {col}: {missing_values}")
                
                for metric in quality_metrics:
                    st.text(f"• {metric}")

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"❌ Application error: {str(e)}")
        st.info("Please try refreshing the page or contact support if the issue persists.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        st.info("👋 Dashboard session ended.")
    except Exception as e:
        logger.critical(f"Critical application error: {str(e)}")
        st.error("❌ Critical error occurred. Please refresh the page.")
        st.exception(e)
