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
from plotly.subplots import make_subplots
import plotly.express as px

# =========================================
# CONFIG
# =========================================
class Config:
    PAGE_TITLE = "Al Khair Business Performance"
    LAYOUT = "wide"
    # Auto-refresh data source (no manual upload needed)
    AUTO_REFRESH_URL = (
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vQG7boLWl2bNLCPR05NXv6EFpPPcFfXsiXPQ7rAGYr3q8Nkc2Ijg8BqEwVofcMLSg/pubhtml"
    )
    BRANCHES = {
        "Al khair - 102": {"name": "Al Khair", "code": "102", "color": "#1e40af", "icon": "🏢"},
        "Noora - 104": {"name": "Noora", "code": "104", "color": "#059669", "icon": "🌟"},
        "Hamra - 109": {"name": "Hamra", "code": "109", "color": "#dc2626", "icon": "🏪"},
        "Magnus - 107": {"name": "Magnus", "code": "107", "color": "#7c3aed", "icon": "⭐"},
    }
    TARGETS = {"sales_achievement": 95.0, "nob_achievement": 90.0, "abv_achievement": 90.0}
    REFRESH_INTERVAL = 300  # Auto-refresh every 5 minutes

config = Config()
st.set_page_config(page_title=config.PAGE_TITLE, layout=config.LAYOUT, initial_sidebar_state="expanded")

# =========================================
# ENHANCED CSS
# =========================================
def apply_enhanced_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        html, body, [data-testid="stAppViewContainer"] { 
            font-family: 'Inter', sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        .main .block-container { 
            padding: 1rem 2rem; 
            max-width: 1600px; 
            background: rgba(255,255,255,0.95);
            border-radius: 20px;
            margin: 1rem auto;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        /* Header styling */
        .dashboard-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            margin-bottom: 2rem;
            text-align: center;
        }
        
        .dashboard-title {
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .dashboard-subtitle {
            font-size: 1.1rem;
            opacity: 0.9;
            font-weight: 300;
        }

        /* Enhanced metric cards */
        .metric-card {
            background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            border: 1px solid rgba(255,255,255,0.8);
            transition: all 0.3s ease;
            height: 100%;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0,0,0,0.15);
        }
        
        .metric-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
            text-align: center;
        }
        
        .metric-value {
            font-size: 2.2rem;
            font-weight: 800;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #64748b;
            text-align: center;
            font-weight: 500;
        }
        
        .metric-change {
            font-size: 0.85rem;
            font-weight: 600;
            text-align: center;
            margin-top: 0.5rem;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
        }
        
        .metric-change.positive {
            background: #ecfdf5;
            color: #059669;
        }
        
        .metric-change.negative {
            background: #fef2f2;
            color: #dc2626;
        }

        /* Branch filter cards */
        .branch-filter-container {
            display: flex;
            gap: 1rem;
            margin: 1rem 0;
            flex-wrap: wrap;
        }
        
        .branch-filter-card {
            background: white;
            border: 2px solid #e5e7eb;
            border-radius: 15px;
            padding: 1rem 1.5rem;
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: center;
            min-width: 120px;
        }
        
        .branch-filter-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .branch-filter-card.selected {
            border-color: #3b82f6;
            background: #eff6ff;
            box-shadow: 0 5px 15px rgba(59,130,246,0.2);
        }
        
        .branch-icon {
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
        }
        
        .branch-name {
            font-weight: 600;
            font-size: 0.9rem;
        }

        /* Enhanced branch performance cards */
        .branch-performance-card {
            background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
            border-radius: 20px;
            padding: 1.5rem;
            box-shadow: 0 8px 25px rgba(0,0,0,0.08);
            border-left: 5px solid;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
        }
        
        .branch-performance-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 30px rgba(0,0,0,0.12);
        }
        
        .performance-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }
        
        .performance-title {
            font-size: 1.3rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .performance-score {
            font-size: 1.1rem;
            font-weight: 800;
            padding: 0.4rem 1rem;
            border-radius: 25px;
        }
        
        .performance-metrics {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin-top: 1rem;
        }
        
        .performance-metric {
            text-align: center;
            padding: 0.8rem;
            background: rgba(255,255,255,0.7);
            border-radius: 12px;
        }
        
        .performance-metric-value {
            font-size: 1.4rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        
        .performance-metric-label {
            font-size: 0.8rem;
            color: #64748b;
            font-weight: 500;
        }

        /* Status indicators */
        .status-excellent { background: #ecfdf5; color: #059669; }
        .status-good { background: #eff6ff; color: #2563eb; }
        .status-warning { background: #fffbeb; color: #d97706; }
        .status-danger { background: #fef2f2; color: #dc2626; }

        /* Auto-refresh indicator */
        .refresh-indicator {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(59,130,246,0.9);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 25px;
            font-size: 0.8rem;
            font-weight: 600;
            z-index: 1000;
            backdrop-filter: blur(10px);
        }

        /* Hide streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )

# =========================================
# AUTO DATA LOADING
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
        d["BranchIcon"] = meta.get("icon", "🏪")
        d["BranchColor"] = meta.get("color", "#6b7280")
        
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
            df.groupby(["BranchName", "BranchIcon", "BranchColor"])
            .agg({
                "SalesTarget": "sum",
                "SalesActual": "sum",
                "SalesPercent": "mean",
                "NOBTarget": "sum",
                "NOBActual": "sum",
                "NOBPercent": "mean",
                "ABVTarget": "mean",
                "ABVActual": "mean",
                "ABVPercent": "mean",
            })
            .round(2)
        )

    if "Date" in df.columns and df["Date"].notna().any():
        k["date_range"] = {
            "start": df["Date"].min(), 
            "end": df["Date"].max(), 
            "days": int(df["Date"].dt.date.nunique())
        }

    # Performance score calculation
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
# ENHANCED VISUALIZATIONS
# =========================================
def create_enhanced_trend_chart(df: pd.DataFrame, metric_col: str, target_col: str, title: str) -> go.Figure:
    """Create enhanced daily trend chart with better styling"""
    if df.empty or "Date" not in df.columns:
        return go.Figure()
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=[title, "Performance %"],
        vertical_spacing=0.1,
        shared_xaxis=True
    )
    
    # Group by date and branch
    agg_func = "mean" if "ABV" in metric_col else "sum"
    daily_data = df.groupby(["Date", "BranchName", "BranchColor"]).agg({
        metric_col: agg_func,
        target_col: agg_func if target_col in df.columns else "sum"
    }).reset_index()
    
    # Add performance percentage
    if target_col in daily_data.columns:
        daily_data["Performance"] = (daily_data[metric_col] / daily_data[target_col] * 100).fillna(0)
    
    colors = px.colors.qualitative.Set1
    
    for i, branch in enumerate(sorted(daily_data["BranchName"].unique())):
        branch_data = daily_data[daily_data["BranchName"] == branch]
        color = branch_data["BranchColor"].iloc[0] if not branch_data.empty else colors[i % len(colors)]
        
        # Main trend line with area fill
        fig.add_trace(
            go.Scatter(
                x=branch_data["Date"],
                y=branch_data[metric_col],
                name=f"{branch} - Actual",
                mode="lines+markers",
                line=dict(width=3, color=color),
                fill="tonexty" if i > 0 else "tozeroy",
                fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(color)) + [0.1])}",
                hovertemplate=f"<b>{branch}</b><br>Date: %{{x|%Y-%m-%d}}<br>Actual: %{{y:,.0f}}<extra></extra>",
            ),
            row=1, col=1
        )
        
        # Target line
        if target_col in branch_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=branch_data["Date"],
                    y=branch_data[target_col],
                    name=f"{branch} - Target",
                    mode="lines",
                    line=dict(width=2, color=color, dash="dot"),
                    hovertemplate=f"<b>{branch}</b><br>Date: %{{x|%Y-%m-%d}}<br>Target: %{{y:,.0f}}<extra></extra>",
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Performance percentage
            fig.add_trace(
                go.Scatter(
                    x=branch_data["Date"],
                    y=branch_data["Performance"],
                    name=f"{branch} %",
                    mode="lines+markers",
                    line=dict(width=2, color=color),
                    hovertemplate=f"<b>{branch}</b><br>Date: %{{x|%Y-%m-%d}}<br>Performance: %{{y:.1f}}%<extra></extra>",
                    showlegend=False
                ),
                row=2, col=1
            )
    
    # Add target line for performance
    fig.add_hline(y=100, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    
    fig.update_layout(
        height=600,
        showlegend=True,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        hovermode="x unified"
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)")
    
    return fig

def create_branch_comparison_radar(bp: pd.DataFrame) -> go.Figure:
    """Create a radar chart for branch comparison"""
    if bp.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    metrics = ["SalesPercent", "NOBPercent", "ABVPercent"]
    metric_labels = ["Sales %", "NOB %", "ABV %"]
    
    for branch in bp.index:
        values = [bp.loc[branch, metric] for metric in metrics]
        values.append(values[0])  # Close the radar
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metric_labels + [metric_labels[0]],
            fill='toself',
            name=branch,
            hovertemplate=f"<b>{branch}</b><br>%{{theta}}: %{{r:.1f}}%<extra></extra>"
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 120]
            )),
        showlegend=True,
        title="Branch Performance Radar",
        height=400
    )
    
    return fig

# =========================================
# ENHANCED UI COMPONENTS
# =========================================
def render_dashboard_header():
    """Render enhanced dashboard header"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f"""
        <div class="dashboard-header">
            <div class="dashboard-title">🏢 Al Khair Business Performance</div>
            <div class="dashboard-subtitle">Real-time Business Analytics • Last Updated: {current_time}</div>
        </div>
        <div class="refresh-indicator">
            🔄 Auto-refresh: {config.REFRESH_INTERVAL//60}min
        </div>
    """, unsafe_allow_html=True)

def render_enhanced_branch_filter(df: pd.DataFrame) -> List[str]:
    """Render enhanced branch filter cards"""
    if "BranchName" not in df.columns:
        return []
    
    all_branches = sorted(df["BranchName"].unique())
    st.markdown("### 🏪 Select Branches")
    
    # Use session state to track selected branches
    if "selected_branches" not in st.session_state:
        st.session_state.selected_branches = all_branches
    
    cols = st.columns(len(all_branches))
    for i, branch in enumerate(all_branches):
        with cols[i]:
            branch_info = df[df["BranchName"] == branch].iloc[0]
            icon = branch_info.get("BranchIcon", "🏪")
            color = branch_info.get("BranchColor", "#6b7280")
            
            is_selected = branch in st.session_state.selected_branches
            
            if st.button(
                f"{icon}\n{branch}",
                key=f"branch_{branch}",
                help=f"Toggle {branch}",
                use_container_width=True
            ):
                if is_selected:
                    st.session_state.selected_branches.remove(branch)
                else:
                    st.session_state.selected_branches.append(branch)
                st.rerun()
    
    return st.session_state.selected_branches

def render_enhanced_kpi_cards(k: Dict[str, Any]):
    """Render enhanced KPI cards"""
    st.markdown("### 📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Sales Card
    with col1:
        sales_change = "positive" if k.get("total_sales_variance", 0) >= 0 else "negative"
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">💰</div>
                <div class="metric-value" style="color: #059669;">SAR {k.get('total_sales_actual',0):,.0f}</div>
                <div class="metric-label">Total Sales</div>
                <div class="metric-change {sales_change}">
                    {k.get('overall_sales_percent',0):.1f}% of Target
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # NOB Card
    with col2:
        nob_change = "positive" if k.get("overall_nob_percent", 0) >= 90 else "negative"
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">🛍️</div>
                <div class="metric-value" style="color: #2563eb;">{k.get('total_nob_actual',0):,.0f}</div>
                <div class="metric-label">Total Baskets</div>
                <div class="metric-change {nob_change}">
                    {k.get('overall_nob_percent',0):.1f}% Achievement
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # ABV Card
    with col3:
        abv_change = "positive" if k.get("overall_abv_percent", 0) >= 90 else "negative"
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">💎</div>
                <div class="metric-value" style="color: #7c3aed;">SAR {k.get('avg_abv_actual',0):,.2f}</div>
                <div class="metric-label">Avg Basket Value</div>
                <div class="metric-change {abv_change}">
                    {k.get('overall_abv_percent',0):.1f}% vs Target
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Performance Score Card
    with col4:
        score_change = "positive" if k.get("performance_score", 0) >= 80 else "negative"
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">⭐</div>
                <div class="metric-value" style="color: #dc2626;">{k.get('performance_score',0):.0f}/100</div>
                <div class="metric-label">Performance Score</div>
                <div class="metric-change {score_change}">
                    Weighted Average
                </div>
            </div>
        """, unsafe_allow_html=True)

def render_enhanced_branch_cards(bp: pd.DataFrame):
    """Render enhanced branch performance cards"""
    if bp.empty:
        return
    
    st.markdown("### 🏪 Branch Performance")
    
    for branch in bp.index:
        branch_data = bp.loc[branch]
        icon = branch_data.name[1] if isinstance(branch_data.name, tuple) else "🏪"
        color = branch_data.name[2] if isinstance(branch_data.name, tuple) and len(branch_data.name) > 2 else "#6b7280"
        
        # Calculate overall performance
        avg_perf = np.mean([
            branch_data.get("SalesPercent", 0),
            branch_data.get("NOBPercent", 0),
            branch_data.get("ABVPercent", 0)
        ])
        
        # Determine status
        if avg_perf >= 95:
            status_class = "status-excellent"
        elif avg_perf >= 85:
            status_class = "status-good"
        elif avg_perf >= 75:
            status_class = "status-warning"
        else:
            status_class = "status-danger"
        
        st.markdown(f"""
            <div class="branch-performance-card" style="border-left-color: {color};">
                <div class="performance-header">
                    <div class="performance-title">
                        <span style="font-size: 1.5rem;">{icon}</span>
                        {branch if isinstance(branch, str) else branch[0]}
                    </div>
                    <div class="performance-score {status_class}">
                        {avg_perf:.1f}%
                    </div>
                </div>
                <div class="performance-metrics">
                    <div class="performance-metric">
                        <div class="performance-metric-value" style="color: #059669;">
                            {branch_data.get('SalesPercent', 0):.1f}%
                        </div>
                        <div class="performance-metric-label">Sales</div>
                    </div>
                    <div class="performance-metric">
                        <div class="performance-metric-value" style="color: #2563eb;">
                            {branch_data.get('NOBPercent', 0):.1f}%
                        </div>
                        <div class="performance-metric-label">Baskets</div>
                    </div>
                    <div class="performance-metric">
                        <div class="performance-metric-value" style="color: #7c3aed;">
                            {branch_data.get('ABVPercent', 0):.1f}%
                        </div>
                        <div class="performance-metric-label">Basket Value</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# =========================================
# MAIN APPLICATION
# =========================================
def main():
    apply_enhanced_css()
    
    # Auto-refresh indicator
    if st.button("🔄", help="Manual Refresh", key="manual_refresh"):
        auto_load_data.clear()
        st.rerun()
    
    # Render header
    render_dashboard_header()
    
    # Auto-load data (no manual upload needed)
    with st.spinner("🔄 Loading latest data..."):
        sheets_map = auto_load_data()
    
    if not sheets_map:
        st.error("❌ Unable to load data automatically. Please check your data source.")
        st.stop()
    
    # Process data
    df = process_branch_data(sheets_map)
    if df.empty:
        st.error("❌ No valid data found. Please check your data format.")
        st.stop()
    
    # Enhanced branch filter
    selected_branches = render_enhanced_branch_filter(df)
    if not selected_branches:
        st.warning("⚠️ Please select at least one branch.")
        st.stop()
    
    # Filter data by selected branches
    df_filtered = df[df["BranchName"].isin(selected_branches)].copy()
    
    # Date filter
    if "Date" in df_filtered.columns and df_filtered["Date"].notna().any():
        st.markdown("### 📅 Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date", 
                value=df_filtered["Date"].min().date(),
                key="start_date"
            )
        with col2:
            end_date = st.date_input(
                "End Date", 
                value=df_filtered["Date"].max().date(),
                key="end_date"
            )
        
        # Apply date filter
        mask = (df_filtered["Date"].dt.date >= start_date) & (df_filtered["Date"].dt.date <= end_date)
        df_filtered = df_filtered.loc[mask].copy()
    
    if df_filtered.empty:
        st.warning("⚠️ No data available for selected filters.")
        st.stop()
    
    # Calculate KPIs
    kpis = calc_kpis(df_filtered)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Trends", "🎯 Analysis", "📤 Export"])
    
    with tab1:
        # Enhanced KPI cards
        render_enhanced_kpi_cards(kpis)
        
        st.markdown("---")
        
        # Enhanced branch cards
        if "branch_performance" in kpis and not kpis["branch_performance"].empty:
            render_enhanced_branch_cards(kpis["branch_performance"])
        
        # Quick insights
        with st.expander("💡 Quick Insights", expanded=True):
            insights = []
            if "branch_performance" in kpis and not kpis["branch_performance"].empty:
                bp = kpis["branch_performance"]
                best_sales = bp["SalesPercent"].idxmax()
                worst_sales = bp["SalesPercent"].idxmin()
                best_sales_name = best_sales[0] if isinstance(best_sales, tuple) else best_sales
                worst_sales_name = worst_sales[0] if isinstance(worst_sales, tuple) else worst_sales
                
                insights.extend([
                    f"🥇 **Top Performer**: {best_sales_name} with {bp.loc[best_sales, 'SalesPercent']:.1f}% sales achievement",
                    f"📉 **Needs Attention**: {worst_sales_name} with {bp.loc[worst_sales, 'SalesPercent']:.1f}% sales achievement",
                    f"💰 **Total Revenue**: SAR {kpis.get('total_sales_actual', 0):,.0f} ({kpis.get('overall_sales_percent', 0):.1f}% of target)",
                    f"🛍️ **Customer Traffic**: {kpis.get('total_nob_actual', 0):,.0f} baskets processed"
                ])
                
                if kpis.get('total_sales_variance', 0) < 0:
                    insights.append(f"⚠️ **Revenue Gap**: SAR {abs(kpis['total_sales_variance']):,.0f} below target")
            
            for insight in insights:
                st.markdown(insight)
    
    with tab2:
        st.markdown("### 📈 Enhanced Daily Trends")
        
        # Time period selector
        period_options = {
            "Last 7 Days": 7,
            "Last 14 Days": 14,
            "Last 30 Days": 30,
            "Last 90 Days": 90,
            "All Time": None
        }
        
        selected_period = st.selectbox("📅 Time Period", options=list(period_options.keys()), index=2)
        
        # Filter by time period
        if period_options[selected_period] and "Date" in df_filtered.columns:
            cutoff_date = df_filtered["Date"].max() - timedelta(days=period_options[selected_period])
            trend_df = df_filtered[df_filtered["Date"] >= cutoff_date].copy()
        else:
            trend_df = df_filtered.copy()
        
        # Create enhanced trend charts
        trend_tabs = st.tabs(["💰 Sales Trends", "🛍️ Basket Trends", "💎 Value Trends"])
        
        with trend_tabs[0]:
            if not trend_df.empty and "SalesActual" in trend_df.columns:
                fig_sales = create_enhanced_trend_chart(
                    trend_df, "SalesActual", "SalesTarget", "Daily Sales Performance"
                )
                st.plotly_chart(fig_sales, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("No sales data available for the selected period.")
        
        with trend_tabs[1]:
            if not trend_df.empty and "NOBActual" in trend_df.columns:
                fig_nob = create_enhanced_trend_chart(
                    trend_df, "NOBActual", "NOBTarget", "Daily Basket Count Performance"
                )
                st.plotly_chart(fig_nob, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("No basket data available for the selected period.")
        
        with trend_tabs[2]:
            if not trend_df.empty and "ABVActual" in trend_df.columns:
                fig_abv = create_enhanced_trend_chart(
                    trend_df, "ABVActual", "ABVTarget", "Average Basket Value Performance"
                )
                st.plotly_chart(fig_abv, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("No basket value data available for the selected period.")
    
    with tab3:
        st.markdown("### 🎯 Performance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if "branch_performance" in kpis and not kpis["branch_performance"].empty:
                # Create radar chart
                fig_radar = create_branch_comparison_radar(kpis["branch_performance"])
                st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})
        
        with col2:
            # Performance matrix
            st.markdown("#### Performance Matrix")
            if "branch_performance" in kpis and not kpis["branch_performance"].empty:
                bp = kpis["branch_performance"].copy()
                
                # Create performance matrix
                matrix_data = []
                for branch in bp.index:
                    branch_name = branch[0] if isinstance(branch, tuple) else branch
                    row = {
                        "Branch": branch_name,
                        "Sales %": bp.loc[branch, "SalesPercent"],
                        "NOB %": bp.loc[branch, "NOBPercent"],
                        "ABV %": bp.loc[branch, "ABVPercent"],
                        "Overall": np.mean([
                            bp.loc[branch, "SalesPercent"],
                            bp.loc[branch, "NOBPercent"],
                            bp.loc[branch, "ABVPercent"]
                        ])
                    }
                    matrix_data.append(row)
                
                matrix_df = pd.DataFrame(matrix_data)
                matrix_df = matrix_df.round(1)
                
                # Style the dataframe
                st.dataframe(
                    matrix_df.style.background_gradient(
                        subset=["Sales %", "NOB %", "ABV %", "Overall"],
                        cmap="RdYlGn",
                        vmin=70,
                        vmax=110
                    ).format({
                        "Sales %": "{:.1f}%",
                        "NOB %": "{:.1f}%",
                        "ABV %": "{:.1f}%",
                        "Overall": "{:.1f}%"
                    }),
                    use_container_width=True
                )
        
        # Trend analysis
        st.markdown("#### 📊 Trend Analysis")
        if "Date" in df_filtered.columns and len(df_filtered) > 1:
            # Calculate weekly trends
            df_filtered["Week"] = df_filtered["Date"].dt.isocalendar().week
            weekly_trends = df_filtered.groupby(["Week", "BranchName"]).agg({
                "SalesActual": "sum",
                "NOBActual": "sum",
                "ABVActual": "mean"
            }).reset_index()
            
            # Show trend direction
            trend_summary = []
            for branch in selected_branches:
                branch_data = weekly_trends[weekly_trends["BranchName"] == branch].sort_values("Week")
                if len(branch_data) >= 2:
                    sales_trend = "📈" if branch_data["SalesActual"].iloc[-1] > branch_data["SalesActual"].iloc[0] else "📉"
                    nob_trend = "📈" if branch_data["NOBActual"].iloc[-1] > branch_data["NOBActual"].iloc[0] else "📉"
                    abv_trend = "📈" if branch_data["ABVActual"].iloc[-1] > branch_data["ABVActual"].iloc[0] else "📉"
                    
                    trend_summary.append({
                        "Branch": branch,
                        "Sales Trend": sales_trend,
                        "Baskets Trend": nob_trend,
                        "Value Trend": abv_trend
                    })
            
            if trend_summary:
                trend_df = pd.DataFrame(trend_summary)
                st.dataframe(trend_df, use_container_width=True)
    
    with tab4:
        st.markdown("### 📤 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Excel export
            if not df_filtered.empty:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                    df_filtered.to_excel(writer, sheet_name="Raw Data", index=False)
                    if "branch_performance" in kpis and not kpis["branch_performance"].empty:
                        kpis["branch_performance"].to_excel(writer, sheet_name="Branch Summary")
                
                st.download_button(
                    "📊 Download Excel Report",
                    data=buffer.getvalue(),
                    file_name=f"AlKhair_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        
        with col2:
            # CSV export
            if not df_filtered.empty:
                csv_data = df_filtered.to_csv(index=False)
                st.download_button(
                    "📄 Download CSV Data",
                    data=csv_data,
                    file_name=f"AlKhair_Data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        # Data summary
        st.markdown("#### 📋 Data Summary")
        summary_data = {
            "Total Records": len(df_filtered),
            "Date Range": f"{df_filtered['Date'].min().date()} to {df_filtered['Date'].max().date()}" if "Date" in df_filtered.columns else "N/A",
            "Branches": len(selected_branches),
            "Total Sales": f"SAR {kpis.get('total_sales_actual', 0):,.0f}",
            "Total Baskets": f"{kpis.get('total_nob_actual', 0):,.0f}",
            "Avg Basket Value": f"SAR {kpis.get('avg_abv_actual', 0):,.2f}"
        }
        
        for key, value in summary_data.items():
            st.metric(key, value)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.info("Please refresh the page or contact support if the issue persists.")
