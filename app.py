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
import plotly.express as px

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
    REFRESH_INTERVAL = 300

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
        
        /* Aggressive removal of ALL possible spacing */
        * {
            margin: 0 !important;
            padding: 0 !important;
            box-sizing: border-box !important;
        }
        
        /* Re-apply necessary spacing only where needed */
        .kpi-card, .performance-card, .branch-card {
            margin: 0 !important;
            padding: 2rem !important;
        }
        
        .sidebar-section {
            margin-bottom: 2rem !important;
        }
        
        .sidebar-stat {
            padding: 0.75rem 0 !important;
        }
        
        .performance-metrics {
            gap: 1.5rem !important;
        }
        
        .kpi-grid, .performance-grid, .branch-grid {
            gap: 2rem !important;
            margin: 2rem 0 !important;
        }
        
        html, body {
            margin: 0 !important;
            padding: 0 !important;
            height: 100% !important;
        }
        
        [data-testid="stAppViewContainer"] { 
            font-family: 'Poppins', sans-serif !important;
            background: #ffffff !important;
            color: #1a202c;
            margin: 0 !important;
            padding: 0 !important;
        }
        
        .stApp {
            margin: 0 !important;
            padding: 0 !important;
        }
        
        .stApp > header {
            display: none !important;
        }
        
        .main {
            padding: 0 !important;
            margin: 0 !important;
        }
        
        .main .block-container { 
            padding: 0 !important;
            max-width: 100% !important;
            margin: 0 !important;
            margin-top: -200px !important;
            padding-top: 200px !important;
        }
        
        /* Dashboard Layout - Move up aggressively */
        .dashboard-layout {
            display: flex;
            min-height: 100vh;
            margin: 0;
            padding: 0;
            margin-top: -200px;
            padding-top: 200px;
        }
        
        /* Left Sidebar */
        .sidebar {
            width: 280px;
            background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
            border-right: 1px solid #e2e8f0;
            padding: 1rem 1.5rem;
            position: fixed;
            height: 100vh;
            overflow-y: auto;
            z-index: 100;
            top: 0;
            left: 0;
        }
        
        .sidebar-logo {
            text-align: center;
            margin-bottom: 2rem;
            padding: 1rem;
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            color: white;
            border-radius: 15px;
            font-size: 1.2rem;
            font-weight: 700;
        }
        
        .sidebar-section {
            margin-bottom: 2rem;
        }
        
        .sidebar-title {
            font-size: 0.9rem;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 1rem;
        }
        
        .sidebar-nav {
            list-style: none;
            padding: 0;
        }
        
        .sidebar-nav-item {
            margin-bottom: 0.5rem;
        }
        
        .sidebar-nav-link {
            display: flex;
            align-items: center;
            padding: 0.75rem 1rem;
            color: #475569;
            text-decoration: none;
            border-radius: 10px;
            transition: all 0.2s ease;
            font-weight: 500;
        }
        
        .sidebar-nav-link:hover {
            background: rgba(99,102,241,0.1);
            color: #6366f1;
        }
        
        .sidebar-nav-link.active {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            color: white;
        }
        
        .sidebar-nav-icon {
            margin-right: 0.75rem;
            font-size: 1.1rem;
        }
        
        .sidebar-stats {
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        
        .sidebar-stat {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 0;
            border-bottom: 1px solid #f1f5f9;
        }
        
        .sidebar-stat:last-child {
            border-bottom: none;
        }
        
        .sidebar-stat-label {
            font-size: 0.85rem;
            color: #64748b;
            font-weight: 500;
        }
        
        .sidebar-stat-value {
            font-weight: 700;
            color: #1e293b;
        }
        
        /* Main Content */
        .main-content {
            margin-left: 280px;
            padding: 1rem 2rem;
            flex: 1;
            width: calc(100% - 280px);
        }
        
        /* Header */
        .modern-header {
            text-align: center;
            padding: 1.5rem 0;
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            border-radius: 15px;
            color: white;
            margin-bottom: 1.5rem;
            position: relative;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(99,102,241,0.3);
        }
        
        .modern-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="50" r="2" fill="white" opacity="0.1"/></svg>');
            background-size: 30px 30px;
        }
        
        .header-title {
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            position: relative;
            z-index: 1;
        }
        
        .header-subtitle {
            font-size: 1.1rem;
            opacity: 0.9;
            font-weight: 300;
            position: relative;
            z-index: 1;
        }
        
        /* Branch Filter Cards - Light colors */
        .branch-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }
        
        .branch-card {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            border-radius: 15px;
            padding: 1.5rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            border: 2px solid #e5e7eb;
            position: relative;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        
        .branch-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: var(--branch-color);
            transform: scaleX(0);
            transition: transform 0.3s ease;
        }
        
        .branch-card:hover::before {
            transform: scaleX(1);
        }
        
        .branch-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.15);
            border-color: var(--branch-color);
        }
        
        .branch-card.selected {
            border-color: var(--branch-color);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        }
        
        .branch-card.selected::before {
            transform: scaleX(1);
        }
        
        .branch-icon {
            font-size: 2.5rem;
            margin-bottom: 1rem;
            display: block;
        }
        
        .branch-name {
            font-weight: 600;
            font-size: 1.1rem;
            color: #1a202c;
        }
        
        /* KPI Cards - Colored backgrounds */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 2rem;
            margin: 2rem 0;
        }
        
        .kpi-card {
            border-radius: 20px;
            padding: 2rem;
            text-align: center;
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
            border: 1px solid rgba(255,255,255,0.3);
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }
        
        .kpi-card.sales {
            background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
            border-color: #10b981;
        }
        
        .kpi-card.baskets {
            background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
            border-color: #3b82f6;
        }
        
        .kpi-card.value {
            background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%);
            border-color: #8b5cf6;
        }
        
        .kpi-card.score {
            background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
            border-color: #ef4444;
        }
        
        .kpi-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: radial-gradient(circle at top right, var(--kpi-color, #6366f1) 0%, transparent 50%);
            opacity: 0.05;
            z-index: 0;
        }
        
        .kpi-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.15);
        }
        
        .kpi-content {
            position: relative;
            z-index: 1;
        }
        
        .kpi-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
            opacity: 0.8;
        }
        
        .kpi-value {
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
            color: var(--kpi-color, #1a202c);
        }
        
        .kpi-label {
            font-size: 1rem;
            color: #64748b;
            font-weight: 500;
            margin-bottom: 1rem;
        }
        
        .kpi-change {
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 25px;
            font-size: 0.9rem;
            font-weight: 600;
        }
        
        .kpi-change.positive {
            background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
            color: #065f46;
        }
        
        .kpi-change.negative {
            background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
            color: #991b1b;
        }
        
        /* Performance Cards - Light colored backgrounds */
        .performance-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            margin: 2rem 0;
        }
        
        .performance-card {
            border-radius: 20px;
            padding: 2rem;
            position: relative;
            overflow: hidden;
            border-left: 6px solid var(--performance-color);
            transition: all 0.3s ease;
            border: 1px solid rgba(255,255,255,0.3);
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }
        
        .performance-card.status-excellent {
            background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        }
        
        .performance-card.status-good {
            background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        }
        
        .performance-card.status-warning {
            background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        }
        
        .performance-card.status-danger {
            background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
        }
        
        .performance-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.15);
        }
        
        .performance-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
        }
        
        .performance-title {
            font-size: 1.4rem;
            font-weight: 700;
            color: #1a202c;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        
        .performance-score {
            font-size: 1.2rem;
            font-weight: 800;
            padding: 0.75rem 1.5rem;
            border-radius: 30px;
            color: white;
            background: var(--performance-color);
        }
        
        .performance-metrics {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.5rem;
        }
        
        .performance-metric {
            text-align: center;
            padding: 1rem;
            background: rgba(255,255,255,0.8);
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        
        .performance-metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            color: var(--performance-color);
        }
        
        .performance-metric-label {
            font-size: 0.9rem;
            color: #64748b;
            font-weight: 500;
        }
        
        /* Status Colors */
        .status-excellent { --performance-color: #10b981; }
        .status-good { --performance-color: #6366f1; }
        .status-warning { --performance-color: #f59e0b; }
        .status-danger { --performance-color: #ef4444; }
        
        /* Insights */
        .insights-container {
            background: #f8fafc;
            border-radius: 15px;
            padding: 2rem;
            margin: 2rem 0;
            border: 1px solid #e5e7eb;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        
        .insight-item {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 1rem;
            margin: 0.5rem 0;
            background: rgba(255,255,255,0.7);
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        
        /* Streamlit Overrides */
        .stTabs [data-baseweb="tab-list"] {
            background: #f8fafc;
            border-radius: 12px;
            padding: 0.5rem;
            gap: 0.5rem;
            border: 1px solid #e5e7eb;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 12px !important;
            padding: 1rem 2rem !important;
            font-weight: 600 !important;
            border: none !important;
            background: transparent !important;
            transition: all 0.3s ease !important;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
            color: white !important;
            box-shadow: 0 4px 12px rgba(99,102,241,0.3) !important;
        }
        
        /* Date inputs */
        .stDateInput > div > div > input {
            border-radius: 12px !important;
            border: 2px solid #e2e8f0 !important;
            padding: 0.75rem !important;
            transition: all 0.3s ease !important;
        }
        
        .stDateInput > div > div > input:focus {
            border-color: #6366f1 !important;
            box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
        }
        
        /* Selectbox */
        .stSelectbox > div > div {
            border-radius: 12px !important;
            border: 2px solid #e2e8f0 !important;
            transition: all 0.3s ease !important;
        }
        
        .stSelectbox > div > div:focus-within {
            border-color: #6366f1 !important;
            box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
        }
        
        /* Refresh button */
        .refresh-btn {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            color: white;
            border: none;
            border-radius: 50px;
            padding: 1rem;
            font-size: 1.2rem;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(99,102,241,0.3);
            transition: all 0.3s ease;
        }
        
        .refresh-btn:hover {
            transform: scale(1.05);
            box-shadow: 0 6px 20px rgba(99,102,241,0.4);
        }
        
        /* Auto refresh indicator */
        .auto-refresh {
            position: fixed;
            top: 20px;
            left: 20px;
            background: rgba(16,185,129,0.9);
            color: white;
            padding: 0.75rem 1.5rem;
            border-radius: 25px;
            font-size: 0.9rem;
            font-weight: 600;
            z-index: 1000;
            box-shadow: 0 4px 12px rgba(16,185,129,0.3);
        }
        
        /* Hide Streamlit elements completely */
        #MainMenu {display: none !important;}
        footer {display: none !important;}
        .stDeployButton {display: none !important;}
        header[data-testid="stHeader"] {display: none !important;}
        [data-testid="stToolbar"] {display: none !important;}
        [data-testid="stDecoration"] {display: none !important;}
        .stActionButton {display: none !important;}
        
        /* Force remove all top spacing */
        [data-testid="stAppViewContainer"] {
            padding-top: 0 !important;
            margin-top: 0 !important;
            position: relative !important;
            top: 0 !important;
        }
        
        [data-testid="stAppViewContainer"] > .main {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }
        
        [data-testid="stSidebar"] {
            display: none !important;
        }
        
        /* Force the entire app to start from top */
        * {
            margin-top: 0 !important;
        }
        
        *:first-child {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
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
# CHART FUNCTIONS
# =========================================
def create_modern_trend_chart(df: pd.DataFrame, metric_col: str, target_col: str, title: str) -> go.Figure:
    """Create modern trend chart"""
    if df.empty or "Date" not in df.columns:
        return go.Figure()
    
    agg_func = "mean" if "ABV" in metric_col else "sum"
    daily_data = df.groupby(["Date", "BranchName", "BranchColor"]).agg({
        metric_col: agg_func,
        target_col: agg_func if target_col in df.columns else "sum"
    }).reset_index()
    
    fig = go.Figure()
    
    for branch in sorted(daily_data["BranchName"].unique()):
        branch_data = daily_data[daily_data["BranchName"] == branch].sort_values("Date")
        if branch_data.empty:
            continue
            
        color = branch_data["BranchColor"].iloc[0]
        
        # Actual values
        fig.add_trace(
            go.Scatter(
                x=branch_data["Date"],
                y=branch_data[metric_col],
                name=f"{branch}",
                mode="lines+markers",
                line=dict(width=4, color=color),
                marker=dict(size=8, color=color),
                fill="tonexty" if len(fig.data) > 0 else "tozeroy",
                fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)",
                hovertemplate=f"<b>{branch}</b><br>Date: %{{x}}<br>Actual: %{{y:,.0f}}<extra></extra>",
            )
        )
        
        # Target line
        if target_col in branch_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=branch_data["Date"],
                    y=branch_data[target_col],
                    name=f"{branch} Target",
                    mode="lines",
                    line=dict(width=2, color=color, dash="dot"),
                    showlegend=False,
                    hovertemplate=f"<b>{branch} Target</b><br>Date: %{{x}}<br>Target: %{{y:,.0f}}<extra></extra>",
                )
            )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, weight=700)),
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
    """Create performance percentage chart"""
    if df.empty or "Date" not in df.columns or target_col not in df.columns:
        return go.Figure()
    
    agg_func = "mean" if "ABV" in metric_col else "sum"
    daily_data = df.groupby(["Date", "BranchName", "BranchColor"]).agg({
        metric_col: agg_func,
        target_col: agg_func
    }).reset_index()
    
    daily_data["Performance"] = (daily_data[metric_col] / daily_data[target_col] * 100).fillna(0)
    
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
        title=dict(text=f"{title} Performance %", font=dict(size=18, weight=600)),
        height=300,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)"),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)", title="Performance %"),
    )
    
    return fig

def create_radar_chart(bp: pd.DataFrame) -> go.Figure:
    """Fixed radar chart"""
    if bp.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    metrics = ["SalesPercent", "NOBPercent", "ABVPercent"]
    metric_labels = ["Sales %", "NOB %", "ABV %"]
    colors = ["#6366f1", "#10b981", "#ef4444", "#8b5cf6"]
    
    for i, branch_name in enumerate(bp.index):
        values = [bp.loc[branch_name, metric] for metric in metrics]
        values.append(values[0])  # Close the radar
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metric_labels + [metric_labels[0]],
            fill='toself',
            name=str(branch_name),  # Convert to string to fix the tuple issue
            line_color=colors[i % len(colors)],
            hovertemplate=f"<b>{branch_name}</b><br>%{{theta}}: %{{r:.1f}}%<extra></extra>"
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 120],
                showgrid=True,
                gridcolor="rgba(0,0,0,0.1)"
            )),
        showlegend=True,
        title=dict(text="Branch Performance Comparison", font=dict(size=18, weight=600)),
        height=400,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig

# =========================================
# UI COMPONENTS
# =========================================
def render_sidebar(kpis: Dict[str, Any], df: pd.DataFrame):
    """Render the left sidebar"""
    st.markdown("""
        <div class="sidebar">
            <div class="sidebar-logo">
                🏢 Al Khair Dashboard
            </div>
            
            <div class="sidebar-section">
                <div class="sidebar-title">Navigation</div>
                <ul class="sidebar-nav">
                    <li class="sidebar-nav-item">
                        <a href="#overview" class="sidebar-nav-link active">
                            <span class="sidebar-nav-icon">📊</span>
                            Overview
                        </a>
                    </li>
                    <li class="sidebar-nav-item">
                        <a href="#trends" class="sidebar-nav-link">
                            <span class="sidebar-nav-icon">📈</span>
                            Trends
                        </a>
                    </li>
                    <li class="sidebar-nav-item">
                        <a href="#analysis" class="sidebar-nav-link">
                            <span class="sidebar-nav-icon">🎯</span>
                            Analysis
                        </a>
                    </li>
                    <li class="sidebar-nav-item">
                        <a href="#export" class="sidebar-nav-link">
                            <span class="sidebar-nav-icon">📤</span>
                            Export
                        </a>
                    </li>
                </ul>
            </div>
            
            <div class="sidebar-section">
                <div class="sidebar-title">Quick Stats</div>
                <div class="sidebar-stats">
                    <div class="sidebar-stat">
                        <span class="sidebar-stat-label">Total Branches</span>
                        <span class="sidebar-stat-value">{}</span>
                    </div>
                    <div class="sidebar-stat">
                        <span class="sidebar-stat-label">Active Days</span>
                        <span class="sidebar-stat-value">{}</span>
                    </div>
                    <div class="sidebar-stat">
                        <span class="sidebar-stat-label">Sales Achievement</span>
                        <span class="sidebar-stat-value">{:.1f}%</span>
                    </div>
                    <div class="sidebar-stat">
                        <span class="sidebar-stat-label">Total Records</span>
                        <span class="sidebar-stat-value">{:,}</span>
                    </div>
                </div>
            </div>
            
            <div class="sidebar-section">
                <div class="sidebar-title">System Status</div>
                <div class="sidebar-stats">
                    <div class="sidebar-stat">
                        <span class="sidebar-stat-label">Data Status</span>
                        <span class="sidebar-stat-value" style="color: #10b981;">● Live</span>
                    </div>
                    <div class="sidebar-stat">
                        <span class="sidebar-stat-label">Last Update</span>
                        <span class="sidebar-stat-value">{}</span>
                    </div>
                    <div class="sidebar-stat">
                        <span class="sidebar-stat-label">Next Refresh</span>
                        <span class="sidebar-stat-value">{}min</span>
                    </div>
                </div>
            </div>
        </div>
    """.format(
        df["BranchName"].nunique() if "BranchName" in df.columns else 0,
        kpis.get("date_range", {}).get("days", 0),
        kpis.get("overall_sales_percent", 0),
        len(df),
        datetime.now().strftime("%H:%M"),
        config.REFRESH_INTERVAL // 60
    ), unsafe_allow_html=True)

def render_header():
    """Render modern header"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f"""
        <div class="modern-header">
            <div class="header-title">🏢 Al Khair Business Performance</div>
            <div class="header-subtitle">Real-time Business Analytics • Last Updated: {current_time}</div>
        </div>
        <div class="auto-refresh">
            🔄 Auto-refresh: {config.REFRESH_INTERVAL//60}min
        </div>
    """, unsafe_allow_html=True)

def render_branch_filter(df: pd.DataFrame) -> List[str]:
    """Render modern branch filter"""
    if "BranchName" not in df.columns:
        return []
    
    st.markdown("### 🏪 Select Branches")
    
    if "selected_branches" not in st.session_state:
        st.session_state.selected_branches = list(df["BranchName"].unique())
    
    branches_data = df[["BranchName", "BranchIcon", "BranchColor"]].drop_duplicates()
    
    cols = st.columns(len(branches_data))
    for i, (_, row) in enumerate(branches_data.iterrows()):
        with cols[i]:
            branch_name = row["BranchName"]
            icon = row["BranchIcon"]
            color = row["BranchColor"]
            is_selected = branch_name in st.session_state.selected_branches
            
            if st.button(
                f"{icon}\n{branch_name}",
                key=f"branch_{i}",
                help=f"Toggle {branch_name}",
                use_container_width=True,
                type="primary" if is_selected else "secondary"
            ):
                if is_selected:
                    st.session_state.selected_branches.remove(branch_name)
                else:
                    st.session_state.selected_branches.append(branch_name)
                st.rerun()
    
    return st.session_state.selected_branches

def render_kpi_cards(kpis: Dict[str, Any]):
    """Render modern KPI cards with light colors"""
    st.markdown("### 📊 Key Performance Indicators")
    
    kpi_data = [
        {
            "icon": "💰",
            "value": f"SAR {kpis.get('total_sales_actual', 0):,.0f}",
            "label": "Total Sales",
            "change": f"{kpis.get('overall_sales_percent', 0):.1f}% of Target",
            "color": "#10b981",
            "positive": kpis.get("total_sales_variance", 0) >= 0,
            "class": "sales"
        },
        {
            "icon": "🛍️",
            "value": f"{kpis.get('total_nob_actual', 0):,.0f}",
            "label": "Total Baskets",
            "change": f"{kpis.get('overall_nob_percent', 0):.1f}% Achievement",
            "color": "#6366f1",
            "positive": kpis.get("overall_nob_percent", 0) >= 90,
            "class": "baskets"
        },
        {
            "icon": "💎",
            "value": f"SAR {kpis.get('avg_abv_actual', 0):,.2f}",
            "label": "Avg Basket Value",
            "change": f"{kpis.get('overall_abv_percent', 0):.1f}% vs Target",
            "color": "#8b5cf6",
            "positive": kpis.get("overall_abv_percent", 0) >= 90,
            "class": "value"
        },
        {
            "icon": "⭐",
            "value": f"{kpis.get('performance_score', 0):.0f}/100",
            "label": "Performance Score",
            "change": "Weighted Average",
            "color": "#ef4444",
            "positive": kpis.get("performance_score", 0) >= 80,
            "class": "score"
        }
    ]
    
    cols = st.columns(len(kpi_data))
    for i, (col, kpi) in enumerate(zip(cols, kpi_data)):
        with col:
            change_class = "positive" if kpi["positive"] else "negative"
            st.markdown(f"""
                <div class="kpi-card {kpi['class']}" style="--kpi-color: {kpi['color']}">
                    <div class="kpi-content">
                        <div class="kpi-icon">{kpi['icon']}</div>
                        <div class="kpi-value">{kpi['value']}</div>
                        <div class="kpi-label">{kpi['label']}</div>
                        <div class="kpi-change {change_class}">{kpi['change']}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

def render_performance_cards(bp: pd.DataFrame):
    """Render modern performance cards"""
    if bp.empty:
        return
    
    st.markdown("### 🏪 Branch Performance")
    
    # Get branch metadata
    branch_colors = {"Al Khair": "#6366f1", "Noora": "#10b981", "Hamra": "#ef4444", "Magnus": "#8b5cf6"}
    branch_icons = {"Al Khair": "🏢", "Noora": "🌟", "Hamra": "🏪", "Magnus": "⭐"}
    
    cols = st.columns(min(len(bp), 2))
    for i, branch_name in enumerate(bp.index):
        with cols[i % 2]:
            branch_data = bp.loc[branch_name]
            icon = branch_icons.get(branch_name, "🏪")
            color = branch_colors.get(branch_name, "#6366f1")
            
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
                <div class="performance-card {status_class}" style="--performance-color: {color}">
                    <div class="performance-header">
                        <div class="performance-title">
                            <span>{icon}</span>
                            {branch_name}
                        </div>
                        <div class="performance-score">
                            {avg_perf:.1f}%
                        </div>
                    </div>
                    <div class="performance-metrics">
                        <div class="performance-metric">
                            <div class="performance-metric-value">
                                {branch_data.get('SalesPercent', 0):.1f}%
                            </div>
                            <div class="performance-metric-label">Sales</div>
                        </div>
                        <div class="performance-metric">
                            <div class="performance-metric-value">
                                {branch_data.get('NOBPercent', 0):.1f}%
                            </div>
                            <div class="performance-metric-label">Baskets</div>
                        </div>
                        <div class="performance-metric">
                            <div class="performance-metric-value">
                                {branch_data.get('ABVPercent', 0):.1f}%
                            </div>
                            <div class="performance-metric-label">Basket Value</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

def render_insights(kpis: Dict[str, Any]):
    """Render insights section"""
    insights = []
    if "branch_performance" in kpis and not kpis["branch_performance"].empty:
        bp = kpis["branch_performance"]
        best_sales = bp["SalesPercent"].idxmax()
        worst_sales = bp["SalesPercent"].idxmin()
        
        insights.extend([
            f"🥇 **Top Performer**: {best_sales} with {bp.loc[best_sales, 'SalesPercent']:.1f}% sales achievement",
            f"📉 **Needs Attention**: {worst_sales} with {bp.loc[worst_sales, 'SalesPercent']:.1f}% sales achievement",
            f"💰 **Total Revenue**: SAR {kpis.get('total_sales_actual', 0):,.0f} ({kpis.get('overall_sales_percent', 0):.1f}% of target)",
            f"🛍️ **Customer Traffic**: {kpis.get('total_nob_actual', 0):,.0f} baskets processed"
        ])
        
        if kpis.get('total_sales_variance', 0) < 0:
            insights.append(f"⚠️ **Revenue Gap**: SAR {abs(kpis['total_sales_variance']):,.0f} below target")
    
    if insights:
        st.markdown("""
            <div class="insights-container">
                <h3 style="margin-bottom: 1rem; color: #1a202c;">💡 Key Insights</h3>
        """, unsafe_allow_html=True)
        
        for insight in insights:
            st.markdown(f"""
                <div class="insight-item">
                    {insight}
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# =========================================
# MAIN APPLICATION
# =========================================
def main():
    apply_modern_css()
    
    # Load data first
    with st.spinner("🔄 Loading latest data..."):
        sheets_map = auto_load_data()
    
    if not sheets_map:
        st.error("❌ Unable to load data. Please check your data source.")
        st.stop()
    
    # Process data
    df = process_branch_data(sheets_map)
    if df.empty:
        st.error("❌ No valid data found.")
        st.stop()
    
    # Calculate KPIs early for sidebar
    kpis = calc_kpis(df)
    
    # Layout with sidebar
    st.markdown('<div class="dashboard-layout">', unsafe_allow_html=True)
    
    # Render sidebar
    render_sidebar(kpis, df)
    
    # Main content area
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    # Manual refresh button
    if st.button("🔄", help="Manual Refresh", key="refresh", type="primary"):
        auto_load_data.clear()
        st.rerun()
    
    # Header
    render_header()
    
    # Branch filter
    selected_branches = render_branch_filter(df)
    if not selected_branches:
        st.warning("⚠️ Please select at least one branch.")
        st.stop()
    
    # Filter data
    df_filtered = df[df["BranchName"].isin(selected_branches)].copy()
    
    # Date filter
    if "Date" in df_filtered.columns and df_filtered["Date"].notna().any():
        st.markdown("### 📅 Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=df_filtered["Date"].min().date())
        with col2:
            end_date = st.date_input("End Date", value=df_filtered["Date"].max().date())
        
        mask = (df_filtered["Date"].dt.date >= start_date) & (df_filtered["Date"].dt.date <= end_date)
        df_filtered = df_filtered.loc[mask].copy()
    
    if df_filtered.empty:
        st.warning("⚠️ No data available for selected filters.")
        st.stop()
    
    # Recalculate KPIs for filtered data
    kpis_filtered = calc_kpis(df_filtered)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Trends", "🎯 Analysis", "📤 Export"])
    
    with tab1:
        render_kpi_cards(kpis_filtered)
        st.markdown("---")
        
        if "branch_performance" in kpis_filtered and not kpis_filtered["branch_performance"].empty:
            render_performance_cards(kpis_filtered["branch_performance"])
        
        render_insights(kpis_filtered)
    
    with tab2:
        st.markdown("### 📈 Daily Trends")
        
        period_options = {
            "Last 7 Days": 7,
            "Last 14 Days": 14,
            "Last 30 Days": 30,
            "Last 90 Days": 90,
            "All Time": None
        }
        
        selected_period = st.selectbox("📅 Time Period", options=list(period_options.keys()), index=2)
        
        if period_options[selected_period] and "Date" in df_filtered.columns:
            cutoff_date = df_filtered["Date"].max() - timedelta(days=period_options[selected_period])
            trend_df = df_filtered[df_filtered["Date"] >= cutoff_date].copy()
        else:
            trend_df = df_filtered.copy()
        
        trend_tabs = st.tabs(["💰 Sales", "🛍️ Baskets", "💎 Value"])
        
        with trend_tabs[0]:
            if not trend_df.empty and "SalesActual" in trend_df.columns:
                fig_sales = create_modern_trend_chart(trend_df, "SalesActual", "SalesTarget", "Daily Sales Performance")
                st.plotly_chart(fig_sales, use_container_width=True, config={"displayModeBar": False})
                
                if "SalesTarget" in trend_df.columns:
                    fig_sales_perf = create_performance_chart(trend_df, "SalesActual", "SalesTarget", "Sales")
                    st.plotly_chart(fig_sales_perf, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("No sales data available.")
        
        with trend_tabs[1]:
            if not trend_df.empty and "NOBActual" in trend_df.columns:
                fig_nob = create_modern_trend_chart(trend_df, "NOBActual", "NOBTarget", "Daily Basket Count")
                st.plotly_chart(fig_nob, use_container_width=True, config={"displayModeBar": False})
                
                if "NOBTarget" in trend_df.columns:
                    fig_nob_perf = create_performance_chart(trend_df, "NOBActual", "NOBTarget", "NOB")
                    st.plotly_chart(fig_nob_perf, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("No basket data available.")
        
        with trend_tabs[2]:
            if not trend_df.empty and "ABVActual" in trend_df.columns:
                fig_abv = create_modern_trend_chart(trend_df, "ABVActual", "ABVTarget", "Average Basket Value")
                st.plotly_chart(fig_abv, use_container_width=True, config={"displayModeBar": False})
                
                if "ABVTarget" in trend_df.columns:
                    fig_abv_perf = create_performance_chart(trend_df, "ABVActual", "ABVTarget", "ABV")
                    st.plotly_chart(fig_abv_perf, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("No basket value data available.")
    
    with tab3:
        st.markdown("### 🎯 Performance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if "branch_performance" in kpis_filtered and not kpis_filtered["branch_performance"].empty:
                fig_radar = create_radar_chart(kpis_filtered["branch_performance"])
                st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})
        
        with col2:
            st.markdown("#### Performance Matrix")
            if "branch_performance" in kpis_filtered and not kpis_filtered["branch_performance"].empty:
                bp = kpis_filtered["branch_performance"].copy()
                
                matrix_data = []
                for branch in bp.index:
                    row = {
                        "Branch": str(branch),
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
                
                matrix_df = pd.DataFrame(matrix_data).round(1)
                st.dataframe(
                    matrix_df.style.background_gradient(
                        subset=["Sales %", "NOB %", "ABV %", "Overall"],
                        cmap="RdYlGn",
                        vmin=70,
                        vmax=110
                    ),
                    use_container_width=True
                )
    
    with tab4:
        st.markdown("### 📤 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not df_filtered.empty:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                    df_filtered.to_excel(writer, sheet_name="Raw Data", index=False)
                    if "branch_performance" in kpis_filtered:
                        kpis_filtered["branch_performance"].to_excel(writer, sheet_name="Summary")
                
                st.download_button(
                    "📊 Download Excel Report",
                    data=buffer.getvalue(),
                    file_name=f"AlKhair_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        
        with col2:
            if not df_filtered.empty:
                csv_data = df_filtered.to_csv(index=False)
                st.download_button(
                    "📄 Download CSV Data",
                    data=csv_data,
                    file_name=f"AlKhair_Data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    # Close layout containers
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.info("Please refresh the page if the issue persists.")
