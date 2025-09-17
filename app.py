from __future__ import annotations

import io
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# =========================================
# ENHANCED CONFIG & DATA STRUCTURES
# =========================================
@dataclass
class BranchConfig:
    """Enhanced branch configuration"""
    name: str
    code: str
    color: str
    target_sales: Optional[float] = None
    target_nob: Optional[float] = None
    target_abv: Optional[float] = None

@dataclass
class Alert:
    """Performance alert structure"""
    level: str  # 'success', 'warning', 'error', 'info'
    message: str
    metric: Optional[str] = None
    value: Optional[float] = None

class Config:
    PAGE_TITLE = "Al Khair Business Performance"
    LAYOUT = "wide"
    DEFAULT_PUBLISHED_URL = (
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vROaePKJN3XhRor42BU4Kgd9fCAgW7W8vWJbwjveQWoJy8HCRBUAYh2s0AxGsBa3w/pub?output=xlsx"
    )
    
    BRANCHES = {
        "Al khair - 102": BranchConfig("Al Khair", "102", "#3b82f6"),
        "Noora - 104": BranchConfig("Noora", "104", "#10b981"),
        "Hamra - 109": BranchConfig("Hamra", "109", "#f59e0b"),
        "Magnus - 107": BranchConfig("Magnus", "107", "#8b5cf6"),
    }
    
    TARGETS = {
        "sales_achievement": 95.0,
        "nob_achievement": 90.0,
        "abv_achievement": 90.0,
        "performance_score": 80.0
    }
    
    # Performance thresholds for alerts
    ALERT_THRESHOLDS = {
        "critical_sales": 70.0,
        "warning_sales": 85.0,
        "critical_nob": 70.0,
        "warning_nob": 80.0,
        "liquidity_warning": 10000.0  # SAR
    }
    
    CARDS_PER_ROW = 3
    CACHE_TTL = 300  # 5 minutes

config = Config()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================================
# ENHANCED CSS WITH MOBILE RESPONSIVENESS
# =========================================
def apply_enhanced_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&display=swap');
        html, body, [data-testid="stAppViewContainer"] { font-family: 'Inter', sans-serif; }

        /* Enhanced main area with better spacing */
        .main .block-container { 
            padding: 1rem 1.2rem; 
            max-width: 1800px; 
            transition: all 0.3s ease;
        }

        /* HERO HEADER with gradient animation */
        .hero { text-align: center; margin: 0 0 20px 0; }
        .hero-bar { 
            width: min(1100px, 95%); 
            height: 18px; 
            margin: 8px auto 18px;
            border-radius: 14px; 
            background: linear-gradient(90deg, #3b82f6 0%, #4338ca 50%, #10b981 100%);
            background-size: 200% 100%;
            animation: gradientShift 3s ease infinite;
            box-shadow: 0 12px 30px rgba(67, 56, 202, .25);
        }
        
        @keyframes gradientShift {
            0%, 100% { background-position: 0% 0%; }
            50% { background-position: 100% 0%; }
        }
        
        .hero-title { 
            font-weight: 900; 
            font-size: clamp(24px, 3.6vw, 42px); 
            letter-spacing: .2px;
            color: #111827; 
            display: inline-flex; 
            align-items: center; 
            gap: .55rem;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .hero-emoji { font-size: 1.15em; animation: bounce 2s infinite; }
        
        @keyframes bounce {
            0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
            40% { transform: translateY(-10px); }
            60% { transform: translateY(-5px); }
        }

        /* Enhanced metric tiles */
        [data-testid="metric-container"] {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%) !important;
            border-radius: 16px !important;
            border: 1px solid rgba(0,0,0,.06) !important;
            padding: 18px !important;
            text-align: center !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
        }
        
        [data-testid="metric-container"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1) !important;
        }

        /* Alert styles */
        .alert-container {
            margin: 10px 0;
            padding: 12px 16px;
            border-radius: 8px;
            border-left: 4px solid;
            animation: slideIn 0.5s ease;
        }
        
        @keyframes slideIn {
            from { transform: translateX(-20px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        
        .alert-success { background: #ecfdf5; border-color: #10b981; color: #065f46; }
        .alert-warning { background: #fffbeb; border-color: #f59e0b; color: #92400e; }
        .alert-error { background: #fef2f2; border-color: #ef4444; color: #991b1b; }
        .alert-info { background: #eff6ff; border-color: #3b82f6; color: #1e40af; }

        /* Enhanced branch cards */
        .card { 
            background: linear-gradient(135deg, #fcfcfc 0%, #f8fafc 100%); 
            border: 1px solid #f1f5f9; 
            border-radius: 16px; 
            padding: 18px; 
            height: 100%;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: var(--branch-color, #3b82f6);
            border-radius: 16px 16px 0 0;
        }
        
        .card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 30px rgba(0,0,0,0.1);
        }

        /* Quick action buttons */
        .quick-actions {
            display: flex;
            gap: 8px;
            margin: 10px 0;
            flex-wrap: wrap;
        }
        
        .quick-btn {
            padding: 6px 12px;
            background: linear-gradient(135deg, #3b82f6, #1e40af);
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .quick-btn:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        }

        /* Mobile responsiveness */
        @media (max-width: 768px) {
            .main .block-container { padding: 0.5rem 1rem; }
            
            .metrics-row [data-testid="stHorizontalBlock"] {
                flex-direction: column !important;
                gap: 8px !important;
            }
            
            .hero-title {
                font-size: clamp(20px, 5vw, 28px) !important;
                flex-direction: column;
                gap: 0.3rem !important;
            }
            
            .branch-metrics {
                grid-template-columns: 1fr 1fr !important;
                gap: 8px !important;
            }
            
            [data-testid="stSidebar"] {
                width: 100% !important;
                min-width: 100% !important;
            }
        }

        /* Enhanced sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f7f9fc 0%, #e5e7eb 100%);
            border-right: 2px solid #cbd5e1;
            width: 260px;
            min-width: 240px;
            max-width: 280px;
        }

        /* Data quality indicators */
        .quality-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .quality-good { background: #10b981; }
        .quality-warning { background: #f59e0b; }
        .quality-error { background: #ef4444; }

        /* Loading spinner */
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #3b82f6;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Enhanced tabs */
        .stTabs [data-baseweb="tab-list"] { 
            gap: 12px; 
            border-bottom: none;
            background: #f8fafc;
            padding: 8px;
            border-radius: 12px;
        }
        
        .stTabs [data-baseweb="tab"] {
            position: relative;
            border-radius: 8px !important;
            padding: 12px 20px !important;
            font-weight: 700 !important;
            background: transparent !important;
            color: #64748b !important;
            border: none !important;
            transition: all 0.3s ease !important;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #ffffff, #f1f5f9) !important;
            color: #1e293b !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
            transform: translateY(-1px) !important;
        }

        /* Performance indicators */
        .perf-indicator {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        
        .perf-excellent { background: #ecfdf5; color: #059669; }
        .perf-good { background: #eff6ff; color: #2563eb; }
        .perf-warning { background: #fffbeb; color: #d97706; }
        .perf-critical { background: #fef2f2; color: #dc2626; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# =========================================
# ENHANCED DATA LOADING WITH BETTER ERROR HANDLING
# =========================================
def validate_google_sheets_url(url: str) -> bool:
    """Validate Google Sheets URL format"""
    if not url:
        return False
    patterns = [
        r'https://docs\.google\.com/spreadsheets/d/[\w-]+',
        r'https://drive\.google\.com/file/d/[\w-]+'
    ]
    return any(re.match(pattern, url.strip()) for pattern in patterns)

def _normalize_gsheet_url(url: str) -> str:
    """Enhanced URL normalization with validation"""
    u = (url or "").strip()
    
    if not validate_google_sheets_url(u):
        logger.warning(f"Invalid Google Sheets URL format: {u}")
        return config.DEFAULT_PUBLISHED_URL
    
    u = re.sub(r"/pubhtml(\?.*)?$", "/pub?output=xlsx", u)
    u = re.sub(r"/pub\?output=csv(\&.*)?$", "/pub?output=xlsx", u)
    return u or config.DEFAULT_PUBLISHED_URL

@st.cache_data(ttl=config.CACHE_TTL, show_spinner=False)
def load_workbook_with_enhanced_error_handling(published_url: str) -> Dict[str, pd.DataFrame]:
    """Enhanced data loading with comprehensive error handling"""
    url = _normalize_gsheet_url(published_url)
    
    try:
        with st.spinner("🔄 Loading data from Google Sheets..."):
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
        content_type = (response.headers.get("Content-Type") or "").lower()
        
        # Try Excel format first
        if "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" in content_type or "output=xlsx" in url:
            try:
                xls = pd.ExcelFile(io.BytesIO(response.content))
                sheets = {}
                
                for sheet_name in xls.sheet_names:
                    try:
                        df = xls.parse(sheet_name).dropna(how="all").dropna(axis=1, how="all")
                        if not df.empty:
                            sheets[sheet_name] = df
                            logger.info(f"Successfully loaded sheet: {sheet_name} ({len(df)} rows)")
                    except Exception as e:
                        logger.warning(f"Failed to parse sheet {sheet_name}: {e}")
                        continue
                
                if sheets:
                    st.success(f"✅ Successfully loaded {len(sheets)} sheets")
                    return sheets
                    
            except Exception as e:
                logger.error(f"Excel parsing failed: {e}")
                
        # Fallback to CSV
        try:
            df = pd.read_csv(io.BytesIO(response.content))
        except Exception:
            df = pd.read_csv(io.BytesIO(response.content), sep=";")
            
        df = df.dropna(how="all").dropna(axis=1, how="all")
        
        if not df.empty:
            st.info("📄 Loaded data as CSV format")
            return {"Sheet1": df}
        else:
            st.warning("📋 The loaded data appears to be empty")
            return {}
            
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. Please check your internet connection and try again.")
        return {}
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            st.error("📄 Spreadsheet not found. Please verify the URL is correct and the sheet is published.")
        elif e.response.status_code == 403:
            st.error("🔒 Access denied. Please ensure the spreadsheet is publicly viewable.")
        else:
            st.error(f"🌐 HTTP Error {e.response.status_code}: {e.response.reason}")
        return {}
    except pd.errors.EmptyDataError:
        st.warning("📋 The spreadsheet appears to be empty or contains no readable data.")
        return {}
    except Exception as e:
        st.error(f"❌ Unexpected error loading data: {str(e)}")
        logger.error(f"Data loading error: {e}")
        return {}

# =========================================
# ENHANCED DATA PROCESSING
# =========================================
def _parse_numeric_enhanced(s: pd.Series) -> pd.Series:
    """Enhanced numeric parsing with better error handling"""
    if s.empty:
        return s
    
    # Convert to string and clean
    cleaned = (
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("SAR", "", regex=False)
        .str.strip()
    )
    
    # Handle common text values
    cleaned = cleaned.replace({
        "N/A": np.nan,
        "n/a": np.nan,
        "#N/A": np.nan,
        "-": np.nan,
        "": np.nan
    })
    
    return pd.to_numeric(cleaned, errors="coerce")

def process_branch_data_optimized(excel_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Optimized branch data processing with better performance"""
    if not excel_data:
        return pd.DataFrame()
    
    # Column mapping with more variations
    mapping = {
        "Date": "Date", "Day": "Day", "DATE": "Date",
        "Sales Target": "SalesTarget", "SalesTarget": "SalesTarget", "SALES TARGET": "SalesTarget",
        "Sales Achivement": "SalesActual", "Sales Achievement": "SalesActual", "SalesActual": "SalesActual",
        "Sales Achievemnet": "SalesActual", "SALES ACTUAL": "SalesActual",
        "Sales %": "SalesPercent", " Sales %": "SalesPercent", "SalesPercent": "SalesPercent",
        "NOB Target": "NOBTarget", "NOBTarget": "NOBTarget", "NOB TARGET": "NOBTarget",
        "NOB Achievemnet": "NOBActual", "NOB Achievement": "NOBActual", "NOBActual": "NOBActual",
        "NOB ACTUAL": "NOBActual", "Number of Baskets": "NOBActual",
        "NOB %": "NOBPercent", " NOB %": "NOBPercent", "NOBPercent": "NOBPercent",
        "ABV Target": "ABVTarget", "ABVTarget": "ABVTarget", "ABV TARGET": "ABVTarget",
        "ABV Achievement": "ABVActual", " ABV Achievement": "ABVActual", "ABVActual": "ABVActual",
        "Average Basket Value": "ABVActual", "ABV ACTUAL": "ABVActual",
        "ABV %": "ABVPercent", "ABVPercent": "ABVPercent", "ABV PERCENT": "ABVPercent",
        "BANK": "Bank", "Bank": "Bank", "CASH IN BANK": "Bank",
        "CASH": "Cash", "Cash": "Cash", "CASH IN HAND": "Cash",
        "TOTAL LIQUIDITY": "TotalLiquidity", "Total Liquidity": "TotalLiquidity",
        "Change in Liquidity": "ChangeLiquidity", "% of Change": "PctChange",
    }
    
    dfs_with_metadata = []
    
    for sheet_name, df in excel_data.items():
        if df.empty:
            continue
            
        df_copy = df.copy()
        
        # Clean column names
        df_copy.columns = df_copy.columns.astype(str).str.strip()
        
        # Apply column mapping
        df_copy = df_copy.rename(columns=mapping)
        
        # Handle date column detection
        if "Date" not in df_copy.columns:
            date_candidates = [col for col in df_copy.columns 
                             if any(keyword in col.lower() for keyword in ['date', 'day', 'time'])]
            if date_candidates:
                df_copy.rename(columns={date_candidates[0]: "Date"}, inplace=True)
        
        # Add branch metadata
        df_copy["Branch"] = sheet_name
        branch_config = config.BRANCHES.get(sheet_name)
        if branch_config:
            df_copy["BranchName"] = branch_config.name
            df_copy["BranchCode"] = branch_config.code
        else:
            df_copy["BranchName"] = sheet_name
            df_copy["BranchCode"] = "000"
        
        dfs_with_metadata.append(df_copy)
    
    if not dfs_with_metadata:
        return pd.DataFrame()
    
    # Combine all dataframes
    combined = pd.concat(dfs_with_metadata, ignore_index=True)
    
    # Process dates
    if "Date" in combined.columns:
        combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
    
    # Process numeric columns in batch
    numeric_columns = [
        "SalesTarget", "SalesActual", "SalesPercent",
        "NOBTarget", "NOBActual", "NOBPercent",
        "ABVTarget", "ABVActual", "ABVPercent",
        "Bank", "Cash", "TotalLiquidity", "ChangeLiquidity", "PctChange"
    ]
    
    for col in numeric_columns:
        if col in combined.columns:
            combined[col] = _parse_numeric_enhanced(combined[col])
    
    # Data quality checks
    quality_issues = []
    
    # Check for missing critical data
    if "Date" in combined.columns and combined["Date"].isna().sum() > len(combined) * 0.5:
        quality_issues.append("⚠️ More than 50% of dates are missing or invalid")
    
    if "SalesActual" in combined.columns and combined["SalesActual"].isna().sum() > len(combined) * 0.3:
        quality_issues.append("⚠️ Significant amount of sales data is missing")
    
    # Store quality issues in session state for display
    st.session_state.data_quality_issues = quality_issues
    
    logger.info(f"Processed {len(combined)} rows from {len(dfs_with_metadata)} sheets")
    return combined

# =========================================
# ENHANCED ANALYTICS & ALERTS
# =========================================
def generate_performance_alerts(kpis: Dict[str, Any], df: pd.DataFrame) -> List[Alert]:
    """Generate intelligent performance alerts"""
    alerts = []
    
    # Overall performance alerts
    sales_pct = kpis.get("overall_sales_percent", 0)
    if sales_pct < config.ALERT_THRESHOLDS["critical_sales"]:
        alerts.append(Alert(
            "error", 
            f"🚨 Critical: Overall sales at {sales_pct:.1f}% - immediate action required",
            "sales", sales_pct
        ))
    elif sales_pct < config.ALERT_THRESHOLDS["warning_sales"]:
        alerts.append(Alert(
            "warning",
            f"⚠️ Warning: Sales performance at {sales_pct:.1f}% - below expectations",
            "sales", sales_pct
        ))
    
    # Branch-specific alerts
    if "branch_performance" in kpis and not kpis["branch_performance"].empty:
        bp = kpis["branch_performance"]
        
        # Find underperforming branches
        low_sales = bp[bp["SalesPercent"] < config.ALERT_THRESHOLDS["warning_sales"]]
        if not low_sales.empty:
            branch_list = ", ".join(low_sales.index.tolist())
            alerts.append(Alert(
                "warning",
                f"📉 Branches below target: {branch_list}",
                "branch_performance"
            ))
        
        # Find top performers for positive reinforcement
        top_performers = bp[bp["SalesPercent"] >= config.TARGETS["sales_achievement"]]
        if not top_performers.empty:
            branch_list = ", ".join(top_performers.index.tolist())
            alerts.append(Alert(
                "success",
                f"🌟 Excellent performance: {branch_list}",
                "branch_performance"
            ))
    
    # Liquidity alerts
    if "TotalLiquidity" in df.columns and not df["TotalLiquidity"].isna().all():
        latest_liquidity = df.dropna(subset=["TotalLiquidity"])["TotalLiquidity"].iloc[-1]
        if latest_liquidity < config.ALERT_THRESHOLDS["liquidity_warning"]:
            alerts.append(Alert(
                "warning",
                f"💰 Low liquidity alert: SAR {latest_liquidity:,.0f}",
                "liquidity", latest_liquidity
            ))
    
    # Data quality alerts
    if hasattr(st.session_state, 'data_quality_issues') and st.session_state.data_quality_issues:
        for issue in st.session_state.data_quality_issues:
            alerts.append(Alert("info", issue, "data_quality"))
    
    return alerts

def calc_enhanced_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Enhanced KPI calculations with trends and advanced analytics"""
    if df.empty:
        return {}
    
    # Get base KPIs
    kpis = calc_kpis(df)  # Your original function
    
    # Add trend analysis
    if "Date" in df.columns and len(df) > 1:
        # Group by date for trend analysis
        daily_sales = df.groupby("Date")["SalesActual"].sum().sort_index()
        
        if len(daily_sales) > 7:
            # Calculate 7-day moving average
            ma_7 = daily_sales.rolling(window=7, min_periods=1).mean()
            
            # Calculate trend (slope of last 7 days)
            if len(ma_7) >= 7:
                recent_values = ma_7.tail(7).values
                x = np.arange(len(recent_values))
                if len(recent_values) > 1:
                    slope, _ = np.polyfit(x, recent_values, 1)
                    kpis["sales_trend_slope"] = float(slope)
                    kpis["sales_trend_direction"] = "increasing" if slope > 0 else "decreasing"
                    kpis["sales_trend_strength"] = abs(slope) / recent_values.mean() if recent_values.mean() > 0 else 0
    
    # Performance consistency analysis
    if "branch_performance" in kpis and not kpis["branch_performance"].empty:
        bp = kpis["branch_performance"]
        
        # Calculate coefficient of variation for consistency
        sales_cv = bp["SalesPercent"].std() / bp["SalesPercent"].mean() if bp["SalesPercent"].mean() > 0 else 0
        kpis["performance_consistency"] = 1 / (1 + sales_cv)  # Higher is better
        
        # Identify performance gaps
        max_perf = bp["SalesPercent"].max()
        min_perf = bp["SalesPercent"].min()
        kpis["performance_gap"] = max_perf - min_perf
    
    # Advanced scoring with weights
    score_components = []
    weights = []
    
    if kpis.get("overall_sales_percent", 0) > 0:
        score_components.append(min(kpis["overall_sales_percent"] / 100, 1.2))
        weights.append(40)
    
    if kpis.get("overall_nob_percent", 0) > 0:
        score_components.append(min(kpis["overall_nob_percent"] / 100, 1.2))
        weights.append(35)
        
    if kpis.get("overall_abv_percent", 0) > 0:
        score_components.append(min(kpis["overall_abv_percent"] / 100, 1.2))
        weights.append(25)
    
    if score_components and weights:
        weighted_score = sum(s * w for s, w in zip(score_components, weights)) / sum(weights) * 100
        kpis["performance_score"] = weighted_score
        
        # Performance grade
        if weighted_score >= 95:
            kpis["performance_grade"] = "A+"
        elif weighted_score >= 90:
            kpis["performance_grade"] = "A"
        elif weighted_score >= 85:
            kpis["performance_grade"] = "B+"
        elif weighted_score >= 80:
            kpis["performance_grade"] = "B"
        elif weighted_score >= 75:
            kpis["performance_grade"] = "C+"
        else:
            kpis["performance_grade"] = "C"
    
    return kpis

def calc_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Original KPI calculation function (preserved for compatibility)"""
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
        k["date_range"] = {"start": df["Date"].min(), "end": df["Date"].max(), "days": int(df["Date"].dt.date.nunique())}
    
    score=w=0.0
    if k.get("overall_sales_percent",0)>0: score += min(k["overall_sales_percent"]/100,1.2)*40; w+=40
    if k.get("overall_nob_percent",0)>0:   score += min(k["overall_nob_percent"]/100,1.2)*35; w+=35
    if k.get("overall_abv_percent",0)>0:   score += min(k["overall_abv_percent"]/100,1.2)*25; w+=25
    k["performance_score"] = (score/w*100) if w>0 else 0.0
    return k

# =========================================
# ENHANCED VISUALIZATION FUNCTIONS
# =========================================
def create_advanced_sales_chart(df: pd.DataFrame) -> go.Figure:
    """Create advanced sales visualization with targets and trends"""
    if df.empty or "Date" not in df.columns:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    daily_data = df.groupby(["Date", "BranchName"]).agg({
        "SalesActual": "sum",
        "SalesTarget": "sum"
    }).reset_index()
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Daily Sales Performance", "Achievement Rate"),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    colors = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"]
    
    for i, branch in enumerate(sorted(daily_data["BranchName"].unique())):
        branch_data = daily_data[daily_data["BranchName"] == branch]
        color = colors[i % len(colors)]
        
        # Actual sales
        fig.add_trace(
            go.Scatter(
                x=branch_data["Date"],
                y=branch_data["SalesActual"],
                name=f"{branch} Actual",
                line=dict(color=color, width=3),
                fill="tonexty" if i > 0 else "tozeroy",
                hovertemplate=f"<b>{branch}</b><br>Date: %{{x}}<br>Sales: SAR %{{y:,.0f}}<extra></extra>"
            ),
            row=1, col=1
        )
        
        # Target line
        fig.add_trace(
            go.Scatter(
                x=branch_data["Date"],
                y=branch_data["SalesTarget"],
                name=f"{branch} Target",
                line=dict(color=color, width=2, dash="dash"),
                showlegend=False,
                hovertemplate=f"<b>{branch} Target</b><br>Date: %{{x}}<br>Target: SAR %{{y:,.0f}}<extra></extra>"
            ),
            row=1, col=1
        )
        
        # Achievement rate
        achievement_rate = (branch_data["SalesActual"] / branch_data["SalesTarget"] * 100).fillna(0)
        fig.add_trace(
            go.Scatter(
                x=branch_data["Date"],
                y=achievement_rate,
                name=f"{branch} Achievement %",
                line=dict(color=color, width=2),
                showlegend=False,
                hovertemplate=f"<b>{branch}</b><br>Date: %{{x}}<br>Achievement: %{{y:.1f}}%<extra></extra>"
            ),
            row=2, col=1
        )
    
    # Add target line at 100%
    if not daily_data.empty:
        fig.add_hline(y=100, line_dash="dot", line_color="red", 
                     annotation_text="Target: 100%", row=2, col=1)
    
    fig.update_layout(
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Sales (SAR)", row=1, col=1)
    fig.update_yaxes(title_text="Achievement %", row=2, col=1)
    
    return fig

def create_performance_heatmap(bp: pd.DataFrame) -> go.Figure:
    """Create performance heatmap for branch comparison"""
    if bp.empty:
        return go.Figure()
    
    # Prepare data for heatmap
    metrics = ["SalesPercent", "NOBPercent", "ABVPercent"]
    metric_labels = ["Sales %", "NOB %", "ABV %"]
    
    z_data = []
    for metric in metrics:
        z_data.append(bp[metric].values)
    
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=bp.index,
        y=metric_labels,
        colorscale="RdYlGn",
        zmid=90,  # Center at 90%
        colorbar=dict(title="Performance %"),
        hovertemplate="Branch: %{x}<br>Metric: %{y}<br>Value: %{z:.1f}%<extra></extra>"
    ))
    
    fig.update_layout(
        title="Branch Performance Heatmap",
        height=300,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig

def create_liquidity_dashboard(df: pd.DataFrame) -> Tuple[go.Figure, Dict[str, Any]]:
    """Create comprehensive liquidity dashboard"""
    if df.empty or "TotalLiquidity" not in df.columns:
        return go.Figure(), {}
    
    # Process daily liquidity data
    daily_liq = df.dropna(subset=["Date", "TotalLiquidity"]).copy()
    daily_liq = daily_liq.groupby("Date")["TotalLiquidity"].sum().reset_index().sort_values("Date")
    
    if daily_liq.empty:
        return go.Figure(), {}
    
    # Calculate analytics
    daily_liq["Change"] = daily_liq["TotalLiquidity"].diff()
    daily_liq["Change_Pct"] = daily_liq["TotalLiquidity"].pct_change() * 100
    daily_liq["MA_7"] = daily_liq["TotalLiquidity"].rolling(7, min_periods=1).mean()
    
    # Create subplot figure
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Liquidity Trend", "Daily Changes", "Volatility"),
        vertical_spacing=0.08,
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Main liquidity trend
    fig.add_trace(
        go.Scatter(
            x=daily_liq["Date"],
            y=daily_liq["TotalLiquidity"],
            name="Total Liquidity",
            line=dict(color="#3b82f6", width=3),
            fill="tozeroy",
            fillcolor="rgba(59, 130, 246, 0.1)"
        ),
        row=1, col=1
    )
    
    # 7-day moving average
    fig.add_trace(
        go.Scatter(
            x=daily_liq["Date"],
            y=daily_liq["MA_7"],
            name="7-day Average",
            line=dict(color="#f59e0b", width=2, dash="dash")
        ),
        row=1, col=1
    )
    
    # Daily changes
    colors = ["green" if x >= 0 else "red" for x in daily_liq["Change"].fillna(0)]
    fig.add_trace(
        go.Bar(
            x=daily_liq["Date"],
            y=daily_liq["Change"],
            name="Daily Change",
            marker_color=colors,
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Volatility (rolling standard deviation)
    volatility = daily_liq["Change"].rolling(7, min_periods=1).std()
    fig.add_trace(
        go.Scatter(
            x=daily_liq["Date"],
            y=volatility,
            name="Volatility (7d)",
            line=dict(color="#8b5cf6", width=2),
            showlegend=False
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    
    # Calculate summary metrics
    current_liquidity = daily_liq["TotalLiquidity"].iloc[-1]
    prev_liquidity = daily_liq["TotalLiquidity"].iloc[-2] if len(daily_liq) > 1 else current_liquidity
    change_amount = current_liquidity - prev_liquidity
    change_pct = (change_amount / prev_liquidity * 100) if prev_liquidity != 0 else 0
    
    analytics = {
        "current_liquidity": current_liquidity,
        "change_amount": change_amount,
        "change_pct": change_pct,
        "avg_daily_change": daily_liq["Change"].mean(),
        "volatility": daily_liq["Change"].std(),
        "max_liquidity": daily_liq["TotalLiquidity"].max(),
        "min_liquidity": daily_liq["TotalLiquidity"].min()
    }
    
    return fig, analytics

# =========================================
# ENHANCED UI COMPONENTS
# =========================================
def render_alerts(alerts: List[Alert]):
    """Render performance alerts with enhanced styling"""
    if not alerts:
        return
    
    for alert in alerts:
        icon_map = {
            "success": "🟢",
            "warning": "🟡", 
            "error": "🔴",
            "info": "🔵"
        }
        
        icon = icon_map.get(alert.level, "ℹ️")
        
        st.markdown(
            f"""
            <div class="alert-container alert-{alert.level}">
                {icon} {alert.message}
            </div>
            """,
            unsafe_allow_html=True
        )

def render_quick_actions():
    """Render quick action buttons for common operations"""
    st.markdown("### ⚡ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 This Month", use_container_width=True):
            today = datetime.today()
            st.session_state.start_date = today.replace(day=1).date()
            st.session_state.end_date = today.date()
            st.rerun()
    
    with col2:
        if st.button("📈 Last 30 Days", use_container_width=True):
            today = datetime.today()
            st.session_state.start_date = (today - timedelta(days=30)).date()
            st.session_state.end_date = today.date()
            st.rerun()
    
    with col3:
        if st.button("📅 Last Week", use_container_width=True):
            today = datetime.today()
            st.session_state.start_date = (today - timedelta(days=7)).date()
            st.session_state.end_date = today.date()
            st.rerun()
    
    with col4:
        if st.button("🔄 Reset Filters", use_container_width=True):
            # Reset to all branches and full date range
            if 'all_branches' in st.session_state:
                st.session_state.selected_branches = st.session_state.all_branches.copy()
            if 'full_date_range' in st.session_state:
                st.session_state.start_date = st.session_state.full_date_range[0]
                st.session_state.end_date = st.session_state.full_date_range[1]
            st.rerun()

def render_enhanced_branch_cards(bp: pd.DataFrame):
    """Enhanced branch cards with better visualization"""
    st.markdown("### 🏪 Branch Overview")
    if bp.empty:
        st.info("No branch summary available.")
        return

    items = list(bp.index)
    per_row = max(1, int(getattr(config, "CARDS_PER_ROW", 3)))

    for i in range(0, len(items), per_row):
        cols = st.columns(per_row, gap="medium")
        for j in range(per_row):
            col_idx = i + j
            if col_idx >= len(items):
                continue
                
            with cols[j]:
                branch_name = items[col_idx]
                r = bp.loc[branch_name]
                
                # Calculate overall score
                scores = [r.get("SalesPercent", 0), r.get("NOBPercent", 0), r.get("ABVPercent", 0)]
                avg_score = np.mean([s for s in scores if not np.isnan(s)])
                
                # Determine performance level
                if avg_score >= 95:
                    perf_class = "perf-excellent"
                    perf_icon = "🌟"
                elif avg_score >= 85:
                    perf_class = "perf-good" 
                    perf_icon = "✅"
                elif avg_score >= 75:
                    perf_class = "perf-warning"
                    perf_icon = "⚠️"
                else:
                    perf_class = "perf-critical"
                    perf_icon = "🚨"
                
                # Get branch color
                branch_config = None
                for sheet_key, config_obj in config.BRANCHES.items():
                    if config_obj.name == branch_name:
                        branch_config = config_obj
                        break
                
                color = branch_config.color if branch_config else "#6b7280"
                
                st.markdown(
                    f"""
                    <div class="card" style="--branch-color: {color};">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
                            <div style="font-weight:800;font-size:1.1rem;color:#1e293b;">{branch_name}</div>
                            <span class="perf-indicator {perf_class}">{perf_icon} {avg_score:.1f}%</span>
                        </div>
                        
                        <div class="branch-metrics">
                            <div>
                                <div class="label">Sales %</div>
                                <div class="value" style="color:{color};">{r.get('SalesPercent', 0):.1f}%</div>
                            </div>
                            <div>
                                <div class="label">NOB %</div>
                                <div class="value" style="color:{color};">{r.get('NOBPercent', 0):.1f}%</div>
                            </div>
                            <div>
                                <div class="label">ABV %</div>
                                <div class="value" style="color:{color};">{r.get('ABVPercent', 0):.1f}%</div>
                            </div>
                        </div>
                        
                        <div style="margin-top:12px;display:flex;justify-content:space-between;font-size:0.8rem;color:#64748b;">
                            <span>Target: SAR {r.get('SalesTarget', 0):,.0f}</span>
                            <span>Actual: SAR {r.get('SalesActual', 0):,.0f}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

def render_data_quality_sidebar():
    """Render data quality indicators in sidebar"""
    if not hasattr(st.session_state, 'data_quality_issues'):
        return
    
    with st.sidebar:
        st.markdown('<div class="sb-section">Data Quality</div>', unsafe_allow_html=True)
        
        if st.session_state.data_quality_issues:
            st.markdown("🟡 Issues detected")
            with st.expander("View Details"):
                for issue in st.session_state.data_quality_issues:
                    st.markdown(f"• {issue}")
        else:
            st.markdown("🟢 Data looks good")

# =========================================
# ENHANCED MAIN FUNCTION
# =========================================
def main():
    # Set page config
    st.set_page_config(
        page_title=config.PAGE_TITLE, 
        layout=config.LAYOUT, 
        initial_sidebar_state="expanded"
    )
    
    apply_enhanced_css()

    # Load data with enhanced error handling
    try:
        sheets_map = load_workbook_with_enhanced_error_handling(config.DEFAULT_PUBLISHED_URL)
        if not sheets_map:
            st.error("❌ Unable to load data. Please check the connection and try again.")
            st.stop()
            
    except Exception as e:
        st.error(f"❌ Critical error loading data: {str(e)}")
        st.stop()

    # Process data
    df_all = process_branch_data_optimized(sheets_map)
    if df_all.empty:
        st.error("❌ Could not process data. Check column names and sheet structure.")
        st.stop()

    # Initialize session state
    all_branches = sorted(df_all["BranchName"].dropna().unique()) if "BranchName" in df_all else []
    st.session_state.all_branches = all_branches
    
    if "selected_branches" not in st.session_state:
        st.session_state.selected_branches = list(all_branches)

    # Store full date range for reset functionality
    if "Date" in df_all.columns and df_all["Date"].notna().any():
        st.session_state.full_date_range = (
            df_all["Date"].min().date(),
            df_all["Date"].max().date()
        )

    # Enhanced sidebar
    with st.sidebar:
        st.markdown('<div class="sb-title">📊 <span>AL KHAIR DASHBOARD</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-subtle">Real-time business intelligence</div>', unsafe_allow_html=True)

        # Quick actions in sidebar
        st.markdown('<div class="sb-section">Quick Actions</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", use_container_width=True):
                load_workbook_with_enhanced_error_handling.clear()
                st.rerun()
        with col2:
            if st.button("📊 Export", use_container_width=True):
                st.session_state.show_export = True

        st.markdown('<div class="sb-hr"></div>', unsafe_allow_html=True)

        # Branch selection with select all/none
        st.markdown('<div class="sb-section">Branch Selection</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Select All", use_container_width=True):
                st.session_state.selected_branches = list(all_branches)
                st.rerun()
        with col2:
            if st.button("Select None", use_container_width=True):
                st.session_state.selected_branches = []
                st.rerun()
        
        # Individual branch checkboxes
        selected_branches = []
        for i, branch in enumerate(all_branches):
            if st.checkbox(
                branch, 
                value=(branch in st.session_state.selected_branches),
                key=f"branch_check_{i}"
            ):
                selected_branches.append(branch)
        
        st.session_state.selected_branches = selected_branches

        # Enhanced date filtering
        df_for_bounds = (
            df_all[df_all["BranchName"].isin(st.session_state.selected_branches)] 
            if st.session_state.selected_branches 
            else df_all
        )
        
        if "Date" in df_for_bounds.columns and df_for_bounds["Date"].notna().any():
            date_min, date_max = df_for_bounds["Date"].min().date(), df_for_bounds["Date"].max().date()
        else:
            date_min = date_max = datetime.today().date()

        st.markdown('<div class="sb-section">Date Range</div>', unsafe_allow_html=True)
        
        # Quick date presets
        today = datetime.today().date()
        if st.button("This Month", use_container_width=True):
            st.session_state.start_date = today.replace(day=1)
            st.session_state.end_date = today
            st.rerun()
            
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Last 7d", use_container_width=True):
                st.session_state.start_date = today - timedelta(days=7)
                st.session_state.end_date = today
                st.rerun()
        with col2:
            if st.button("Last 30d", use_container_width=True):
                st.session_state.start_date = today - timedelta(days=30)
                st.session_state.end_date = today
                st.rerun()

        # Custom date inputs
        start_date = st.date_input(
            "Start Date:",
            value=st.session_state.get("start_date", date_min),
            min_value=date_min,
            max_value=date_max
        )
        
        end_date = st.date_input(
            "End Date:",
            value=st.session_state.get("end_date", date_max),
            min_value=date_min,
            max_value=date_max
        )
        
        st.session_state.start_date = start_date
        st.session_state.end_date = end_date

        # Data quality indicators
        render_data_quality_sidebar()

    # Hero header with enhanced animation
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
    df_filtered = df_all.copy()
    
    if "BranchName" in df_filtered.columns and st.session_state.selected_branches:
        df_filtered = df_filtered[df_filtered["BranchName"].isin(st.session_state.selected_branches)]
    
    if "Date" in df_filtered.columns and df_filtered["Date"].notna().any():
        date_mask = (
            (df_filtered["Date"].dt.date >= st.session_state.start_date) & 
            (df_filtered["Date"].dt.date <= st.session_state.end_date)
        )
        df_filtered = df_filtered.loc[date_mask]

    if df_filtered.empty:
        st.warning("⚠️ No data available for the selected filters. Try adjusting your selection.")
        st.stop()

    # Calculate enhanced KPIs and generate alerts
    kpis = calc_enhanced_kpis(df_filtered)
    alerts = generate_performance_alerts(kpis, df_filtered)

    # Display alerts
    if alerts:
        render_alerts(alerts)

    # Enhanced insights with trend information
    with st.expander("⚡ AI-Powered Insights", expanded=True):
        insights = []
        
        # Performance insights
        if kpis.get("performance_score", 0) >= config.TARGETS["performance_score"]:
            grade = kpis.get("performance_grade", "B")
            insights.append(f"🎯 **Excellent Performance**: Overall grade {grade} ({kpis['performance_score']:.1f}/100)")
        elif kpis.get("performance_score", 0) >= 70:
            insights.append(f"📊 **Good Performance**: Score {kpis['performance_score']:.1f}/100 - room for improvement")
        else:
            insights.append(f"⚠️ **Performance Alert**: Score {kpis['performance_score']:.1f}/100 - immediate attention needed")

        # Trend insights
        if kpis.get("sales_trend_direction") == "increasing":
            insights.append(f"📈 **Positive Trend**: Sales trending upward with {kpis.get('sales_trend_strength', 0)*100:.1f}% momentum")
        elif kpis.get("sales_trend_direction") == "decreasing":
            insights.append(f"📉 **Declining Trend**: Sales trending downward - investigate causes")

        # Branch consistency
        if kpis.get("performance_consistency", 0) > 0.8:
            insights.append("🎯 **High Consistency**: All branches performing uniformly")
        elif kpis.get("performance_gap", 0) > 20:
            insights.append(f"⚖️ **Performance Gap**: {kpis['performance_gap']:.1f}% difference between best and worst branch")

        # Top performers
        if "branch_performance" in kpis and not kpis["branch_performance"].empty:
            bp = kpis["branch_performance"]
            top_branch = bp["SalesPercent"].idxmax()
            top_score = bp["SalesPercent"].max()
            insights.append(f"🥇 **Top Performer**: {top_branch} leading with {top_score:.1f}% sales achievement")

        if insights:
            for insight in insights:
                st.markdown(f"• {insight}")
        else:
            st.markdown("📊 Analysis complete - all metrics within normal parameters")

    # Enhanced tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Dashboard", 
        "📈 Advanced Analytics", 
        "💧 Liquidity Intelligence", 
        "🎯 Performance Deep-Dive",
        "📊 Export & Reports"
    ])

    with tab1:
        # Enhanced overview with better metrics display
        st.markdown("### 🏆 Executive Summary")

        # Main KPI row
        col1, col2, col3, col4, col5 = st.columns(5, gap="medium")
        
        # Sales metrics
        with col1:
            delta_color = "normal" if kpis.get("overall_sales_percent", 0) >= config.TARGETS["sales_achievement"] else "inverse"
            st.metric(
                "💰 Total Sales",
                f"SAR {kpis.get('total_sales_actual', 0):,.0f}",
                delta=f"{kpis.get('overall_sales_percent', 0):.1f}% of target",
                delta_color=delta_color
            )
        
        with col2:
            variance = kpis.get("total_sales_variance", 0)
            variance_color = "normal" if variance >= 0 else "inverse"
            st.metric(
                "📊 Sales Variance",
                f"SAR {variance:,.0f}",
                delta=f"{kpis.get('overall_sales_percent', 0) - 100:+.1f}%",
                delta_color=variance_color
            )
        
        with col3:
            nob_color = "normal" if kpis.get("overall_nob_percent", 0) >= config.TARGETS["nob_achievement"] else "inverse"
            st.metric(
                "🛍️ Total Baskets",
                f"{kpis.get('total_nob_actual', 0):,.0f}",
                delta=f"{kpis.get('overall_nob_percent', 0):.1f}% achievement",
                delta_color=nob_color
            )
        
        with col4:
            abv_color = "normal" if kpis.get("overall_abv_percent", 0) >= config.TARGETS["abv_achievement"] else "inverse"
            st.metric(
                "💎 Avg Basket Value",
                f"SAR {kpis.get('avg_abv_actual', 0):.2f}",
                delta=f"{kpis.get('overall_abv_percent', 0):.1f}% vs target",
                delta_color=abv_color
            )
        
        with col5:
            score_color = "normal" if kpis.get("performance_score", 0) >= config.TARGETS["performance_score"] else "off"
            grade = kpis.get("performance_grade", "C")
            st.metric(
                "⭐ Performance Score",
                f"{kpis.get('performance_score', 0):.0f}/100",
                delta=f"Grade: {grade}",
                delta_color=score_color
            )

        # Enhanced branch cards
        if "branch_performance" in kpis and not kpis["branch_performance"].empty:
            render_enhanced_branch_cards(kpis["branch_performance"])
        
        # Performance comparison chart
        if "branch_performance" in kpis and not kpis["branch_performance"].empty:
            st.markdown("### 📊 Performance Heatmap")
            heatmap_fig = create_performance_heatmap(kpis["branch_performance"])
            st.plotly_chart(heatmap_fig, use_container_width=True, config={"displayModeBar": False})

    with tab2:
        st.markdown("### 📈 Advanced Sales Analytics")
        
        if "Date" in df_filtered.columns and df_filtered["Date"].notna().any():
            # Advanced sales chart
            advanced_chart = create_advanced_sales_chart(df_filtered)
            st.plotly_chart(advanced_chart, use_container_width=True, config={"displayModeBar": False})
            
            # Trend analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Trend Analysis")
                if kpis.get("sales_trend_slope"):
                    trend_direction = "📈 Increasing" if kpis["sales_trend_slope"] > 0 else "📉 Decreasing"
                    st.metric("Sales Trend", trend_direction, f"{kpis['sales_trend_slope']:.2f}/day")
                
                consistency_score = kpis.get("performance_consistency", 0) * 100
                st.metric("Branch Consistency", f"{consistency_score:.1f}%", "Higher is better")
            
            with col2:
                st.markdown("#### 📊 Performance Statistics")
                if "branch_performance" in kpis:
                    bp = kpis["branch_performance"]
                    st.metric("Performance Gap", f"{kpis.get('performance_gap', 0):.1f}%", "Between best/worst")
                    
                    best_branch = bp["SalesPercent"].idxmax()
                    worst_branch = bp["SalesPercent"].idxmin()
                    st.write(f"**Best**: {best_branch} ({bp.loc[best_branch, 'SalesPercent']:.1f}%)")
                    st.write(f"**Needs Attention**: {worst_branch} ({bp.loc[worst_branch, 'SalesPercent']:.1f}%)")
            
            # Forecasting section
            st.markdown("#### 🔮 Simple Forecast")
            try:
                # Simple 7-day forecast based on moving average
                daily_sales = df_filtered.groupby("Date")["SalesActual"].sum().sort_index()
                if len(daily_sales) >= 7:
                    ma_7 = daily_sales.rolling(window=7).mean()
                    last_ma = ma_7.iloc[-1]
                    
                    # Create forecast dates
                    last_date = daily_sales.index[-1]
                    forecast_dates = pd.date_range(
                        start=last_date + pd.Timedelta(days=1),
                        periods=7
                    )
                    
                    # Simple forecast (could be enhanced with proper time series models)
                    forecast_values = [last_ma] * 7
                    
                    # Create forecast chart
                    fig_forecast = go.Figure()
                    
                    # Historical data
                    fig_forecast.add_trace(go.Scatter(
                        x=daily_sales.index[-30:],  # Last 30 days
                        y=daily_sales.values[-30:],
                        name="Historical Sales",
                        mode="lines+markers",
                        line=dict(color="#3b82f6", width=2)
                    ))
                    
                    # Moving average
                    fig_forecast.add_trace(go.Scatter(
                        x=ma_7.index[-30:],
                        y=ma_7.values[-30:],
                        name="7-day Average",
                        mode="lines",
                        line=dict(color="#10b981", width=2, dash="dash")
                    ))
                    
                    # Forecast
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=forecast_values,
                        name="7-day Forecast",
                        mode="lines+markers",
                        line=dict(color="#f59e0b", width=2, dash="dot"),
                        marker=dict(symbol="diamond", size=8)
                    ))
                    
                    fig_forecast.update_layout(
                        title="Sales Forecast (7 Days)",
                        height=400,
                        showlegend=True,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)"
                    )
                    
                    st.plotly_chart(fig_forecast, use_container_width=True, config={"displayModeBar": False})
                    
                    # Forecast summary
                    total_forecast = sum(forecast_values)
                    st.info(f"📊 **7-Day Forecast**: SAR {total_forecast:,.0f} (based on recent trends)")
                    
            except Exception as e:
                st.warning("⚠️ Forecasting requires more historical data")
        else:
            st.info("Date information not available for trend analysis")

    with tab3:
        st.markdown("### 💧 Liquidity Intelligence Center")
        
        if "TotalLiquidity" in df_filtered.columns:
            # Create comprehensive liquidity dashboard
            liq_fig, liq_analytics = create_liquidity_dashboard(df_filtered)
            
            if liq_analytics:
                # Liquidity KPIs
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    current_liq = liq_analytics["current_liquidity"]
                    st.metric("💰 Current Liquidity", f"SAR {current_liq:,.0f}")
                
                with col2:
                    change_amount = liq_analytics["change_amount"]
                    change_pct = liq_analytics["change_pct"]
                    color = "normal" if change_amount >= 0 else "inverse"
                    st.metric(
                        "📈 Daily Change", 
                        f"SAR {change_amount:,.0f}",
                        delta=f"{change_pct:+.1f}%",
                        delta_color=color
                    )
                
                with col3:
                    avg_change = liq_analytics["avg_daily_change"]
                    st.metric("📊 Avg Daily Δ", f"SAR {avg_change:,.0f}")
                
                with col4:
                    volatility = liq_analytics["volatility"]
                    st.metric("📉 Volatility", f"SAR {volatility:,.0f}")
                
                # Liquidity chart
                st.plotly_chart(liq_fig, use_container_width=True, config={"displayModeBar": False})
                
                # Liquidity insights
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🎯 Liquidity Health")
                    
                    max_liq = liq_analytics["max_liquidity"]
                    min_liq = liq_analytics["min_liquidity"]
                    range_pct = ((max_liq - min_liq) / min_liq * 100) if min_liq > 0 else 0
                    
                    st.write(f"**Range**: SAR {min_liq:,.0f} - SAR {max_liq:,.0f}")
                    st.write(f"**Variation**: {range_pct:.1f}%")
                    
                    # Health indicator
                    if current_liq > liq_analytics["avg_daily_change"] * 30:  # 30 days runway
                        st.success("✅ Healthy liquidity position")
                    elif current_liq > liq_analytics["avg_daily_change"] * 15:  # 15 days runway
                        st.warning("⚠️ Moderate liquidity - monitor closely")
                    else:
                        st.error("🚨 Low liquidity - immediate attention required")
                
                with col2:
                    st.markdown("#### 🔍 Risk Indicators")
                    
                    # Calculate risk metrics
                    if volatility > current_liq * 0.1:  # High volatility
                        st.warning("⚠️ High volatility detected")
                    else:
                        st.success("✅ Stable liquidity pattern")
                    
                    # Trend indicator
                    if avg_change > 0:
                        st.success(f"📈 Positive trend: +SAR {avg_change:,.0f}/day")
                    elif avg_change < 0:
                        st.error(f"📉 Negative trend: SAR {avg_change:,.0f}/day")
                    else:
                        st.info("➡️ Stable liquidity")
                        
                    # Days of runway calculation
                    if avg_change < 0 and current_liq > 0:
                        days_runway = int(current_liq / abs(avg_change))
                        if days_runway < 30:
                            st.error(f"⏰ {days_runway} days runway at current burn rate")
                        elif days_runway < 90:
                            st.warning(f"⏰ {days_runway} days runway at current burn rate")
                        else:
                            st.success(f"⏰ {days_runway} days runway at current burn rate")
            else:
                st.info("📊 Insufficient liquidity data for analysis")
        else:
            st.info("💧 Liquidity columns not found in the data")

    with tab4:
        st.markdown("### 🎯 Performance Deep-Dive")
        
        if "branch_performance" in kpis and not kpis["branch_performance"].empty:
            bp = kpis["branch_performance"]
            
            # Performance matrix
            st.markdown("#### 📊 Performance Matrix")
            
            # Create a more detailed comparison table
            detailed_table = bp.copy()
            detailed_table["Overall_Score"] = (
                detailed_table["SalesPercent"] * 0.4 + 
                detailed_table["NOBPercent"] * 0.35 + 
                detailed_table["ABVPercent"] * 0.25
            ).round(1)
            
            # Add performance grades
            detailed_table["Grade"] = detailed_table["Overall_Score"].apply(
                lambda x: "A+" if x >= 95 else "A" if x >= 90 else "B+" if x >= 85 else "B" if x >= 80 else "C+" if x >= 75 else "C"
            )
            
            # Format for display
            display_table = detailed_table[[
                "SalesPercent", "NOBPercent", "ABVPercent", "Overall_Score", "Grade",
                "SalesActual", "SalesTarget"
            ]].round(1)
            
            display_table.columns = [
                "Sales %", "NOB %", "ABV %", "Overall Score", "Grade", 
                "Sales Actual", "Sales Target"
            ]
            
            # Style the table
            def style_performance(val, col_name):
                if col_name in ["Sales %", "NOB %", "ABV %", "Overall Score"]:
                    if val >= 95:
                        return "background-color: #dcfce7; color: #166534"
                    elif val >= 85:
                        return "background-color: #dbeafe; color: #1e40af"
                    elif val >= 75:
                        return "background-color: #fef3c7; color: #92400e"
                    else:
                        return "background-color: #fee2e2; color: #991b1b"
                return ""
            
            styled_table = display_table.style.format({
                "Sales %": "{:.1f}",
                "NOB %": "{:.1f}",
                "ABV %": "{:.1f}",
                "Overall Score": "{:.1f}",
                "Sales Actual": "{:,.0f}",
                "Sales Target": "{:,.0f}"
            })
            
            st.dataframe(styled_table, use_container_width=True)
            
            # Branch deep-dive selector
            st.markdown("#### 🔍 Branch Deep-Dive")
            selected_branch = st.selectbox("Select branch for detailed analysis:", bp.index.tolist())
            
            if selected_branch:
                branch_data = df_filtered[df_filtered["BranchName"] == selected_branch].copy()
                
                if not branch_data.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"##### 📊 {selected_branch} Performance")
                        branch_row = bp.loc[selected_branch]
                        
                        # Key metrics
                        st.metric("Sales Achievement", f"{branch_row['SalesPercent']:.1f}%")
                        st.metric("NOB Achievement", f"{branch_row['NOBPercent']:.1f}%")
                        st.metric("ABV Achievement", f"{branch_row['ABVPercent']:.1f}%")
                        
                        # Performance vs target
                        sales_gap = branch_row['SalesActual'] - branch_row['SalesTarget']
                        gap_color = "normal" if sales_gap >= 0 else "inverse"
                        st.metric("Sales Gap", f"SAR {sales_gap:,.0f}", delta_color=gap_color)
                    
                    with col2:
                        st.markdown(f"##### 📈 {selected_branch} Trends")
                        
                        if "Date" in branch_data.columns and len(branch_data) > 1:
                            # Create daily trend for selected branch
                            daily_branch = branch_data.groupby("Date").agg({
                                "SalesActual": "sum",
                                "SalesTarget": "sum"
                            }).reset_index().sort_values("Date")
                            
                            fig_branch = go.Figure()
                            
                            fig_branch.add_trace(go.Scatter(
                                x=daily_branch["Date"],
                                y=daily_branch["SalesActual"],
                                name="Actual Sales",
                                mode="lines+markers",
                                line=dict(color="#3b82f6", width=3)
                            ))
                            
                            fig_branch.add_trace(go.Scatter(
                                x=daily_branch["Date"],
                                y=daily_branch["SalesTarget"],
                                name="Target",
                                mode="lines",
                                line=dict(color="#ef4444", width=2, dash="dash")
                            ))
                            
                            fig_branch.update_layout(
                                title=f"{selected_branch} Daily Performance",
                                height=300,
                                showlegend=True,
                                plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)"
                            )
                            
                            st.plotly_chart(fig_branch, use_container_width=True, config={"displayModeBar": False})
                            
                            # Performance insights for selected branch
                            recent_performance = daily_branch["SalesActual"].tail(7).mean()
                            target_performance = daily_branch["SalesTarget"].tail(7).mean()
                            recent_achievement = (recent_performance / target_performance * 100) if target_performance > 0 else 0
                            
                            if recent_achievement >= 100:
                                st.success(f"✅ {selected_branch} is exceeding targets recently ({recent_achievement:.1f}%)")
                            elif recent_achievement >= 90:
                                st.info(f"📊 {selected_branch} is close to targets ({recent_achievement:.1f}%)")
                            else:
                                st.warning(f"⚠️ {selected_branch} is underperforming recently ({recent_achievement:.1f}%)")
        else:
            st.info("No branch performance data available for deep-dive analysis")

    with tab5:
        st.markdown("### 📊 Export & Reports")
        
        if not df_filtered.empty:
            col1, col2, col3 = st.columns(3)
            
            # Enhanced export options
            with col1:
                st.markdown("#### 📄 Standard Reports")
                
                # Excel export with multiple sheets
                def create_excel_report():
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                        # Main data
                        df_filtered.to_excel(writer, sheet_name="Raw Data", index=False)
                        
                        # Branch summary
                        if "branch_performance" in kpis:
                            kpis["branch_performance"].to_excel(writer, sheet_name="Branch Summary")
                        
                        # Daily aggregates
                        if "Date" in df_filtered.columns:
                            daily_summary = df_filtered.groupby("Date").agg({
                                "SalesActual": "sum",
                                "SalesTarget": "sum", 
                                "NOBActual": "sum",
                                "TotalLiquidity": "sum"
                            }).reset_index()
                            daily_summary.to_excel(writer, sheet_name="Daily Summary", index=False)
                        
                        # KPI summary
                        kpi_df = pd.DataFrame([kpis]).T
                        kpi_df.columns = ["Value"]
                        kpi_df.to_excel(writer, sheet_name="KPIs")
                    
                    return buffer.getvalue()
                
                excel_data = create_excel_report()
                st.download_button(
                    "📊 Download Complete Report (Excel)",
                    data=excel_data,
                    file_name=f"AlKhair_Complete_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
                
                # CSV export
                csv_data = df_filtered.to_csv(index=False)
                st.download_button(
                    "📄 Download Raw Data (CSV)",
                    data=csv_data,
                    file_name=f"AlKhair_Data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                st.markdown("#### 📈 Analytics Reports")
                
                if "branch_performance" in kpis:
                    branch_csv = kpis["branch_performance"].to_csv()
                    st.download_button(
                        "🏪 Branch Performance (CSV)",
                        data=branch_csv,
                        file_name=f"AlKhair_Branches_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                # KPI report
                kpi_report = pd.DataFrame([kpis]).T
                kpi_report.columns = ["Value"]
                kpi_csv = kpi_report.to_csv()
                st.download_button(
                    "🎯 KPI Summary (CSV)",
                    data=kpi_csv,
                    file_name=f"AlKhair_KPIs_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col3:
                st.markdown("#### 🎨 Visualizations")
                st.info("📊 Charts can be downloaded using the camera icon in each chart's toolbar")
                
                # Print-friendly summary
                st.markdown("##### 📋 Executive Summary")
                summary_text = f"""
**Al Khair Business Performance Report**
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

**Overall Performance:**
• Total Sales: SAR {kpis.get('total_sales_actual', 0):,.0f}
• Sales Achievement: {kpis.get('overall_sales_percent', 0):.1f}%
• Performance Score: {kpis.get('performance_score', 0):.0f}/100
• Grade: {kpis.get('performance_grade', 'C')}

**Period:** {st.session_state.start_date} to {st.session_state.end_date}
**Branches:** {', '.join(st.session_state.selected_branches)}
                """
                
                st.text_area("Summary for copy/paste:", summary_text, height=200)
        else:
            st.info("No data available for export with current filters")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"🚨 Critical application error: {str(e)}")
        logger.error(f"Application error: {e}")
        
        # Show error details in expander
        with st.expander("🔧 Error Details (for debugging)"):
            st.exception(e)
