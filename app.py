# Enhanced Alkhair Family Market Dashboard - Single Source
# Beautiful modern dashboard with glassmorphism design

import io
import re
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# =========================
# CONFIGURATION
# =========================
@dataclass
class Config:
    """Application configuration settings"""
    PAGE_TITLE = "Alkhair Family Market — Executive Dashboard"
    LAYOUT = "wide"
    REFRESH_MINUTES = 1
    
    # Single data source
    DATA_SOURCE_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQG7boLWl2bNLCPR05NXv6EFpPPcFfXsiXPQ7rAGYr3q8Nkc2Ijg8BqEwVofcMLSg/pub?output=csv"
    
    # Comprehensive column aliases
    COLUMN_MAPPINGS = {
        "Date": ["date", "bill_date", "txn_date", "invoice_date", "posting_date", "transaction_date", "sale_date", "entry_date"],
        "Store": ["store", "branch", "location", "shop", "outlet", "store_name", "branch_name"],
        "Sales": ["sales", "amount", "net_sales", "revenue", "netamount", "net_amount", "actual_sales", "sale_amount", "total_sales", "bill_amount"],
        "TargetSales": ["target_sales", "target", "sales_target", "budget_sales", "planned_sales", "target_amount"],
        "COGS": ["cogs", "cost", "purchase_cost", "cost_of_goods", "cost_of_sales", "cost_amount"],
        "Qty": ["qty", "quantity", "qty_sold", "quantity_sold", "units", "pcs", "pieces", "units_sold", "qnty", "qnt", "qty.", "total_qty"],
        "Tickets": ["tickets", "bills", "invoice_count", "transactions", "bills_count", "baskets", "no_of_baskets", "bill_count", "receipt_count"],
        "Category": ["category", "cat", "product_category", "item_category", "type"],
        "Item": ["item", "product", "sku", "item_name", "product_name", "description", "product_desc"],
        "InventoryQty": ["inventoryqty", "inventory_qty", "stock_qty", "stockqty", "stock_quantity", "current_stock"],
        "InventoryValue": ["inventoryvalue", "inventory_value", "stock_value", "stockvalue", "inventory_amount"],
        "BankBalance": ["bank_balance", "bank", "bank_amount", "bankbalance", "account_balance"],
        "CashBalance": ["cash_balance", "cash", "cash_amount", "cashbalance", "petty_cash"],
        "Expenses": ["expenses", "operating_expenses", "opex", "costs", "expenditure", "expense_amount"],
        "CustomerCount": ["customer_count", "customers", "footfall", "visitors", "customer_visits"],
        "Supplier": ["supplier", "vendor", "supplier_name", "vendor_name"],
        "PaymentMethod": ["payment_method", "payment_type", "payment", "pay_method"],
        "Employee": ["employee", "staff", "cashier", "user", "operator", "salesperson"],
        "Discount": ["discount", "discount_amount", "discount_value", "promo", "promotion"],
        "Tax": ["tax", "vat", "tax_amount", "sales_tax"],
        "UnitPrice": ["unit_price", "price", "rate", "selling_price", "sp", "item_price"],
        "UnitCost": ["unit_cost", "cost_price", "cp", "purchase_price"]
    }

config = Config()

# Page setup
st.set_page_config(
    page_title=config.PAGE_TITLE,
    layout=config.LAYOUT,
    initial_sidebar_state="collapsed",
)

# =========================
# BEAUTIFUL STYLING
# =========================
def apply_stunning_css():
    """Apply beautiful modern CSS with glassmorphism effects"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    :root {
        --primary: #3b82f6;
        --secondary: #8b5cf6;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --dark: #1e293b;
        --light: #f8fafc;
        --white: #ffffff;
        --text-primary: #1f2937;
        --text-secondary: #6b7280;
        --text-muted: #9ca3af;
        --border: rgba(203, 213, 225, 0.3);
        --glass: rgba(255, 255, 255, 0.25);
        --glass-border: rgba(255, 255, 255, 0.18);
    }
    
    /* Global Styles */
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #89f7fe 50%, #66a6ff 75%, #667eea 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main .block-container {
        padding: 2rem 2rem 3rem 2rem;
        max-width: 1400px;
    }
    
    /* Glassmorphism Header */
    .hero-header {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 24px;
        padding: 32px;
        margin: -1rem -1rem 3rem -1rem;
        color: white;
        position: relative;
        overflow: hidden;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
    }
    
    .hero-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
        pointer-events: none;
    }
    
    .hero-content {
        position: relative;
        z-index: 1;
    }
    
    /* Beautiful Metric Cards */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 20px !important;
        padding: 28px 24px !important;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
        overflow: hidden !important;
        margin: 12px 0 !important;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-8px) scale(1.02) !important;
        box-shadow: 0 32px 64px rgba(0, 0, 0, 0.15) !important;
        border-color: rgba(59, 130, 246, 0.4) !important;
    }
    
    [data-testid="metric-container"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 50%, var(--success) 100%);
        background-size: 200% 100%;
        animation: shimmer 3s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { background-position: -200% 0; }
        50% { background-position: 200% 0; }
    }
    
    /* Enhanced Metric Styling */
    [data-testid="metric-container"] [data-testid="metric-value"] {
        font-size: 2.5rem !important;
        font-weight: 900 !important;
        color: var(--text-primary) !important;
        line-height: 1 !important;
        margin: 12px 0 8px 0 !important;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    [data-testid="metric-container"] [data-testid="metric-label"] {
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        color: var(--text-secondary) !important;
        text-transform: none !important;
        letter-spacing: 0.02em !important;
        margin-bottom: 8px !important;
    }
    
    [data-testid="metric-container"] [data-testid="metric-delta"] {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        padding: 6px 12px !important;
        border-radius: 12px !important;
        margin-top: 8px !important;
        backdrop-filter: blur(10px);
    }
    
    /* Tab Navigation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 8px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        margin: 2rem 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 56px;
        padding: 14px 28px;
        background: transparent;
        border-radius: 12px;
        color: var(--text-secondary);
        font-weight: 600;
        font-size: 0.95rem;
        border: none;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(59, 130, 246, 0.1);
        color: var(--primary);
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--white) !important;
        color: var(--text-primary) !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15) !important;
        border: 1px solid rgba(59, 130, 246, 0.2) !important;
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
    }
    
    /* Chart Containers */
    .stPlotlyChart > div {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(20px) !important;
        border-radius: 20px !important;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1) !important;
        padding: 24px !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        transition: all 0.3s ease !important;
        margin: 16px 0 !important;
    }
    
    .stPlotlyChart > div:hover {
        box-shadow: 0 32px 64px rgba(0, 0, 0, 0.15) !important;
        transform: translateY(-4px) !important;
    }
    
    /* Section Headers */
    .main h1, .main h2, .main h3 {
        color: var(--white);
        font-weight: 800;
        margin: 2.5rem 0 1.5rem 0;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        position: relative;
    }
    
    .main h3 {
        font-size: 1.4rem;
        padding: 20px 0 16px 0;
        position: relative;
    }
    
    .main h3::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 80px;
        height: 3px;
        background: linear-gradient(90deg, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0.4));
        border-radius: 2px;
    }
    
    /* Filter Section */
    .filter-section {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 24px;
        margin: 24px 0;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
    }
    
    /* DataFrames */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.95) !important;
        border-radius: 16px !important;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        overflow: hidden !important;
        backdrop-filter: blur(20px);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 14px 28px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 10px 25px rgba(59, 130, 246, 0.3) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 20px 40px rgba(59, 130, 246, 0.4) !important;
    }
    
    /* Download Buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, var(--success) 0%, #22d3ee 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 14px 24px !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.3) !important;
        width: 100% !important;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 20px 40px rgba(16, 185, 129, 0.4) !important;
    }
    
    /* Input Elements */
    .stSelectbox > div > div,
    .stDateInput > div > div,
    .stMultiSelect > div > div {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 14px !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Alert Cards */
    .alert-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 16px;
        padding: 20px 24px;
        margin: 16px 0;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .alert-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary), var(--success));
    }
    
    /* Performance Ring */
    .performance-ring {
        width: 140px;
        height: 140px;
        border-radius: 50%;
        background: conic-gradient(from 0deg, var(--success) 0%, var(--success) var(--progress, 70%), rgba(255,255,255,0.3) var(--progress, 70%) 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        margin: 0 auto 20px auto;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
    }
    
    .performance-ring::before {
        content: '';
        width: 100px;
        height: 100px;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 50%;
        position: absolute;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .performance-score {
        position: relative;
        z-index: 1;
        font-size: 1.8rem;
        font-weight: 900;
        color: var(--text-primary);
        text-align: center;
    }
    
    /* Insight Cards */
    .insight-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 16px;
        padding: 20px;
        margin: 12px 0;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .insight-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
    }
    
    .insight-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, var(--primary), var(--secondary));
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 4px 8px 4px 0;
        backdrop-filter: blur(10px);
    }
    
    .status-success {
        background: rgba(16, 185, 129, 0.1);
        color: var(--success);
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    
    .status-warning {
        background: rgba(245, 158, 11, 0.1);
        color: var(--warning);
        border: 1px solid rgba(245, 158, 11, 0.2);
    }
    
    .status-danger {
        background: rgba(239, 68, 68, 0.1);
        color: var(--danger);
        border: 1px solid rgba(239, 68, 68, 0.2);
    }
    
    /* Floating Animation */
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .floating {
        animation: float 6s ease-in-out infinite;
    }
    
    /* Hide Sidebar */
    [data-testid='stSidebar'] { display: none !important; }
    
    /* Enhanced Typography */
    .metric-title {
        font-size: 1rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .metric-value-large {
        font-size: 2.8rem;
        font-weight: 900;
        line-height: 1;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-subtitle {
        font-size: 0.85rem;
        color: var(--text-muted);
        font-weight: 500;
        margin-top: 4px;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
        }
        
        .hero-header {
            padding: 20px;
            margin: -1rem -1rem 2rem -1rem;
        }
        
        [data-testid="metric-container"] {
            padding: 20px 16px !important;
        }
        
        [data-testid="metric-container"] [data-testid="metric-value"] {
            font-size: 2rem !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# =========================
# DATA PROCESSING
# =========================
class DataProcessor:
    """Handles all data processing operations"""
    
    @staticmethod
    def clean_numeric(value: Any) -> float:
        """Convert various formats to numeric values"""
        if isinstance(value, (int, float, np.number)):
            return float(value)
        if value is None or (isinstance(value, str) and not value.strip()):
            return np.nan
        
        s = str(value).strip()
        s = re.sub(r'[^\d\-\.\,()]', '', s)
        
        negative = s.startswith('(') and s.endswith(')')
        if negative:
            s = s[1:-1]
        
        s = s.replace(',', '')
        
        try:
            val = float(s)
            return -val if negative else val
        except:
            return np.nan
    
    @staticmethod
    def find_column(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
        """Find column name from list of possible aliases"""
        columns_lower = {col.lower().strip(): col for col in df.columns}
        
        for alias in aliases:
            if alias in columns_lower:
                return columns_lower[alias]
        
        for col_lower, col_original in columns_lower.items():
            for alias in aliases:
                if alias in col_lower:
                    return col_original
        
        return None
    
    @classmethod
    def standardize_dataframe(cls, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Standardize column names and data types"""
        if df.empty:
            return df.copy(), {}
        
        df = df.copy()
        mapping = {}
        
        for target_col, aliases in config.COLUMN_MAPPINGS.items():
            source_col = cls.find_column(df, aliases)
            
            if source_col:
                mapping[target_col] = source_col
                df.rename(columns={source_col: target_col}, inplace=True)
                
                if target_col == "Date":
                    df[target_col] = pd.to_datetime(df[target_col], errors='coerce', dayfirst=True)
                elif target_col in ["Sales", "TargetSales", "COGS", "Qty", "Tickets", "InventoryQty", 
                                  "InventoryValue", "BankBalance", "CashBalance", "Expenses", "CustomerCount",
                                  "Discount", "Tax", "UnitPrice", "UnitCost"]:
                    df[target_col] = df[target_col].apply(cls.clean_numeric)
                    if target_col in ["Sales", "Qty"]:
                        df[target_col] = df[target_col].fillna(0)
            else:
                mapping[target_col] = "(missing)"
                if target_col in ["Sales", "Qty"]:
                    df[target_col] = 0
                else:
                    df[target_col] = np.nan
        
        # Calculate derived metrics
        if df["COGS"].notna().any():
            df["GrossProfit"] = df["Sales"] - df["COGS"]
            df["GrossMargin"] = np.where(df["Sales"] > 0, (df["GrossProfit"] / df["Sales"]) * 100, 0)
        else:
            df["GrossProfit"] = np.nan
            df["GrossMargin"] = np.nan
        
        if df["TargetSales"].notna().any():
            df["SalesVariance"] = df["Sales"] - df["TargetSales"]
            df["SalesVariancePct"] = np.where(df["TargetSales"] > 0, 
                                           (df["SalesVariance"] / df["TargetSales"]) * 100, 0)
        else:
            df["SalesVariance"] = np.nan
            df["SalesVariancePct"] = np.nan
        
        df["Store"] = df["Store"].fillna("Unknown").astype(str).str.strip()
        
        return df, mapping

# =========================
# ANALYTICS ENGINE
# =========================
class AnalyticsEngine:
    """Advanced analytics and KPI calculations"""
    
    @staticmethod
    def calculate_kpis(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive KPIs"""
        kpis = {}
        
        # Core metrics
        kpis["total_sales"] = float(df["Sales"].sum())
        kpis["total_tickets"] = float(df["Tickets"].sum()) if df["Tickets"].notna().any() else 0
        kpis["total_qty"] = float(df["Qty"].sum())
        kpis["unique_customers"] = float(df["CustomerCount"].sum()) if df["CustomerCount"].notna().any() else 0
        
        # Financial metrics
        kpis["target_sales"] = float(df["TargetSales"].sum()) if df["TargetSales"].notna().any() else 0
        kpis["sales_variance"] = kpis["total_sales"] - kpis["target_sales"]
        kpis["sales_variance_pct"] = (kpis["sales_variance"] / kpis["target_sales"] * 100) if kpis["target_sales"] > 0 else 0
        
        kpis["gross_profit"] = float(df["GrossProfit"].sum()) if df["GrossProfit"].notna().any() else 0
        kpis["gross_margin"] = (kpis["gross_profit"] / kpis["total_sales"] * 100) if kpis["total_sales"] > 0 else 0
        
        kpis["total_discount"] = float(df["Discount"].sum()) if df["Discount"].notna().any() else 0
        kpis["discount_rate"] = (kpis["total_discount"] / kpis["total_sales"] * 100) if kpis["total_sales"] > 0 else 0
        
        # Operational metrics
        kpis["avg_basket"] = kpis["total_sales"] / kpis["total_tickets"] if kpis["total_tickets"] > 0 else 0
        kpis["items_per_basket"] = kpis["total_qty"] / kpis["total_tickets"] if kpis["total_tickets"] > 0 else 0
        kpis["conversion_rate"] = kpis["items_per_basket"]  # Items per transaction
        
        # Time-based metrics
        if df["Date"].notna().any():
            unique_days = df["Date"].dt.date.nunique()
            kpis["days_count"] = unique_days
            kpis["avg_daily_sales"] = kpis["total_sales"] / unique_days if unique_days > 0 else 0
            kpis["avg_daily_tickets"] = kpis["total_tickets"] / unique_days if unique_days > 0 else 0
            
            # Growth calculation (compare first half vs second half of period)
            if unique_days > 7:
                dates_sorted = df["Date"].dt.date.unique()
                dates_sorted = sorted(dates_sorted)
                mid_point = len(dates_sorted) // 2
                
                first_half = df[df["Date"].dt.date <= dates_sorted[mid_point]]["Sales"].sum()
                second_half = df[df["Date"].dt.date > dates_sorted[mid_point]]["Sales"].sum()
                
                if first_half > 0:
                    kpis["growth_rate"] = ((second_half - first_half) / first_half * 100)
                else:
                    kpis["growth_rate"] = 0
            else:
                kpis["growth_rate"] = 0
        else:
            kpis["days_count"] = 0
            kpis["avg_daily_sales"] = 0
            kpis["avg_daily_tickets"] = 0
            kpis["growth_rate"] = 0
        
        # Liquidity metrics
        kpis["bank_balance"] = float(df["BankBalance"].iloc[-1]) if df["BankBalance"].notna().any() else 0
        kpis["cash_balance"] = float(df["CashBalance"].iloc[-1]) if df["CashBalance"].notna().any() else 0
        kpis["total_liquidity"] = kpis["bank_balance"] + kpis["cash_balance"]
        
        # Performance score
        kpis["performance_score"] = AnalyticsEngine.calculate_performance_score(kpis)
        
        return kpis
    
    @staticmethod
    def calculate_performance_score(kpis: Dict[str, Any]) -> float:
        """Calculate weighted performance score"""
        score = 0
        weights = 0
        
        # Sales performance (30%)
        if kpis.get("target_sales", 0) > 0:
            target_ratio = min(kpis["total_sales"] / kpis["target_sales"], 1.5)
            score += target_ratio * 30
            weights += 30
        elif kpis.get("total_sales", 0) > 0:
            score += 25  # Base score for having sales
            weights += 30
        
        # Profitability (25%)
        if kpis.get("gross_margin", 0) > 0:
            margin_score = min(kpis["gross_margin"] / 30, 1.0)
            score += margin_score * 25
            weights += 25
        
        # Growth (20%)
        if abs(kpis.get("growth_rate", 0)) > 0:
            growth_score = min(abs(kpis["growth_rate"]) / 10, 1.0)
            if kpis["growth_rate"] > 0:
                score += growth_score * 20
            else:
                score += growth_score * 10  # Partial credit for activity
            weights += 20
        
        # Basket metrics (15%)
        if kpis.get("avg_basket", 0) > 0:
            basket_score = min(kpis["avg_basket"] / 50, 1.0)
            score += basket_score * 15
            weights += 15
        
        # Operational efficiency (10%)
        if kpis.get("items_per_basket", 0) > 0:
            efficiency_score = min(kpis["items_per_basket"] / 3, 1.0)
            score += efficiency_score * 10
            weights += 10
        
        return (score / weights * 100) if weights > 0 else 0

# =========================
# VISUALIZATION ENGINE
# =========================
class VisualizationEngine:
    """Create stunning visualizations"""
    
    @staticmethod
    def create_sales_trend_chart(df: pd.DataFrame) -> go.Figure:
        """Beautiful sales trend with multiple metrics"""
        if not df["Date"].notna().any():
            return go.Figure()
        
        daily_data = df.groupby(df["Date"].dt.date).agg({
            "Sales": "sum",
            "Tickets": "sum",
            "Qty": "sum",
            "GrossProfit": "sum"
        }).reset_index()
        
        daily_data["AvgBasket"] = daily_data["Sales"] / daily_data["Tickets"]
        daily_data["ItemsPerBasket"] = daily_data["Qty"] / daily_data["Tickets"]
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Sales Performance', 'Basket Analytics'),
            vertical_spacing=0.15,
            specs=[[{"secondary_y": True}], [{"secondary_y": True}]]
        )
        
        # Sales bars
        fig.add_trace(
            go.Bar(
                x=daily_data["Date"],
                y=daily_data["Sales"],
                name="Daily Sales",
                marker=dict(
                    color=daily_data["Sales"],
                    colorscale="Blues",
                    showscale=False
                ),
                text=daily_data["Sales"].round(0),
                textposition="outside"
            ),
            row=1, col=1
        )
        
        # Gross profit line
        fig.add_trace(
            go.Scatter(
                x=daily_data["Date"],
                y=daily_data["GrossProfit"],
                name="Gross Profit",
                line=dict(color="#10b981", width=4),
                mode="lines+markers",
                marker=dict(size=8)
            ),
            row=1, col=1, secondary_y=True
        )
        
        # Average basket
        fig.add_trace(
            go.Scatter(
                x=daily_data["Date"],
                y=daily_data["AvgBasket"],
                name="Avg Basket Size",
                line=dict(color="#f59e0b", width=3),
                mode="lines+markers",
                fill="tonexty"
            ),
            row=2, col=1
        )
        
        # Items per basket
        fig.add_trace(
            go.Scatter(
                x=daily_data["Date"],
                y=daily_data["ItemsPerBasket"],
                name="Items per Basket",
                line=dict(color="#8b5cf6", width=3),
                mode="lines+markers"
            ),
            row=2, col=1, secondary_y=True
        )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", size=12)
        )
        
        # Update axes
        fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.2)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.2)")
        
        return fig
    
    @staticmethod
    def create_category_sunburst(df: pd.DataFrame) -> go.Figure:
        """Create beautiful sunburst chart for categories and items"""
        if not df["Category"].notna().any():
            return go.Figure()
        
        # Prepare data for sunburst
        category_data = df.groupby(["Category", "Item"])["Sales"].sum().reset_index()
        category_data = category_data.sort_values("Sales", ascending=False)
        
        # Limit items per category for clarity
        top_categories = category_data.groupby("Category")["Sales"].sum().nlargest(6).index
        filtered_data = category_data[category_data["Category"].isin(top_categories)]
        
        # Create sunburst
        fig = go.Figure(go.Sunburst(
            labels=["Total"] + list(filtered_data["Category"].unique()) + list(filtered_data["Item"]),
            parents=[""] + ["Total"] * len(filtered_data["Category"].unique()) + list(filtered_data["Category"]),
            values=[filtered_data["Sales"].sum()] + list(filtered_data.groupby("Category")["Sales"].sum()) + list(filtered_data["Sales"]),
            branchvalues="total",
            hovertemplate="<b>%{label}</b><br>Sales: SAR %{value:,.0f}<br>Percentage: %{percentParent}<extra></extra>",
            maxdepth=2
        ))
        
        fig.update_layout(
            title="Sales Distribution: Categories & Top Items",
            font=dict(family="Inter", size=12),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_performance_gauge(score: float) -> go.Figure:
        """Create beautiful performance gauge"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            title={"text": "Performance Score", "font": {"size": 20, "family": "Inter"}},
            delta={"reference": 70, "increasing": {"color": "#10b981"}, "decreasing": {"color": "#ef4444"}},
            gauge={
                "axis": {"range": [None, 100], "tickwidth": 1, "tickcolor": "white"},
                "bar": {"color": "#3b82f6"},
                "bgcolor": "rgba(255,255,255,0.1)",
                "borderwidth": 2,
                "bordercolor": "white",
                "steps": [
                    {"range": [0, 50], "color": "rgba(239, 68, 68, 0.3)"},
                    {"range": [50, 75], "color": "rgba(245, 158, 11, 0.3)"},
                    {"range": [75, 100], "color": "rgba(16, 185, 129, 0.3)"}
                ],
                "threshold": {
                    "line": {"color": "white", "width": 4},
                    "thickness": 0.75,
                    "value": 90
                }
            }
        ))
        
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "white", "family": "Inter"},
            height=300
        )
        
        return fig

# =========================
# TAB FUNCTIONS
# =========================
def render_dashboard_overview(df: pd.DataFrame, kpis: Dict[str, Any]):
    """Render beautiful overview dashboard"""
    
    # Hero metrics section
    st.markdown("### 💰 Financial Performance")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "💰 Total Sales",
            f"SAR {kpis['total_sales']:,.0f}",
            delta=f"Daily: SAR {kpis['avg_daily_sales']:,.0f}"
        )
    
    with col2:
        variance_delta = f"{kpis['sales_variance_pct']:+.1f}%" if kpis['target_sales'] > 0 else "No Target"
        variance_color = "normal" if kpis['sales_variance_pct'] >= 0 else "inverse"
        st.metric(
            "🎯 vs Target", 
            f"SAR {kpis['sales_variance']:,.0f}",
            delta=variance_delta,
            delta_color=variance_color
        )
    
    with col3:
        st.metric(
            "💎 Gross Profit",
            f"SAR {kpis['gross_profit']:,.0f}",
            delta=f"Margin: {kpis['gross_margin']:.1f}%"
        )
    
    with col4:
        discount_color = "inverse" if kpis['discount_rate'] > 10 else "normal"
        st.metric(
            "🏷️ Discounts",
            f"SAR {kpis['total_discount']:,.0f}",
            delta=f"Rate: {kpis['discount_rate']:.1f}%",
            delta_color=discount_color
        )
    
    with col5:
        growth_color = "normal" if kpis['growth_rate'] >= 0 else "inverse"
        st.metric(
            "📈 Growth",
            f"{kpis['growth_rate']:+.1f}%",
            delta="Period Growth",
            delta_color=growth_color
        )
    
    # Operational metrics
    st.markdown("### 🛍️ Customer & Operations")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "🛒 Total Baskets",
            f"{kpis['total_tickets']:,.0f}",
            delta=f"Daily: {kpis['avg_daily_tickets']:.0f}"
        )
    
    with col2:
        st.metric(
            "🎯 Avg Basket",
            f"SAR {kpis['avg_basket']:.2f}",
            delta="Per Transaction"
        )
    
    with col3:
        st.metric(
            "📦 Items/Basket",
            f"{kpis['items_per_basket']:.2f}",
            delta="Conversion Rate"
        )
    
    with col4:
        st.metric(
            "👥 Customers",
            f"{kpis['unique_customers']:,.0f}",
            delta="Total Served"
        )
    
    with col5:
        st.metric(
            "📊 Days Active",
            f"{kpis['days_count']}",
            delta="Analysis Period"
        )
    
    # Performance visualization
    st.markdown("### 📈 Performance Analytics")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if df["Date"].notna().any():
            trend_chart = VisualizationEngine.create_sales_trend_chart(df)
            st.plotly_chart(trend_chart, use_container_width=True)
    
    with col2:
        # Performance gauge
        gauge_chart = VisualizationEngine.create_performance_gauge(kpis["performance_score"])
        st.plotly_chart(gauge_chart, use_container_width=True)
        
        # Key insights
        st.markdown("""
        <div class="insight-card">
            <h4 style="margin:0 0 12px 0; color: var(--text-primary);">💡 Quick Insights</h4>
        """, unsafe_allow_html=True)
        
        if kpis['performance_score'] >= 80:
            st.markdown("🎉 **Excellent** performance across all metrics")
        elif kpis['performance_score'] >= 60:
            st.markdown("👍 **Good** performance with room for optimization")
        else:
            st.markdown("⚠️ **Focus needed** on key performance areas")
        
        if kpis['growth_rate'] > 5:
            st.markdown(f"📈 Strong growth momentum: **+{kpis['growth_rate']:.1f}%**")
        elif kpis['growth_rate'] < -5:
            st.markdown(f"📉 Monitor declining trend: **{kpis['growth_rate']:.1f}%**")
        
        if kpis['items_per_basket'] > 2.5:
            st.markdown(f"🛍️ Excellent cross-selling: **{kpis['items_per_basket']:.1f}** items/basket")
        
        st.markdown("</div>", unsafe_allow_html=True)

def render_analytics_tab(df: pd.DataFrame, kpis: Dict[str, Any]):
    """Render detailed analytics"""
    st.markdown("## 📊 Advanced Analytics")
    
    if df["Category"].notna().any():
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Category performance
            category_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
            
            fig = go.Figure(go.Bar(
                y=category_sales.index,
                x=category_sales.values,
                orientation='h',
                marker=dict(
                    color=category_sales.values,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Sales (SAR)")
                ),
                text=[f"SAR {val:,.0f}" for val in category_sales.values],
                textposition="outside"
            ))
            
            fig.update_layout(
                title="Category Performance Ranking",
                xaxis_title="Sales (SAR)",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=400,
                font=dict(family="Inter")
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sunburst chart
            sunburst = VisualizationEngine.create_category_sunburst(df)
            st.plotly_chart(sunburst, use_container_width=True)
    
    # Store comparison if multiple stores
    if df["Store"].nunique() > 1:
        st.markdown("### 🏪 Store Performance Matrix")
        
        store_analysis = df.groupby("Store").agg({
            "Sales": "sum",
            "Tickets": "sum",
            "Qty": "sum",
            "GrossProfit": "sum"
        }).reset_index()
        
        store_analysis["AvgBasket"] = store_analysis["Sales"] / store_analysis["Tickets"]
        store_analysis["GrossMargin"] = (store_analysis["GrossProfit"] / store_analysis["Sales"] * 100).round(1)
        store_analysis["ItemsPerBasket"] = (store_analysis["Qty"] / store_analysis["Tickets"]).round(2)
        
        # Create heatmap-style comparison
        fig = go.Figure()
        
        # Sales comparison
        fig.add_trace(go.Bar(
            x=store_analysis["Store"],
            y=store_analysis["Sales"],
            name="Sales",
            marker_color="#3b82f6",
            text=[f"SAR {val:,.0f}" for val in store_analysis["Sales"]],
            textposition="outside"
        ))
        
        fig.update_layout(
            title="Store Sales Comparison",
            xaxis_title="Store",
            yaxis_title="Sales (SAR)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Store metrics table
        display_stores = store_analysis.copy()
        display_stores["Sales"] = display_stores["Sales"].apply(lambda x: f"SAR {x:,.0f}")
        display_stores["GrossProfit"] = display_stores["GrossProfit"].apply(lambda x: f"SAR {x:,.0f}")
        display_stores["AvgBasket"] = display_stores["AvgBasket"].apply(lambda x: f"SAR {x:.2f}")
        
        st.dataframe(display_stores, use_container_width=True, hide_index=True)

def render_products_tab(df: pd.DataFrame):
    """Render product analytics"""
    st.markdown("## ⭐ Product Intelligence")
    
    if df["Item"].notna().any():
        # Product analysis
        product_analysis = df.groupby("Item").agg({
            "Sales": "sum",
            "Qty": "sum",
            "Tickets": "sum",
            "GrossProfit": "sum"
        }).reset_index()
        
        product_analysis["AvgPrice"] = product_analysis["Sales"] / product_analysis["Qty"]
        product_analysis["Frequency"] = product_analysis["Qty"] / product_analysis["Tickets"]
        product_analysis["MarginPct"] = (product_analysis["GrossProfit"] / product_analysis["Sales"] * 100).round(1)
        
        # Analysis selector
        col1, col2 = st.columns([3, 1])
        
        with col2:
            analysis_type = st.selectbox(
                "Analysis View",
                ["🏆 Top Sellers", "💰 Most Profitable", "📈 Best Margins", "🔥 High Frequency"]
            )
        
        with col1:
            if "Top Sellers" in analysis_type:
                display_data = product_analysis.sort_values("Sales", ascending=False).head(20)
                chart_data = display_data.head(10)
                
                fig = go.Figure(go.Bar(
                    x=chart_data["Sales"],
                    y=chart_data["Item"].str[:30],
                    orientation='h',
                    marker=dict(
                        color=chart_data["Sales"],
                        colorscale="Blues",
                        showscale=True
                    ),
                    text=[f"SAR {val:,.0f}" for val in chart_data["Sales"]],
                    textposition="outside"
                ))
                
                fig.update_layout(
                    title="Top 10 Products by Sales",
                    xaxis_title="Sales (SAR)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif "Most Profitable" in analysis_type:
                display_data = product_analysis.sort_values("GrossProfit", ascending=False).head(20)
                chart_data = display_data.head(10)
                
                fig = go.Figure(go.Bar(
                    x=chart_data["GrossProfit"],
                    y=chart_data["Item"].str[:30],
                    orientation='h',
                    marker=dict(
                        color=chart_data["GrossProfit"],
                        colorscale="Greens",
                        showscale=True
                    ),
                    text=[f"SAR {val:,.0f}" for val in chart_data["GrossProfit"]],
                    textposition="outside"
                ))
                
                fig.update_layout(
                    title="Top 10 Most Profitable Products",
                    xaxis_title="Gross Profit (SAR)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif "Best Margins" in analysis_type:
                display_data = product_analysis[product_analysis["MarginPct"] > 0].sort_values("MarginPct", ascending=False).head(20)
                chart_data = display_data.head(10)
                
                fig = go.Figure(go.Bar(
                    x=chart_data["MarginPct"],
                    y=chart_data["Item"].str[:30],
                    orientation='h',
                    marker=dict(
                        color=chart_data["MarginPct"],
                        colorscale="Oranges",
                        showscale=True
                    ),
                    text=[f"{val:.1f}%" for val in chart_data["MarginPct"]],
                    textposition="outside"
                ))
                
                fig.update_layout(
                    title="Top 10 Products by Margin",
                    xaxis_title="Gross Margin (%)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            else:  # High Frequency
                display_data = product_analysis.sort_values("Frequency", ascending=False).head(20)
                chart_data = display_data.head(10)
                
                fig = go.Figure(go.Bar(
                    x=chart_data["Frequency"],
                    y=chart_data["Item"].str[:30],
                    orientation='h',
                    marker=dict(
                        color=chart_data["Frequency"],
                        colorscale="Purples",
                        showscale=True
                    ),
                    text=[f"{val:.2f}" for val in chart_data["Frequency"]],
                    textposition="outside"
                ))
                
                fig.update_layout(
                    title="Top 10 High-Frequency Products",
                    xaxis_title="Items per Basket",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Product table
        st.markdown("### 📋 Detailed Product Analysis")
        
        # Format display data
        display_products = display_data.copy()
        display_products["Sales"] = display_products["Sales"].apply(lambda x: f"SAR {x:,.0f}")
        display_products["GrossProfit"] = display_products["GrossProfit"].apply(lambda x: f"SAR {x:,.0f}")
        display_products["AvgPrice"] = display_products["AvgPrice"].apply(lambda x: f"SAR {x:.2f}")
        display_products["Frequency"] = display_products["Frequency"].apply(lambda x: f"{x:.2f}")
        
        # Select relevant columns
        cols_to_show = ["Item", "Sales", "Qty", "GrossProfit", "MarginPct", "AvgPrice", "Frequency"]
        col_names = ["Product", "Sales", "Qty", "Profit", "Margin %", "Avg Price", "Frequency"]
        
        display_products = display_products[cols_to_show]
        display_products.columns = col_names
        
        st.dataframe(display_products, use_container_width=True)

def render_insights_tab(df: pd.DataFrame, kpis: Dict[str, Any]):
    """Render AI insights and recommendations"""
    st.markdown("## 💡 Business Intelligence & Recommendations")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Performance Analysis")
        
        # Performance insights
        insights = []
        
        # Sales performance
        if kpis['sales_variance_pct'] > 10:
            insights.append(("🎉 Outstanding Performance", f"Sales exceeded target by {kpis['sales_variance_pct']:.1f}%", "success"))
        elif kpis['sales_variance_pct'] > 0:
            insights.append(("✅ Target Achieved", f"Sales above target by {kpis['sales_variance_pct']:.1f}%", "success"))
        elif kpis['sales_variance_pct'] > -10:
            insights.append(("⚠️ Near Target", f"Sales {abs(kpis['sales_variance_pct']):.1f}% below target", "warning"))
        else:
            insights.append(("🔴 Action Required", f"Sales {abs(kpis['sales_variance_pct']):.1f}% below target", "danger"))
        
        # Profitability
        if kpis['gross_margin'] > 30:
            insights.append(("💰 Strong Margins", f"Gross margin at {kpis['gross_margin']:.1f}%", "success"))
        elif kpis['gross_margin'] > 20:
            insights.append(("📊 Healthy Margins", f"Gross margin at {kpis['gross_margin']:.1f}%", "success"))
        else:
            insights.append(("⚠️ Margin Concern", f"Low gross margin: {kpis['gross_margin']:.1f}%", "warning"))
        
        # Customer behavior
        if kpis['items_per_basket'] > 3:
            insights.append(("🛍️ Excellent Cross-selling", f"{kpis['items_per_basket']:.1f} items per basket", "success"))
        elif kpis['items_per_basket'] > 2:
            insights.append(("👍 Good Basket Size", f"{kpis['items_per_basket']:.1f} items per basket", "success"))
        else:
            insights.append(("🎯 Upselling Opportunity", f"Only {kpis['items_per_basket']:.1f} items per basket", "warning"))
        
        # Growth analysis
        if kpis['growth_rate'] > 10:
            insights.append(("🚀 Rapid Growth", f"Strong growth trend: +{kpis['growth_rate']:.1f}%", "success"))
        elif kpis['growth_rate'] > 0:
            insights.append(("📈 Positive Growth", f"Steady growth: +{kpis['growth_rate']:.1f}%", "success"))
        elif kpis['growth_rate'] > -5:
            insights.append(("📊 Stable Period", f"Minor fluctuation: {kpis['growth_rate']:.1f}%", "warning"))
        else:
            insights.append(("📉 Declining Trend", f"Negative growth: {kpis['growth_rate']:.1f}%", "danger"))
        
        # Display insights
        for title, description, status in insights:
            status_class = f"status-{status}"
            st.markdown(f"""
            <div class="insight-card">
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <span class="status-badge {status_class}">{title}</span>
                </div>
                <p style="margin: 0; color: var(--text-secondary); font-weight: 500;">{description}</p>
            </div>
            """, unsafe_allow_html=True
