# app.py — Enhanced Alkhair Family Market Dashboard with Multiple Data Sources
# Supports multiple Google Sheets and Excel files for comprehensive reporting

import io
import re
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass
from functools import lru_cache

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
    
    # Multiple data sources
    DATA_SOURCES = {
        "primary": "https://docs.google.com/spreadsheets/d/e/2PACX-1vSW_ui1m_393ipZv8NAliu1rly6zeifFxMfOWpQF17hjVIDa9Ll8PiGCaz8gTRkMQ/pub?output=csv",
        "dsr_report": "https://docs.google.com/spreadsheets/d/e/2PACX-1vQG7boLWl2bNLCPR05NXv6EFpPPcFfXsiXPQ7rAGYr3q8Nkc2Ijg8BqEwVofcMLSg/pub?output=csv"
    }
    
    # Enhanced column aliases for comprehensive data mapping
    COLUMN_MAPPINGS = {
        "Date": ["date", "bill_date", "txn_date", "invoice_date", "posting_date", "transaction_date", "sale_date"],
        "Store": ["store", "branch", "location", "shop", "outlet", "store_name"],
        "Sales": ["sales", "amount", "net_sales", "revenue", "netamount", "net_amount", "actual_sales", "sale_amount", "total_sales"],
        "TargetSales": ["target_sales", "target", "sales_target", "budget_sales", "planned_sales", "target_amount"],
        "COGS": ["cogs", "cost", "purchase_cost", "cost_of_goods", "cost_of_sales"],
        "Qty": ["qty", "quantity", "qty_sold", "quantity_sold", "units", "pcs", "pieces", "units_sold", "qnty", "qnt", "qty.", "total_qty"],
        "Tickets": ["tickets", "bills", "invoice_count", "transactions", "bills_count", "baskets", "no_of_baskets", "bill_count"],
        "Category": ["category", "cat", "product_category", "item_category"],
        "Item": ["item", "product", "sku", "item_name", "product_name", "description"],
        "InventoryQty": ["inventoryqty", "inventory_qty", "stock_qty", "stockqty", "stock_quantity"],
        "InventoryValue": ["inventoryvalue", "inventory_value", "stock_value", "stockvalue"],
        "BankBalance": ["bank_balance", "bank", "bank_amount", "bankbalance", "account_balance"],
        "CashBalance": ["cash_balance", "cash", "cash_amount", "cashbalance", "petty_cash"],
        "Expenses": ["expenses", "operating_expenses", "opex", "costs", "expenditure", "expense_amount"],
        "CustomerCount": ["customer_count", "customers", "footfall", "visitors", "customer_visits"],
        "Supplier": ["supplier", "vendor", "supplier_name", "vendor_name"],
        "PaymentMethod": ["payment_method", "payment_type", "payment", "pay_method"],
        "Employee": ["employee", "staff", "cashier", "user", "operator"],
        "Discount": ["discount", "discount_amount", "discount_value", "promo"],
        "Tax": ["tax", "vat", "tax_amount", "sales_tax"],
        "Profit": ["profit", "net_profit", "gross_profit", "margin_amount"],
        "UnitPrice": ["unit_price", "price", "rate", "selling_price", "sp"],
        "UnitCost": ["unit_cost", "cost_price", "cp", "purchase_price"]
    }
    
    # Enhanced KPI colors for tiles
    KPI_COLORS = {
        "tickets": "#a7f3d0",
        "sales": "#fde68a", 
        "profit": "#d9f99d",
        "conversion": "#bae6fd",
        "avg_sales": "#fed7aa",
        "bank": "#c7d2fe",
        "cash": "#fce7f3",
        "target": "#f3e8ff",
        "liquidity": "#ddd6fe",
        "variance": "#fef3c7",
        "growth": "#bbf7d0",
        "margin": "#fecaca",
        "customer": "#e0e7ff"
    }
    
    # Target thresholds for performance indicators
    TARGET_THRESHOLDS = {
        "conversion_rate": 2.5,  # Target conversion rate %
        "gross_margin": 30.0,    # Target gross margin %
        "liquidity_ratio": 2.0,  # Target liquidity ratio
        "growth_rate": 5.0,      # Target growth rate %
        "customer_retention": 80.0  # Target customer retention %
    }

# Initialize configuration
config = Config()

# Page setup
st.set_page_config(
    page_title=config.PAGE_TITLE,
    layout=config.LAYOUT,
    initial_sidebar_state="collapsed",
)

# =========================
# STYLING
# =========================
def apply_custom_css():
    """Apply custom CSS styling for the enhanced dashboard"""
    st.markdown("""
    <style>
    :root {
        --bg: #ffffff;
        --card: #ffffff;
        --ink: #1f2937;
        --ink-weak: #6b7280;
        --brand: #0ea5e9;
        --green: #22c55e;
        --amber: #f59e0b;
        --orange: #fb923c;
        --blue: #3b82f6;
        --red: #ef4444;
        --purple: #8b5cf6;
        --shadow: 0 8px 22px rgba(0,0,0,.08);
        --radius: 16px;
        --border: #e5e7eb;
    }
    
    html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg);
        color: var(--ink);
        font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    .card {
        background: var(--card);
        border-radius: var(--radius);
        box-shadow: var(--shadow);
        padding: 18px;
        border: 1px solid var(--border);
        margin-bottom: 1rem;
    }
    
    .brandbar {
        position: sticky;
        top: 0;
        z-index: 20;
        padding: 15px 0;
        margin-bottom: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-bottom: 1px solid var(--border);
        color: white;
        border-radius: 12px;
    }
    
    .data-source-indicator {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 8px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-left: 8px;
        background: rgba(255, 255, 255, 0.2);
        color: white;
    }
    
    .kpi-tile {
        border-radius: 14px;
        padding: 16px 18px;
        color: #0b1220;
        font-weight: 800;
        box-shadow: var(--shadow);
        transition: transform 0.2s;
        position: relative;
        overflow: hidden;
        margin-bottom: 15px;
    }
    
    .kpi-tile:hover {
        transform: translateY(-2px);
    }
    
    .kpi-tile::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    }
    
    .metric-trend {
        font-size: 0.9rem;
        margin-top: 4px;
    }
    
    .trend-up { color: #22c55e; }
    .trend-down { color: #ef4444; }
    .trend-neutral { color: #6b7280; }
    
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1f2937;
        margin: 15px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #e5e7eb;
    }
    
    .performance-indicator {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 8px;
    }
    
    .indicator-good { background: #dcfce7; color: #166534; }
    .indicator-warning { background: #fef3c7; color: #92400e; }
    .indicator-poor { background: #fee2e2; color: #991b1b; }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        background-color: transparent;
        border-radius: 8px;
        color: #6b7280;
        font-weight: 600;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: #1f2937 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    [data-testid='stSidebar'] { display: none !important; }
    
    .data-source-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border: 2px solid #cbd5e1;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }
    
    .alert-info {
        background: #dbeafe;
        border: 1px solid #3b82f6;
        border-radius: 8px;
        padding: 12px;
        color: #1e40af;
        margin: 8px 0;
    }
    
    .alert-success {
        background: #dcfce7;
        border: 1px solid #22c55e;
        border-radius: 8px;
        padding: 12px;
        color: #166534;
        margin: 8px 0;
    }
    
    .alert-warning {
        background: #fef3c7;
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: 12px;
        color: #92400e;
        margin: 8px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# =========================
# DATA PROCESSING UTILITIES
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
        
        # Clean string values
        s = str(value).strip()
        s = re.sub(r'[^\d\-\.\,()]', '', s)
        
        # Handle parentheses as negative
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
        
        # Exact match
        for alias in aliases:
            if alias in columns_lower:
                return columns_lower[alias]
        
        # Fuzzy match
        for col_lower, col_original in columns_lower.items():
            for alias in aliases:
                if alias in col_lower:
                    return col_original
        
        return None
    
    @staticmethod
    def convert_gsheet_url(url: str) -> str:
        """Convert Google Sheets URL to CSV export URL"""
        if "docs.google.com/spreadsheets" not in url:
            return url
        if "/spreadsheets/d/e/" in url and "output=csv" in url:
            return url
        
        match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
        if not match:
            return url
        
        sheet_id = match.group(1)
        gid_match = re.search(r"gid=(\d+)", url)
        gid = gid_match.group(1) if gid_match else "0"
        
        return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    
    @classmethod
    def load_multiple_sources(cls, sources: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load and combine data from multiple sources"""
        combined_data = pd.DataFrame()
        source_info = {}
        
        for source_name, url in sources.items():
            try:
                csv_url = cls.convert_gsheet_url(url)
                df = pd.read_csv(csv_url)
                
                # Add source identifier
                df['DataSource'] = source_name
                
                # Store source info
                source_info[source_name] = {
                    'url': url,
                    'rows': len(df),
                    'columns': list(df.columns),
                    'status': 'success'
                }
                
                # Append to combined data
                if combined_data.empty:
                    combined_data = df.copy()
                else:
                    combined_data = pd.concat([combined_data, df], ignore_index=True, sort=False)
                    
            except Exception as e:
                source_info[source_name] = {
                    'url': url,
                    'rows': 0,
                    'columns': [],
                    'status': 'error',
                    'error': str(e)
                }
        
        return combined_data, source_info
    
    @classmethod
    def standardize_dataframe(cls, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Standardize column names and data types"""
        if df.empty:
            return df.copy(), {}
        
        df = df.copy()
        mapping = {}
        
        # Process each column type
        for target_col, aliases in config.COLUMN_MAPPINGS.items():
            source_col = cls.find_column(df, aliases)
            
            if source_col:
                mapping[target_col] = source_col
                df.rename(columns={source_col: target_col}, inplace=True)
                
                # Apply appropriate data type conversion
                if target_col == "Date":
                    df[target_col] = pd.to_datetime(df[target_col], errors='coerce', dayfirst=True)
                elif target_col in ["Sales", "TargetSales", "COGS", "Qty", "Tickets", "InventoryQty", 
                                  "InventoryValue", "BankBalance", "CashBalance", "Expenses", "CustomerCount",
                                  "Discount", "Tax", "Profit", "UnitPrice", "UnitCost"]:
                    df[target_col] = df[target_col].apply(cls.clean_numeric)
                    if target_col in ["Sales", "Qty"]:
                        df[target_col] = df[target_col].fillna(0)
            else:
                mapping[target_col] = "(missing)"
                if target_col in ["Sales", "Qty"]:
                    df[target_col] = 0
                else:
                    df[target_col] = np.nan
        
        # Calculate derived columns
        if df["COGS"].notna().any():
            df["GrossProfit"] = df["Sales"] - df["COGS"]
            df["GrossMargin"] = np.where(df["Sales"] > 0, (df["GrossProfit"] / df["Sales"]) * 100, 0)
        else:
            df["GrossProfit"] = np.nan
            df["GrossMargin"] = np.nan
        
        # Calculate sales variance
        if df["TargetSales"].notna().any():
            df["SalesVariance"] = df["Sales"] - df["TargetSales"]
            df["SalesVariancePct"] = np.where(df["TargetSales"] > 0, 
                                           (df["SalesVariance"] / df["TargetSales"]) * 100, 0)
        else:
            df["SalesVariance"] = np.nan
            df["SalesVariancePct"] = np.nan
        
        # Calculate liquidity metrics
        if df["BankBalance"].notna().any() and df["CashBalance"].notna().any():
            df["TotalLiquidity"] = df["BankBalance"] + df["CashBalance"]
            df["LiquidityRatio"] = np.where(df["Expenses"] > 0, 
                                          df["TotalLiquidity"] / df["Expenses"], np.inf)
        else:
            df["TotalLiquidity"] = np.nan
            df["LiquidityRatio"] = np.nan
        
        # Calculate unit-level metrics
        if df["UnitPrice"].notna().any() and df["UnitCost"].notna().any():
            df["UnitMargin"] = df["UnitPrice"] - df["UnitCost"]
            df["UnitMarginPct"] = np.where(df["UnitPrice"] > 0, 
                                         (df["UnitMargin"] / df["UnitPrice"]) * 100, 0)
        
        # Clean store names
        df["Store"] = df["Store"].fillna("").astype(str).str.strip()
        
        # Handle Qty fallback
        if df["Qty"].sum() == 0 and df["Tickets"].notna().sum() > 0:
            df["Qty"] = df["Tickets"].fillna(0)
            mapping["Qty"] += " (fallback from Tickets)"
        
        return df, mapping

# =========================
# ANALYTICS ENGINE
# =========================
class AnalyticsEngine:
    """Handles all analytics and calculations"""
    
    @staticmethod
    def calculate_kpis(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive key performance indicators"""
        kpis = {}
        
        # Basic sales KPIs
        kpis["total_sales"] = float(df["Sales"].sum())
        kpis["total_tickets"] = float(df["Tickets"].sum()) if df["Tickets"].notna().any() else 0
        kpis["total_qty"] = float(df["Qty"].sum())
        kpis["total_baskets"] = kpis["total_tickets"]  # Same as tickets
        
        # Target vs Actual
        kpis["target_sales"] = float(df["TargetSales"].sum()) if df["TargetSales"].notna().any() else 0
        kpis["actual_sales"] = kpis["total_sales"]
        kpis["sales_variance"] = kpis["actual_sales"] - kpis["target_sales"]
        kpis["sales_variance_pct"] = (kpis["sales_variance"] / kpis["target_sales"] * 100) if kpis["target_sales"] > 0 else 0
        
        # Financial KPIs
        if df["GrossProfit"].notna().any():
            kpis["gross_profit"] = float(df["GrossProfit"].sum())
            kpis["gross_margin"] = (kpis["gross_profit"] / kpis["total_sales"] * 100) if kpis["total_sales"] > 0 else 0
        else:
            kpis["gross_profit"] = 0
            kpis["gross_margin"] = 0
        
        # Discount and Tax KPIs
        kpis["total_discount"] = float(df["Discount"].sum()) if df["Discount"].notna().any() else 0
        kpis["total_tax"] = float(df["Tax"].sum()) if df["Tax"].notna().any() else 0
        kpis["discount_rate"] = (kpis["total_discount"] / kpis["total_sales"] * 100) if kpis["total_sales"] > 0 else 0
        
        # Liquidity KPIs
        kpis["bank_balance"] = float(df["BankBalance"].iloc[-1]) if df["BankBalance"].notna().any() else 0
        kpis["cash_balance"] = float(df["CashBalance"].iloc[-1]) if df["CashBalance"].notna().any() else 0
        kpis["total_liquidity"] = kpis["bank_balance"] + kpis["cash_balance"]
        
        # Operational KPIs
        kpis["avg_basket"] = kpis["total_sales"] / kpis["total_tickets"] if kpis["total_tickets"] > 0 else 0
        kpis["conversion_rate"] = (kpis["total_qty"] / kpis["total_tickets"] * 100) if kpis["total_tickets"] > 0 else 0
        kpis["customer_count"] = float(df["CustomerCount"].sum()) if df["CustomerCount"].notna().any() else 0
        
        # Time-based metrics
        if df["Date"].notna().any():
            unique_days = df["Date"].dt.date.nunique()
            kpis["days_count"] = unique_days
            kpis["avg_daily_sales"] = kpis["total_sales"] / unique_days if unique_days > 0 else 0
            kpis["avg_daily_tickets"] = kpis["total_tickets"] / unique_days if unique_days > 0 else 0
        else:
            kpis["days_count"] = 0
            kpis["avg_daily_sales"] = 0
            kpis["avg_daily_tickets"] = 0
        
        # Data source metrics
        if "DataSource" in df.columns:
            kpis["data_sources"] = df["DataSource"].nunique()
            kpis["source_breakdown"] = df.groupby("DataSource")["Sales"].sum().to_dict()
        
        # Performance indicators
        kpis["performance_score"] = AnalyticsEngine.calculate_performance_score(kpis)
        
        return kpis
    
    @staticmethod
    def calculate_performance_score(kpis: Dict[str, Any]) -> float:
        """Calculate overall performance score based on key metrics"""
        score = 0
        factors = 0
        
        # Sales target achievement (30% weight)
        if kpis.get("target_sales", 0) > 0:
            target_achievement = min(kpis["actual_sales"] / kpis["target_sales"], 1.5)
            score += target_achievement * 30
            factors += 30
        
        # Gross margin (25% weight)
        if kpis.get("gross_margin", 0) > 0:
            margin_score = min(kpis["gross_margin"] / 30, 1.0)  # 30% target margin
            score += margin_score * 25
            factors += 25
        
        # Conversion rate (20% weight)
        if kpis.get("conversion_rate", 0) > 0:
            conversion_score = min(kpis["conversion_rate"] / config.TARGET_THRESHOLDS["conversion_rate"], 1.0)
            score += conversion_score * 20
            factors += 20
        
        # Liquidity (15% weight)
        if kpis.get("total_liquidity", 0) > 0:
            liquidity_score = min(kpis["total_liquidity"] / 100000, 1.0)  # Assuming 100k target
            score += liquidity_score * 15
            factors += 15
        
        # Average basket size (10% weight)
        if kpis.get("avg_basket", 0) > 0:
            basket_score = min(kpis["avg_basket"] / 50, 1.0)  # Assuming 50 SAR target
            score += basket_score * 10
            factors += 10
        
        return (score / factors * 100) if factors > 0 else 0
    
    @staticmethod
    def generate_insights(df: pd.DataFrame, kpis: Dict[str, Any]) -> List[str]:
        """Generate intelligent insights from data"""
        insights = []
        
        # Performance overview
        performance = kpis.get("performance_score", 0)
        if performance >= 80:
            insights.append(f"🎉 Excellent performance score: **{performance:.1f}/100**")
        elif performance >= 60:
            insights.append(f"👍 Good performance score: **{performance:.1f}/100**")
        else:
            insights.append(f"⚠️ Performance needs improvement: **{performance:.1f}/100**")
        
        # Data source insights
        if kpis.get("data_sources", 0) > 1:
            insights.append(f"📊 Data integrated from **{kpis['data_sources']} sources** for comprehensive analysis")
        
        # Sales target analysis
        if kpis.get("target_sales", 0) > 0:
            variance_pct = kpis.get("sales_variance_pct", 0)
            if variance_pct > 0:
                insights.append(f"📈 Sales exceeded target by **{variance_pct:.1f}%** (SAR {kpis['sales_variance']:,.0f})")
            else:
                insights.append(f"📉 Sales below target by **{abs(variance_pct):.1f}%** (SAR {abs(kpis['sales_variance']):,.0f})")
        
        # Store performance
        if not df.empty and df["Store"].notna().any():
            store_sales = df.groupby("Store")["Sales"].sum().sort_values(ascending=False)
            if not store_sales.empty:
                top_store = store_sales.index[0]
                top_sales = store_sales.iloc[0]
                insights.append(f"🏆 Top performing store: **{top_store}** (SAR {top_sales:,.0f})")
        
        # Liquidity status
        liquidity = kpis.get("total_liquidity", 0)
        if liquidity > 0:
            if liquidity > 50000:
                insights.append(f"💰 Strong liquidity position: **SAR {liquidity:,.0f}**")
            elif liquidity > 20000:
                insights.append(f"💛 Moderate liquidity: **SAR {liquidity:,.0f}**")
            else:
                insights.append(f"🔴 Low liquidity warning: **SAR {liquidity:,.0f}**")
        
        # Discount analysis
        if kpis.get("discount_rate", 0) > 0:
            if kpis["discount_rate"] > 10:
                insights.append(f"⚠️ High discount rate: **{kpis['discount_rate']:.1f}%** - monitor profitability")
            else:
                insights.append(f"🎯 Controlled discount rate: **{kpis['discount_rate']:.1f}%**")
        
        # Basket size analysis
        avg_basket = kpis.get("avg_basket", 0)
        if avg_basket > 0:
            if avg_basket > 50:
                insights.append(f"🛍️ High-value baskets: **SAR {avg_basket:.2f}** average")
            else:
                insights.append(f"🛒 Average basket size: **SAR {avg_basket:.2f}**")
        
        # Category performance
        if df["Category"].notna().any():
            cat_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
            if not cat_sales.empty:
                top_cat = cat_sales.index[0]
                top_cat_sales = cat_sales.iloc[0]
                insights.append(f"📊 Top category: **{top_cat}** (SAR {top_cat_sales:,.0f})")
        
        return insights

# =========================
# VISUALIZATION ENGINE
# =========================
class VisualizationEngine:
    """Handles all chart and visualization creation"""
    
    @staticmethod
    def create_target_vs_actual_chart(df: pd.DataFrame) -> go.Figure:
        """Create target vs actual sales comparison"""
        fig = go.Figure()
        
        if df["Date"].notna().any():
            daily_comparison = df.groupby(df["Date"].dt.date).agg({
                "Sales": "sum",
                "TargetSales": "sum"
            }).reset_index()
            daily_comparison.columns = ["Date", "Actual", "Target"]
            
            # Actual sales bars
            fig.add_trace(go.Bar(
                x=daily_comparison["Date"],
                y=daily_comparison["Actual"],
                name="Actual Sales",
                marker_color="#3b82f6",
                opacity=0.8,
                text=daily_comparison["Actual"].round(0),
                textposition="outside",
                texttemplate='SAR %{text:,.0f}'
            ))
            
            # Target line
            if daily_comparison["Target"].notna().any():
                fig.add_trace(go.Scatter(
                    x=daily_comparison["Date"],
                    y=daily_comparison["Target"],
                    name="Target Sales",
                    line=dict(color="#ef4444", width=3),
                    mode="lines+markers"
                ))
        
        fig.update_layout(
            title="Sales Performance: Target vs Actual",
            xaxis_title="Date",
            yaxis_title="Sales (SAR)",
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            barmode='overlay',
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_data_source_comparison(df: pd.DataFrame) -> go.Figure:
        """Create comparison chart showing data from different sources"""
        if "DataSource" not in df.columns:
            return go.Figure()
        
        source_comparison = df.groupby(["DataSource", df["Date"].dt.date]).agg({
            "Sales": "sum"
        }).reset_index()
        
        fig = go.Figure()
        
        colors = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#8b5cf6"]
        
        for i, source in enumerate(source_comparison["DataSource"].unique()):
            source_data = source_comparison[source_comparison["DataSource"] == source]
            
            fig.add_trace(go.Scatter(
                x=source_data["Date"],
                y=source_data["Sales"],
                name=f"Source: {source.title()}",
                mode="lines+markers",
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title="Sales Comparison by Data Source",
            xaxis_title="Date",
            yaxis_title="Sales (SAR)",
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_comprehensive_performance_chart(df: pd.DataFrame) -> go.Figure:
        """Create a comprehensive performance chart with multiple metrics"""
        if df["Date"].notna().any():
            daily_metrics = df.groupby(df["Date"].dt.date).agg({
                "Sales": "sum",
                "Qty": "sum",
                "Tickets": "sum",
                "GrossProfit": "sum",
                "Discount": "sum"
            }).reset_index()
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Daily Sales', 'Quantity Sold', 'Gross Profit', 'Discount Given'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Sales
            fig.add_trace(
                go.Bar(x=daily_metrics["Date"], y=daily_metrics["Sales"], 
                      name="Sales", marker_color="#3b82f6"),
                row=1, col=1
            )
            
            # Quantity
            fig.add_trace(
                go.Scatter(x=daily_metrics["Date"], y=daily_metrics["Qty"], 
                          name="Quantity", line=dict(color="#22c55e", width=3)),
                row=1, col=2
            )
            
            # Gross Profit
            fig.add_trace(
                go.Bar(x=daily_metrics["Date"], y=daily_metrics["GrossProfit"], 
                      name="Gross Profit", marker_color="#f59e0b"),
                row=2, col=1
            )
            
            # Discount
            fig.add_trace(
                go.Bar(x=daily_metrics["Date"], y=daily_metrics["Discount"], 
                      name="Discount", marker_color="#ef4444"),
                row=2, col=2
            )
            
            fig.update_layout(
                height=600,
                showlegend=False,
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff"
            )
            
            return fig
        
        return go.Figure()
    
    @staticmethod
    def create_liquidity_chart(df: pd.DataFrame) -> go.Figure:
        """Create dedicated liquidity trend chart"""
        fig = go.Figure()
        
        if df["Date"].notna().any():
            daily_liquidity = df.groupby(df["Date"].dt.date).agg({
                "BankBalance": "last",
                "CashBalance": "last",
                "TotalLiquidity": "last"
            }).reset_index()
            daily_liquidity.columns = ["Date", "Bank", "Cash", "Total"]
            
            # Bank balance
            fig.add_trace(go.Scatter(
                x=daily_liquidity["Date"],
                y=daily_liquidity["Bank"],
                name="Bank Balance",
                fill='tonexty',
                line=dict(color="#3b82f6", width=2)
            ))
            
            # Cash balance
            fig.add_trace(go.Scatter(
                x=daily_liquidity["Date"],
                y=daily_liquidity["Cash"],
                name="Cash Balance",
                fill='tonexty',
                line=dict(color="#22c55e", width=2)
            ))
            
            # Total liquidity line
            fig.add_trace(go.Scatter(
                x=daily_liquidity["Date"],
                y=daily_liquidity["Total"],
                name="Total Liquidity",
                line=dict(color="#ef4444", width=3, dash="dash")
            ))
        
        fig.update_layout(
            title="Liquidity Trend Analysis",
            xaxis_title="Date",
            yaxis_title="Amount (SAR)",
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            hovermode='x unified',
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_donut_chart(category_data: pd.DataFrame, title: str) -> go.Figure:
        """Create donut chart for category breakdown"""
        # Limit to top categories for clarity
        top_categories = category_data.head(8)
        if len(category_data) > 8:
            others_sum = category_data.iloc[8:]["Sales"].sum()
            others_df = pd.DataFrame([{"Category": "Others", "Sales": others_sum}])
            top_categories = pd.concat([top_categories, others_df], ignore_index=True)
        
        fig = go.Figure(go.Pie(
            labels=top_categories["Category"],
            values=top_categories["Sales"],
            hole=0.55,
            textinfo="label+percent",
            insidetextorientation="radial",
            marker=dict(
                colors=['#3b82f6', '#f59e0b', '#22c55e', '#ef4444', 
                       '#8b5cf6', '#ec4899', '#14b8a6', '#f97316']
            )
        ))
        
        fig.update_layout(
            title=title,
            showlegend=True,
            legend=dict(
                orientation="v",
                y=0.5,
                x=1.1
            ),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            height=400
        )
        
        return fig

# =========================
# TAB CONTENT FUNCTIONS
# =========================
def render_data_sources_tab(source_info: Dict[str, Any], filtered_data: pd.DataFrame):
    """Render the Data Sources tab content"""
    st.markdown("## 📊 Data Sources Management")
    
    # Data source status overview
    col1, col2, col3, col4 = st.columns(4)
    
    total_sources = len(source_info)
    successful_sources = sum(1 for info in source_info.values() if info['status'] == 'success')
    total_rows = sum(info.get('rows', 0) for info in source_info.values())
    
    with col1:
        st.metric("Total Sources", total_sources)
    with col2:
        st.metric("Active Sources", successful_sources)
    with col3:
        st.metric("Total Records", f"{total_rows:,}")
    with col4:
        coverage = (successful_sources / total_sources * 100) if total_sources > 0 else 0
        st.metric("Coverage", f"{coverage:.0f}%")
    
    # Individual source status
    st.markdown("### 🔗 Source Details")
    
    for source_name, info in source_info.items():
        status_color = "#dcfce7" if info['status'] == 'success' else "#fee2e2"
        status_icon = "✅" if info['status'] == 'success' else "❌"
        
        st.markdown(f"""
        <div class="data-source-card" style="background:{status_color};">
            <h4>{status_icon} {source_name.title()} Report</h4>
            <p><strong>Status:</strong> {info['status'].title()}</p>
            <p><strong>Records:</strong> {info['rows']:,}</p>
            <p><strong>Columns:</strong> {len(info['columns'])}</p>
            {f"<p><strong>Error:</strong> {info['error']}</p>" if info['status'] == 'error' else ""}
            <p><strong>URL:</strong> <code>{info['url'][:80]}...</code></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Data source comparison visualization
    if "DataSource" in filtered_data.columns and filtered_data["DataSource"].nunique() > 1:
        st.markdown("### 📈 Source Performance Comparison")
        
        source_chart = VisualizationEngine.create_data_source_comparison(filtered_data)
        st.plotly_chart(source_chart, use_container_width=True)
        
        # Source breakdown table
        source_breakdown = filtered_data.groupby("DataSource").agg({
            "Sales": "sum",
            "Qty": "sum",
            "Tickets": "sum",
            "Date": ["min", "max"]
        }).round(2)
        
        source_breakdown.columns = ["Total Sales", "Total Qty", "Total Tickets", "Start Date", "End Date"]
        source_breakdown = source_breakdown.reset_index()
        
        st.markdown("### 📊 Source Summary")
        st.dataframe(source_breakdown, use_container_width=True)

def render_overview_tab(filtered_data: pd.DataFrame, kpis: Dict[str, Any]):
    """Render the Overview tab content"""
    st.markdown("## 📊 Executive Overview")
    
    # Data sources indicator
    if kpis.get("data_sources", 0) > 1:
        st.markdown(f"""
        <div class="alert-info">
            📊 <strong>Multi-Source Analytics:</strong> This dashboard integrates data from 
            {kpis['data_sources']} different sources for comprehensive business insights.
        </div>
        """, unsafe_allow_html=True)
    
    # Primary KPIs Row 1
    st.markdown("### Financial Performance")
    
    cols = st.columns(5)
    with cols[0]:
        st.metric("💰 Total Sales", f"SAR {kpis['actual_sales']:,.0f}")
    with cols[1]:
        st.metric("🎯 Target Sales", f"SAR {kpis['target_sales']:,.0f}")
    with cols[2]:
        variance_delta = f"{kpis['sales_variance_pct']:+.1f}%" if kpis['target_sales'] > 0 else None
        st.metric("📊 Sales Variance", f"SAR {kpis['sales_variance']:,.0f}", delta=variance_delta)
    with cols[3]:
        st.metric("💎 Gross Profit", f"SAR {kpis['gross_profit']:,.0f}")
    with cols[4]:
        margin_delta = f"{kpis['gross_margin']:.1f}%" if kpis['gross_margin'] > 0 else None
        st.metric("📈 Gross Margin", margin_delta or "0.0%")
    
    # Secondary KPIs Row 2
    st.markdown("### Operations & Customer Metrics")
    kpi_row2 = [
        ("🛍️ Total Baskets", f"{kpis['total_baskets']:,.0f}", config.KPI_COLORS["tickets"]),
        ("🎯 Avg Basket", f"SAR {kpis['avg_basket']:.2f}", config.KPI_COLORS["avg_sales"]),
        ("💳 Discount Rate", f"{kpis['discount_rate']:.1f}%", config.KPI_COLORS["variance"]),
        ("👥 Total Customers", f"{kpis['customer_count']:,.0f}", config.KPI_COLORS["customer"]),
        ("🔄 Conversion Rate", f"{kpis['conversion_rate']:.1f}%", config.KPI_COLORS["conversion"])
    ]
    
    cols = st.columns(len(kpi_row2))
    for (label, value, color), col in zip(kpi_row2, cols):
        # Add performance indicator for conversion rate
        indicator_html = ""
        if "Conversion Rate" in label:
            if kpis['conversion_rate'] >= config.TARGET_THRESHOLDS["conversion_rate"]:
                indicator_html = '<span class="performance-indicator indicator-good">✓</span>'
            elif kpis['conversion_rate'] >= config.TARGET_THRESHOLDS["conversion_rate"] * 0.7:
                indicator_html = '<span class="performance-indicator indicator-warning">!</span>'
            else:
                indicator_html = '<span class="performance-indicator indicator-poor">⚠</span>'
        
        col.markdown(f"""
        <div class="kpi-tile" style="background:{color}">
            <div style="font-size:13px; color:#334155; font-weight:700;">
                {label}{indicator_html}
            </div>
            <div style="font-size:24px; line-height:1; margin-top:4px; color:#0f172a;">
                {value}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Liquidity Row
    st.markdown("### Liquidity Position")
    kpi_row3 = [
        ("🏦 Bank Balance", f"SAR {kpis['bank_balance']:,.0f}", config.KPI_COLORS["bank"]),
        ("💵 Cash Balance", f"SAR {kpis['cash_balance']:,.0f}", config.KPI_COLORS["cash"]),
        ("💧 Total Liquidity", f"SAR {kpis['total_liquidity']:,.0f}", config.KPI_COLORS["liquidity"]),
        ("📊 Performance Score", f"{kpis['performance_score']:.0f}/100", config.KPI_COLORS["target"]),
        ("📅 Days Analyzed", f"{kpis['days_count']}", config.KPI_COLORS["conversion"])
    ]
    
    cols = st.columns(len(kpi_row3))
    for (label, value, color), col in zip(kpi_row3, cols):
        col.markdown(f"""
        <div class="kpi-tile" style="background-color:{color};">
            <div style="font-size:13px; color:#334155; font-weight:700; margin-bottom:4px;">
                {label}
            </div>
            <div style="font-size:24px; line-height:1; color:#0f172a; font-weight:800;">
                {value}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Comprehensive performance chart
    if filtered_data["Date"].notna().any():
        st.markdown("### 📈 Comprehensive Performance Overview")
        perf_chart = VisualizationEngine.create_comprehensive_performance_chart(filtered_data)
        st.plotly_chart(perf_chart, use_container_width=True)

def render_financial_tab(filtered_data: pd.DataFrame, kpis: Dict[str, Any]):
    """Render the Financial Analytics tab content"""
    st.markdown("## 💰 Financial Analytics")
    
    if filtered_data["Date"].notna().any():
        col1, col2 = st.columns([3, 2])
        
        with col1:
            target_chart = VisualizationEngine.create_target_vs_actual_chart(filtered_data)
            st.plotly_chart(target_chart, use_container_width=True)
        
        with col2:
            liquidity_chart = VisualizationEngine.create_liquidity_chart(filtered_data)
            st.plotly_chart(liquidity_chart, use_container_width=True)
        
        # Financial metrics table
        st.markdown("### 📊 Comprehensive Financial Summary")
        
        financial_metrics = {
            "Revenue": f"SAR {kpis['actual_sales']:,.0f}",
            "Target": f"SAR {kpis['target_sales']:,.0f}",
            "Variance": f"SAR {kpis['sales_variance']:,.0f} ({kpis['sales_variance_pct']:+.1f}%)",
            "Gross Profit": f"SAR {kpis['gross_profit']:,.0f}",
            "Gross Margin": f"{kpis['gross_margin']:.1f}%",
            "Total Discounts": f"SAR {kpis['total_discount']:,.0f}",
            "Discount Rate": f"{kpis['discount_rate']:.1f}%",
            "Total Tax": f"SAR {kpis['total_tax']:,.0f}",
            "Bank Balance": f"SAR {kpis['bank_balance']:,.0f}",
            "Cash Balance": f"SAR {kpis['cash_balance']:,.0f}",
            "Total Liquidity": f"SAR {kpis['total_liquidity']:,.0f}",
            "Avg Daily Sales": f"SAR {kpis['avg_daily_sales']:,.0f}"
        }
        
        # Create two-column layout for metrics
        col1, col2 = st.columns(2)
        
        metrics_items = list(financial_metrics.items())
        mid_point = len(metrics_items) // 2
        
        with col1:
            for metric, value in metrics_items[:mid_point]:
                st.metric(label=metric, value=value)
        
        with col2:
            for metric, value in metrics_items[mid_point:]:
                st.metric(label=metric, value=value)

def render_sales_tab(filtered_data: pd.DataFrame, kpis: Dict[str, Any]):
    """Render the Sales Performance tab content"""
    st.markdown("## 📈 Sales Performance Analysis")
    
    if filtered_data["Date"].notna().any():
        # Enhanced time period selector
        time_periods = ["Last 7 Days", "Last 30 Days", "Current Month", "All Time"]
        selected_period = st.selectbox("Time Period", options=time_periods, index=2)
        
        # Filter based on selected period
        today = datetime.now().date()
        if selected_period == "Last 7 Days":
            start_date = today - timedelta(days=7)
            period_data = filtered_data[filtered_data["Date"].dt.date >= start_date]
        elif selected_period == "Last 30 Days":
            start_date = today - timedelta(days=30)
            period_data = filtered_data[filtered_data["Date"].dt.date >= start_date]
        elif selected_period == "Current Month":
            start_date = today.replace(day=1)
            period_data = filtered_data[filtered_data["Date"].dt.date >= start_date]
        else:
            period_data = filtered_data.copy()
        
        if not period_data.empty:
            # Daily trend analysis
            daily_analysis = period_data.groupby(period_data["Date"].dt.date).agg({
                "Sales": "sum",
                "Qty": "sum",
                "Tickets": "sum",
                "GrossProfit": "sum"
            }).reset_index()
            
            daily_analysis["AvgBasket"] = daily_analysis["Sales"] / daily_analysis["Tickets"]
            daily_analysis["ItemsPerBasket"] = daily_analysis["Qty"] / daily_analysis["Tickets"]
            
            # Visualization
            col1, col2 = st.columns([3, 2])
            
            with col1:
                fig = go.Figure()
                
                # Sales bars
                fig.add_trace(go.Bar(
                    x=daily_analysis["Date"],
                    y=daily_analysis["Sales"],
                    name="Daily Sales",
                    marker_color="#3b82f6",
                    opacity=0.8
                ))
                
                # Average basket line
                fig.add_trace(go.Scatter(
                    x=daily_analysis["Date"],
                    y=daily_analysis["AvgBasket"] * 10,  # Scale for visibility
                    name="Avg Basket (×10)",
                    line=dict(color="#f59e0b", width=3),
                    yaxis="y2"
                ))
                
                fig.update_layout(
                    title=f"Sales Trend - {selected_period}",
                    xaxis_title="Date",
                    yaxis_title="Sales (SAR)",
                    yaxis2=dict(
                        title="Avg Basket (×10)",
                        overlaying="y",
                        side="right"
                    ),
                    plot_bgcolor="#ffffff",
                    paper_bgcolor="#ffffff",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Category performance for the period
                if period_data["Category"].notna().any():
                    cat_sales = period_data.groupby("Category")["Sales"].sum().sort_values(ascending=False).reset_index()
                    donut_chart = VisualizationEngine.create_donut_chart(
                        cat_sales,
                        f"Category Mix - {selected_period}"
                    )
                    st.plotly_chart(donut_chart, use_container_width=True)
        
        # Store comparison
        if filtered_data["Store"].notna().any() and len(filtered_data["Store"].unique()) > 1:
            st.markdown("### 🏪 Multi-Store Performance Analysis")
            
            store_comparison = filtered_data.groupby("Store").agg({
                "Sales": "sum",
                "Tickets": "sum",
                "Qty": "sum",
                "GrossProfit": "sum"
            }).reset_index()
            
            store_comparison["AvgBasket"] = store_comparison["Sales"] / store_comparison["Tickets"]
            store_comparison["GrossMargin"] = (store_comparison["GrossProfit"] / store_comparison["Sales"] * 100).round(1)
            
            # Format for display
            display_stores = store_comparison.copy()
            display_stores["Sales"] = display_stores["Sales"].apply(lambda x: f"SAR {x:,.0f}")
            display_stores["GrossProfit"] = display_stores["GrossProfit"].apply(lambda x: f"SAR {x:,.0f}")
            display_stores["AvgBasket"] = display_stores["AvgBasket"].apply(lambda x: f"SAR {x:.2f}")
            
            st.dataframe(display_stores, use_container_width=True)

def render_items_tab(filtered_data: pd.DataFrame):
    """Render the Top Items Analysis tab content"""
    st.markdown("## ⭐ Product Performance Analytics")
    
    if filtered_data["Item"].notna().any():
        # Top items analysis
        item_analysis = (filtered_data.groupby("Item")
                        .agg({
                            "Sales": "sum", 
                            "Qty": "sum", 
                            "GrossProfit": "sum",
                            "Tickets": "sum",
                            "COGS": "sum"
                        })
                        .sort_values("Sales", ascending=False)
                        .reset_index())
        
        # Calculate additional metrics
        item_analysis["AvgPrice"] = item_analysis["Sales"] / item_analysis["Qty"]
        item_analysis["MarginPct"] = (item_analysis["GrossProfit"] / item_analysis["Sales"] * 100).round(1)
        item_analysis["Frequency"] = item_analysis["Qty"] / item_analysis["Tickets"]
        item_analysis["Velocity"] = item_analysis["Qty"]  # Items sold
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Analysis type selector
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Top Sellers", "Most Profitable", "Best Margin", "Highest Frequency"]
            )
            
            if analysis_type == "Top Sellers":
                display_data = item_analysis.sort_values("Sales", ascending=False).head(15)
                metric_col = "Sales"
            elif analysis_type == "Most Profitable":
                display_data = item_analysis.sort_values("GrossProfit", ascending=False).head(15)
                metric_col = "GrossProfit"
            elif analysis_type == "Best Margin":
                display_data = item_analysis.sort_values("MarginPct", ascending=False).head(15)
                metric_col = "MarginPct"
            else:  # Highest Frequency
                display_data = item_analysis.sort_values("Frequency", ascending=False).head(15)
                metric_col = "Frequency"
            
            # Format for display
            display_items = display_data.copy()
            display_items["Sales"] = display_items["Sales"].apply(lambda x: f"SAR {x:,.0f}")
            display_items["GrossProfit"] = display_items["GrossProfit"].apply(lambda x: f"SAR {x:,.0f}")
            display_items["AvgPrice"] = display_items["AvgPrice"].apply(lambda x: f"SAR {x:.2f}")
            display_items["Frequency"] = display_items["Frequency"].apply(lambda x: f"{x:.2f}")
            display_items["Qty"] = display_items["Qty"].astype(int)
            
            # Select columns for display
            display_columns = ["Item", "Sales", "Qty", "GrossProfit", "MarginPct", "AvgPrice", "Frequency"]
            column_names = ["Item", "Sales", "Qty Sold", "Gross Profit", "Margin %", "Avg Price", "Frequency"]
            
            display_items = display_items[display_columns]
            display_items.columns = column_names
            
            st.dataframe(display_items, use_container_width=True, height=500)
        
        with col2:
            # Performance insights
            st.markdown("### 📊 Product Insights")
            
            total_items = len(item_analysis)
            top_20_pct = int(total_items * 0.2)
            top_20_sales = item_analysis.head(top_20_pct)["Sales"].sum()
            total_sales = item_analysis["Sales"].sum()
            pareto_ratio = (top_20_sales / total_sales * 100) if total_sales > 0 else 0
            
            st.metric(
                "Pareto Analysis",
                f"{pareto_ratio:.1f}%",
                f"Top 20% products contribution"
            )
            
            avg_margin = item_analysis["MarginPct"].mean()
            st.metric(
                "Average Margin",
                f"{avg_margin:.1f}%",
                "Across all products"
            )
            
            high_margin_count = len(item_analysis[item_analysis["MarginPct"] > 30])
            st.metric(
                "High Margin Items",
                f"{high_margin_count}",
                f"Items with >30% margin"
            )
            
            # Top 5 items pie chart
            top_5_items = item_analysis.head(5)
            
            fig = go.Figure(go.Pie(
                labels=top_5_items["Item"].str[:20] + "...",
                values=top_5_items["Sales"],
                hole=0.4,
                textinfo="label+percent",
                marker=dict(colors=['#3b82f6', '#f59e0b', '#22c55e', '#ef4444', '#8b5cf6'])
            ))
            
            fig.update_layout(
                title="Top 5 Items by Sales",
                showlegend=True,
                height=350,
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("No item-level data available for analysis")

def render_insights_tab(filtered_data: pd.DataFrame, kpis: Dict[str, Any]):
    """Render the Business Insights tab content"""
    st.markdown("## 💡 AI-Powered Business Intelligence")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Key Performance Insights")
        
        insights = AnalyticsEngine.generate_insights(filtered_data, kpis)
        if insights:
            for insight in insights:
                st.markdown(f"- {insight}")
        else:
            st.write("No insights available for the current selection")
        
        # Advanced analytics
        st.markdown("### 📈 Advanced Analytics")
        
        if filtered_data["Date"].notna().any():
            # Trend analysis
            if len(filtered_data["Date"].unique()) > 7:
                daily_sales = filtered_data.groupby(filtered_data["Date"].dt.date)["Sales"].sum()
                
                # Calculate trend
                x = np.arange(len(daily_sales))
                y = daily_sales.values
                
                if len(x) > 1:
                    slope = np.polyfit(x, y, 1)[0]
                    trend_direction = "📈 Increasing" if slope > 0 else "📉 Decreasing" if slope < 0 else "➡️ Stable"
                    daily_change = slope
                    
                    st.markdown(f"""
                    <div class="alert-info">
                        <strong>Sales Trend:</strong> {trend_direction}<br>
                        <strong>Daily Change:</strong> SAR {daily_change:.2f} per day<br>
                        <strong>Period:</strong> {len(daily_sales)} days analyzed
                    </div>
                    """, unsafe_allow_html=True)
        
        # Seasonality insights
        if filtered_data["Date"].notna().any():
            day_of_week_sales = filtered_data.groupby(filtered_data["Date"].dt.day_name())["Sales"].mean()
            best_day = day_of_week_sales.idxmax()
            best_day_sales = day_of_week_sales.max()
            
            st.markdown(f"""
            <div class="alert-success">
                <strong>Peak Performance:</strong> {best_day}<br>
                <strong>Average Sales:</strong> SAR {best_day_sales:,.2f}<br>
                <strong>Recommendation:</strong> Focus marketing efforts on {best_day}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Performance Scorecard")
        
        performance_metrics = [
            ("Sales Target Achievement", f"{kpis['sales_variance_pct']:+.1f}%", 
             "✅" if kpis['sales_variance_pct'] >= 0 else "⚠️"),
            ("Gross Margin", f"{kpis['gross_margin']:.1f}%", 
             "✅" if kpis['gross_margin'] >= 25 else "⚠️"),
            ("Conversion Rate", f"{kpis['conversion_rate']:.1f}%", 
             "✅" if kpis['conversion_rate'] >= config.TARGET_THRESHOLDS["conversion_rate"] else "⚠️"),
            ("Discount Control", f"{kpis['discount_rate']:.1f}%", 
             "✅" if kpis['discount_rate'] <= 10 else "⚠️"),
            ("Liquidity Position", f"SAR {kpis['total_liquidity']:,.0f}", 
             "✅" if kpis['total_liquidity'] >= 50000 else "⚠️"),
            ("Overall Score", f"{kpis['performance_score']:.0f}/100", 
             "✅" if kpis['performance_score'] >= 70 else "⚠️")
        ]
        
        for metric, value, status in performance_metrics:
            col_a, col_b, col_c = st.columns([0.1, 0.7, 0.2])
            with col_a:
                st.write(status)
            with col_b:
                st.write(metric)
            with col_c:
                st.write(f"**{value}**")
        
        # Recommendations
        st.markdown("### 💡 Smart Recommendations")
        
        recommendations = []
        
        if kpis['gross_margin'] < 25:
            recommendations.append("🎯 Review pricing strategy to improve margins")
        
        if kpis['discount_rate'] > 10:
            recommendations.append("💰 Optimize discount strategy to protect profitability")
        
        if kpis['conversion_rate'] < config.TARGET_THRESHOLDS["conversion_rate"]:
            recommendations.append("📈 Focus on conversion rate improvement")
        
        if kpis['total_liquidity'] < 50000:
            recommendations.append("💧 Monitor cash flow and liquidity position")
        
        if not recommendations:
            recommendations.append("🎉 All key metrics are performing well!")
        
        for rec in recommendations:
            st.markdown(f"- {rec}")

def render_data_tab(filtered_data: pd.DataFrame, column_mapping: Dict[str, str], source_info: Dict[str, Any]):
    """Render the Data Management tab content"""
    st.markdown("## 📊 Data Management & Export")
    
    # Data overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**📅 Data Range**")
        if filtered_data["Date"].notna().any():
            st.write(f"From: {filtered_data['Date'].min().strftime('%Y-%m-%d')}")
            st.write(f"To: {filtered_data['Date'].max().strftime('%Y-%m-%d')}")
            st.write(f"Days: {filtered_data['Date'].dt.date.nunique()}")
        else:
            st.write("No date information available")
    
    with col2:
        st.markdown("**📊 Data Quality**")
        total_rows = len(filtered_data)
        st.write(f"Total Records: {total_rows:,}")
        st.write(f"Stores: {filtered_data['Store'].nunique()}")
        st.write(f"Categories: {filtered_data['Category'].nunique()}")
        st.write(f"Unique Items: {filtered_data['Item'].nunique()}")
    
    with col3:
        st.markdown("**🎯 Target Thresholds**")
        st.write(f"Conversion Rate: {config.TARGET_THRESHOLDS['conversion_rate']:.1f}%")
        st.write(f"Gross Margin: {config.TARGET_THRESHOLDS['gross_margin']:.1f}%")
        st.write(f"Liquidity Ratio: {config.TARGET_THRESHOLDS['liquidity_ratio']:.1f}")
        st.write(f"Growth Rate: {config.TARGET_THRESHOLDS['growth_rate']:.1f}%")
    
    with col4:
        st.markdown("**🔗 Export Options**")
        
        # Excel export with multiple sheets
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            # Main data
            filtered_data.to_excel(writer, sheet_name='Complete Data', index=False)
            
            # Summary by store
            if filtered_data["Store"].notna().any():
                store_summary = filtered_data.groupby("Store").agg({
                    "Sales": "sum",
                    "Qty": "sum",
                    "Tickets": "sum",
                    "GrossProfit": "sum"
                }).reset_index()
                store_summary.to_excel(writer, sheet_name='Store Summary', index=False)
            
            # Daily summary
            if filtered_data["Date"].notna().any():
                daily_summary = filtered_data.groupby(filtered_data["Date"].dt.date).agg({
                    "Sales": "sum",
                    "Qty": "sum",
                    "Tickets": "sum",
                    "GrossProfit": "sum"
                }).reset_index()
                daily_summary.to_excel(writer, sheet_name='Daily Summary', index=False)
        
        st.download_button(
            label="📥 Download Excel Report",
            data=buffer.getvalue(),
            file_name=f"Alkhair_Dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        
        # CSV export
        csv_buffer = io.StringIO()
        filtered_data.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📄 Download CSV",
            data=csv_buffer.getvalue(),
            file_name=f"Alkhair_Data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Column mapping information
    st.markdown("### 🔗 Column Mapping Details")
    mapping_df = pd.DataFrame({
        "Target Column": list(column_mapping.keys()),
        "Source Column": list(column_mapping.values())
    })
    st.dataframe(mapping_df, use_container_width=True, hide_index=True)
    
    # Data sources summary
    if source_info:
        st.markdown("### 📊 Data Sources Summary")
        sources_df = pd.DataFrame([
            {
                "Source": name.title(),
                "Status": info['status'].title(),
                "Records": info['rows'],
                "Columns": len(info['columns']),
                "URL": info['url'][:50] + "..." if len(info['url']) > 50 else info['url']
            }
            for name, info in source_info.items()
        ])
        st.dataframe(sources_df, use_container_width=True, hide_index=True)
    
    # Raw data preview
    st.markdown("### 📋 Data Preview")
    
    # Show sample of each data source if available
    if "DataSource" in filtered_data.columns and filtered_data["DataSource"].nunique() > 1:
        selected_source = st.selectbox(
            "Select Data Source to Preview",
            options=["All Sources"] + list(filtered_data["DataSource"].unique())
        )
        
        if selected_source == "All Sources":
            preview_data = filtered_data.head(100)
        else:
            preview_data = filtered_data[filtered_data["DataSource"] == selected_source].head(100)
    else:
        preview_data = filtered_data.head(100)
    
    st.dataframe(preview_data, use_container_width=True)

# =========================
# MAIN APPLICATION
# =========================
def main():
    """Main application entry point"""
    
    # Apply custom CSS
    apply_custom_css()
    
    # Auto-refresh
    components.html(
        f"<meta http-equiv='refresh' content='{config.REFRESH_MINUTES * 60}'>",
        height=0
    )
    
    # Enhanced Header
    st.markdown(f"""
    <div class="brandbar">
        <div style="display:flex;align-items:center;justify-content:space-between;padding:0 20px;">
            <div>
                <div style="font-weight:800;font-size:1.3rem;">
                    🏪 Alkhair Family Market — Executive Dashboard
                </div>
                <div style="font-size:0.9rem;opacity:0.8;margin-top:4px;">
                    Multi-Source Business Intelligence & Performance Analytics
                    <span class="data-source-indicator">
                        {len(config.DATA_SOURCES)} Data Sources
                    </span>
                </div>
            </div>
            <div style="text-align:right;font-size:0.85rem;">
                <div>Auto refresh: every {config.REFRESH_MINUTES} min</div>
                <div>Last update: {datetime.now().strftime('%H:%M:%S')}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Refresh button
    col1, col2 = st.columns([10, 1])
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Load data from multiple sources
    @st.cache_data(ttl=0, show_spinner=False)
    def load_multiple_data_sources(sources: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        return DataProcessor.load_multiple_sources(sources)
    
    try:
        with st.spinner("Loading data from multiple sources..."):
            combined_data, source_info = load_multiple_data_sources(config.DATA_SOURCES)
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        st.stop()
    
    # Show data loading status
    successful_sources = sum(1 for info in source_info.values() if info['status'] == 'success')
    if successful_sources == 0:
        st.error("❌ No data sources loaded successfully. Please check your connections.")
        st.stop()
    elif successful_sources < len(source_info):
        st.warning(f"⚠️ {successful_sources}/{len(source_info)} data sources loaded successfully.")
    else:
        st.success(f"✅ All {successful_sources} data sources loaded successfully!")
    
    # Standardize data
    if combined_data.empty:
        st.error("❌ No data available after loading sources.")
        st.stop()
    
    data, column_mapping = DataProcessor.standardize_dataframe(combined_data)
    
    # Filters in main area (before tabs)
    st.markdown("### 🔍 Data Filters")
    col1, col2, col3, col4 = st.columns([2, 2, 3, 3])
    
    # Date range filter
    if data["Date"].notna().any():
        date_min = data["Date"].min().date()
        date_max = data["Date"].max().date()
        
        with col1:
            start_date = st.date_input("Start Date", value=date_min, key="start_date")
        with col2:
            end_date = st.date_input("End Date", value=date_max, key="end_date")
        
        # Apply date filter
        mask = (data["Date"].dt.date >= start_date) & (data["Date"].dt.date <= end_date)
        filtered_data = data[mask].copy()
    else:
        filtered_data = data.copy()
        st.warning("⚠️ No valid date information found in the data")
    
    # Store filter
    if filtered_data["Store"].notna().any():
        stores = sorted([s for s in filtered_data["Store"].unique() if s])
        
        # Default selection logic
        default_stores = []
        for store_name in ["Magnus", "Alkhair", "Zelayi"]:
            if store_name in stores:
                default_stores.append(store_name)
        if not default_stores:
            default_stores = stores
        
        with col3:
            selected_stores = st.multiselect(
                "Select Store(s)",
                options=stores,
                default=default_stores,
                key="store_filter"
            )
        
        if selected_stores:
            filtered_data = filtered_data[filtered_data["Store"].isin(selected_stores)]
    
    # Data source filter
    if "DataSource" in filtered_data.columns and filtered_data["DataSource"].nunique() > 1:
        with col4:
            data_sources = list(filtered_data["DataSource"].unique())
            selected_sources = st.multiselect(
                "Select Data Source(s)",
                options=data_sources,
                default=data_sources,
                key="source_filter"
            )
        
        if selected_sources:
            filtered_data = filtered_data[filtered_data["DataSource"].isin(selected_sources)]
    
    # Calculate KPIs
    kpis = AnalyticsEngine.calculate_kpis(filtered_data)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Overview", 
        "💰 Financial", 
        "📈 Sales", 
        "⭐ Products", 
        "💡 Insights", 
        "📊 Data Sources",
        "📋 Export"
    ])
    
    # Render tab content
    with tab1:
        render_overview_tab(filtered_data, kpis)
    
    with tab2:
        render_financial_tab(filtered_data, kpis)
    
    with tab3:
        render_sales_tab(filtered_data, kpis)
    
    with tab4:
        render_items_tab(filtered_data)
    
    with tab5:
        render_insights_tab(filtered_data, kpis)
    
    with tab6:
        render_data_sources_tab(source_info, filtered_data)
    
    with tab7:
        render_data_tab(filtered_data, column_mapping, source_info)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.info("Please refresh the page or contact support if the issue persists.")
        st.stop()

