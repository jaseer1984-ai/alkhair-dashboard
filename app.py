# app.py — Enhanced Alkhair Family Market Dashboard with Tabbed Interface
# Organized into separate tabs for better user experience

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

# =========================
# CONFIGURATION
# =========================
@dataclass
class Config:
    """Application configuration settings"""
    PAGE_TITLE = "Alkhair Family Market — Executive Dashboard"
    LAYOUT = "wide"
    REFRESH_MINUTES = 1
    GSHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSW_ui1m_393ipZv8NAliu1rly6zeifFxMfOWpQF17hjVIDa9Ll8PiGCaz8gTRkMQ/pub?output=csv"
    
    # Enhanced column aliases for comprehensive data mapping
    COLUMN_MAPPINGS = {
        "Date": ["date", "bill_date", "txn_date", "invoice_date", "posting_date"],
        "Store": ["store", "branch", "location", "shop"],
        "Sales": ["sales", "amount", "net_sales", "revenue", "netamount", "net_amount", "actual_sales"],
        "TargetSales": ["target_sales", "target", "sales_target", "budget_sales", "planned_sales"],
        "COGS": ["cogs", "cost", "purchase_cost", "cost_of_goods"],
        "Qty": ["qty", "quantity", "qty_sold", "quantity_sold", "units", "pcs", "pieces", "units_sold", "qnty", "qnt", "qty."],
        "Tickets": ["tickets", "bills", "invoice_count", "transactions", "bills_count", "baskets", "no_of_baskets"],
        "Category": ["category", "cat"],
        "Item": ["item", "product", "sku", "item_name"],
        "InventoryQty": ["inventoryqty", "inventory_qty", "stock_qty", "stockqty", "stock_quantity"],
        "InventoryValue": ["inventoryvalue", "inventory_value", "stock_value", "stockvalue"],
        "BankBalance": ["bank_balance", "bank", "bank_amount", "bankbalance", "account_balance"],
        "CashBalance": ["cash_balance", "cash", "cash_amount", "cashbalance", "petty_cash"],
        "Expenses": ["expenses", "operating_expenses", "opex", "costs", "expenditure"],
        "CustomerCount": ["customer_count", "customers", "footfall", "visitors"],
        "Supplier": ["supplier", "vendor", "supplier_name"],
        "PaymentMethod": ["payment_method", "payment_type", "payment"]
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
        "variance": "#fef3c7"
    }
    
    # Target thresholds for performance indicators
    TARGET_THRESHOLDS = {
        "conversion_rate": 2.5,  # Target conversion rate %
        "gross_margin": 30.0,    # Target gross margin %
        "liquidity_ratio": 2.0   # Target liquidity ratio
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
                                  "InventoryValue", "BankBalance", "CashBalance", "Expenses", "CustomerCount"]:
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
        
        # Inventory metrics
        if df["InventoryValue"].notna().any():
            kpis["inventory_value"] = float(df["InventoryValue"].sum())
            kpis["inventory_turnover"] = kpis["total_sales"] / kpis["inventory_value"] if kpis.get("inventory_value", 0) > 0 else 0
        
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
def render_overview_tab(filtered_data: pd.DataFrame, kpis: Dict[str, Any]):
    """Render the Overview tab content"""
    st.markdown("## 📊 Executive Overview")
    
    # Primary KPIs Row 1
    st.markdown("### Financial Performance")
    kpi_row1 = [
        ("💰 Total Sales", f"SAR {kpis['actual_sales']:,.0f}", config.KPI_COLORS["sales"]),
        ("🎯 Target Sales", f"SAR {kpis['target_sales']:,.0f}", config.KPI_COLORS["target"]),
        ("📊 Sales Variance", f"{kpis['sales_variance_pct']:+.1f}%", 
         config.KPI_COLORS["variance"] if kpis['sales_variance_pct'] >= 0 else "#fee2e2"),
        ("💎 Gross Profit", f"SAR {kpis['gross_profit']:,.0f}", config.KPI_COLORS["profit"]),
        ("📈 Gross Margin", f"{kpis['gross_margin']:.1f}%", config.KPI_COLORS["profit"])
    ]
    
    cols = st.columns(len(kpi_row1))
    for (label, value, color), col in zip(kpi_row1, cols):
        col.markdown(f"""
        <div class="kpi-tile" style="background:{color}">
            <div style="font-size:13px; color:#334155; font-weight:700;">{label}</div>
            <div style="font-size:24px; line-height:1; margin-top:4px; color:#0f172a;">{value}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Secondary KPIs Row 2
    st.markdown("### Liquidity & Operations")
    kpi_row2 = [
        ("🏦 Bank Balance", f"SAR {kpis['bank_balance']:,.0f}", config.KPI_COLORS["bank"]),
        ("💵 Cash Balance", f"SAR {kpis['cash_balance']:,.0f}", config.KPI_COLORS["cash"]),
        ("💧 Total Liquidity", f"SAR {kpis['total_liquidity']:,.0f}", config.KPI_COLORS["liquidity"]),
        ("🛍️ Total Baskets", f"{kpis['total_baskets']:,.0f}", config.KPI_COLORS["tickets"]),
        ("🎯 Conversion Rate", f"{kpis['conversion_rate']:.1f}%", config.KPI_COLORS["conversion"])
    ]
    
    cols = st.columns(len(kpi_row2))
    for (label, value, color), col in zip(kpi_row2, cols):
        # Add performance indicator
        indicator_class = ""
        indicator_symbol = ""
        if "Conversion Rate" in label:
            if kpis['conversion_rate'] >= config.TARGET_THRESHOLDS["conversion_rate"]:
                indicator_class = "indicator-good"
                indicator_symbol = "✓"
            elif kpis['conversion_rate'] >= config.TARGET_THRESHOLDS["conversion_rate"] * 0.7:
                indicator_class = "indicator-warning"
                indicator_symbol = "!"
            else:
                indicator_class = "indicator-poor"
                indicator_symbol = "⚠"
        
        col.markdown(f"""
        <div class="kpi-tile" style="background:{color}">
            <div style="font-size:13px; color:#334155; font-weight:700;">{label}
                <span class="performance-indicator {indicator_class}">
                    {indicator_symbol}
                </span>
            </div>
            <div style="font-size:24px; line-height:1; margin-top:4px; color:#0f172a;">{value}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance Summary
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.metric(
            label="Performance Score",
            value=f"{kpis['performance_score']:.0f}/100",
            delta=f"Target: 70+"
        )
    
    with col2:
        st.metric(
            label="Avg Daily Sales",
            value=f"SAR {kpis['avg_daily_sales']:,.0f}",
            delta=f"Over {kpis['days_count']} days"
        )
    
    with col3:
        st.metric(
            label="Avg Basket Size",
            value=f"SAR {kpis['avg_basket']:.2f}",
            delta=f"From {kpis['total_baskets']:,.0f} baskets"
        )

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
        st.markdown("### 📊 Financial Summary")
        
        financial_metrics = {
            "Revenue": f"SAR {kpis['actual_sales']:,.0f}",
            "Target": f"SAR {kpis['target_sales']:,.0f}",
            "Variance": f"SAR {kpis['sales_variance']:,.0f} ({kpis['sales_variance_pct']:+.1f}%)",
            "Gross Profit": f"SAR {kpis['gross_profit']:,.0f}",
            "Gross Margin": f"{kpis['gross_margin']:.1f}%",
            "Bank Balance": f"SAR {kpis['bank_balance']:,.0f}",
            "Cash Balance": f"SAR {kpis['cash_balance']:,.0f}",
            "Total Liquidity": f"SAR {kpis['total_liquidity']:,.0f}"
        }
        
        financial_df = pd.DataFrame(list(financial_metrics.items()), columns=['Metric', 'Value'])
        st.dataframe(financial_df, use_container_width=True, hide_index=True)

def render_sales_tab(filtered_data: pd.DataFrame, kpis: Dict[str, Any]):
    """Render the Sales Performance tab content"""
    st.markdown("## 📈 Sales Performance")
    
    if filtered_data["Date"].notna().any():
        # Month selector
        months = sorted(filtered_data["Date"].dt.to_period("M").unique())
        month_strings = [m.strftime("%Y-%m") for m in months]
        
        selected_month = st.selectbox(
            "Select Month for Analysis",
            options=month_strings,
            index=len(month_strings)-1 if month_strings else 0
        )
        
        # Filter for selected month
        if selected_month:
            year, month = map(int, selected_month.split("-"))
            month_data = filtered_data[
                (filtered_data["Date"].dt.year == year) & 
                (filtered_data["Date"].dt.month == month)
            ]
        else:
            month_data = filtered_data.copy()
        
        # Prepare daily data
        if not month_data.empty:
            month_data["Day"] = month_data["Date"].dt.date
            daily_agg = month_data.groupby("Day").agg({
                "Sales": "sum",
                "Qty": "sum",
                "Tickets": "sum"
            }).reset_index()
            daily_agg["Conv"] = np.where(
                daily_agg["Tickets"] > 0,
                daily_agg["Qty"] / daily_agg["Tickets"] * 100,
                0
            )
        else:
            daily_agg = pd.DataFrame(columns=["Day", "Sales", "Qty", "Tickets", "Conv"])
        
        # Visualizations
        col1, col2 = st.columns([3, 2])
        
        with col1:
            if not daily_agg.empty:
                # Create combination chart
                fig = go.Figure()
                
                # Sales bars
                fig.add_trace(go.Bar(
                    x=daily_agg["Day"],
                    y=daily_agg["Sales"],
                    name="Sales",
                    marker_color="#3b82f6",
                    opacity=0.95,
                    text=daily_agg["Sales"].round(0),
                    textposition="outside",
                    texttemplate='%{text:,.0f}'
                ))
                
                # Conversion line
                fig.add_trace(go.Scatter(
                    x=daily_agg["Day"],
                    y=daily_agg["Conv"],
                    name="Sales Conversion",
                    mode="lines+markers",
                    line=dict(width=3, color="#f59e0b"),
                    marker=dict(size=8),
                    yaxis="y2"
                ))
                
                # Layout
                fig.update_layout(
                    title="Daily Sales and Conversion Trend",
                    xaxis=dict(
                        title="",
                        tickangle=-45,
                        showgrid=True,
                        gridcolor="#f0f0f0"
                    ),
                    yaxis=dict(
                        title="Sales (SAR)",
                        showgrid=True,
                        gridcolor="#f0f0f0"
                    ),
                    yaxis2=dict(
                        title="Conversion %",
                        overlaying="y",
                        side="right",
                        showgrid=False,
                        rangemode="tozero"
                    ),
                    legend=dict(
                        orientation="h",
                        y=1.1,
                        x=0
                    ),
                    plot_bgcolor="#ffffff",
                    paper_bgcolor="#ffffff",
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for the selected month")
        
        with col2:
            if month_data["Category"].notna().any():
                cat_sales = month_data.groupby("Category")["Sales"].sum().sort_values(ascending=False).reset_index()
                donut_chart = VisualizationEngine.create_donut_chart(
                    cat_sales,
                    f"Category Breakdown - {selected_month}"
                )
                st.plotly_chart(donut_chart, use_container_width=True)
            else:
                st.info("No category data available")
    
    # Store comparison
    if filtered_data["Store"].notna().any() and len(filtered_data["Store"].unique()) > 1:
        st.markdown("### 🏪 Store Performance Comparison")
        store_sales = filtered_data.groupby("Store")["Sales"].sum().sort_values(ascending=True).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=store_sales["Store"],
            x=store_sales["Sales"],
            orientation='h',
            marker_color='#3b82f6',
            text=store_sales["Sales"].round(0),
            textposition='outside',
            texttemplate='SAR %{text:,.0f}'
        ))
        
        fig.update_layout(
            title="Store Performance Comparison",
            xaxis=dict(title="Sales (SAR)"),
            yaxis=dict(title=""),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_items_tab(filtered_data: pd.DataFrame):
    """Render the Top Items Analysis tab content"""
    st.markdown("## ⭐ Top Items Analysis")
    
    if filtered_data["Item"].notna().any():
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("### 🏆 Top 10 Performing Items")
            
            top_10_items = (filtered_data.groupby("Item")
                           .agg({
                               "Sales": "sum", 
                               "Qty": "sum", 
                               "GrossProfit": "sum",
                               "Tickets": "sum"
                           })
                           .sort_values("Sales", ascending=False)
                           .head(10)
                           .reset_index())
            
            # Calculate additional metrics
            top_10_items["Avg_Price"] = top_10_items["Sales"] / top_10_items["Qty"]
            top_10_items["Margin_Pct"] = (top_10_items["GrossProfit"] / top_10_items["Sales"] * 100).round(1)
            
            # Format for display
            display_items = top_10_items.copy()
            display_items["Sales"] = display_items["Sales"].apply(lambda x: f"SAR {x:,.0f}")
            display_items["GrossProfit"] = display_items["GrossProfit"].apply(lambda x: f"SAR {x:,.0f}")
            display_items["Avg_Price"] = display_items["Avg_Price"].apply(lambda x: f"SAR {x:.2f}")
            display_items["Qty"] = display_items["Qty"].astype(int)
            
            display_items = display_items[["Item", "Sales", "Qty", "GrossProfit", "Margin_Pct", "Avg_Price"]]
            display_items.columns = ["Item", "Sales", "Qty Sold", "Gross Profit", "Margin %", "Avg Price"]
            
            st.dataframe(display_items, use_container_width=True, height=400)
        
        with col2:
            st.markdown("### 📊 Top 5 Items Distribution")
            
            # Top items pie chart
            top_items_chart_data = top_10_items.head(5)  # Top 5 for pie chart clarity
            
            fig = go.Figure(go.Pie(
                labels=top_items_chart_data["Item"],
                values=top_items_chart_data["Sales"],
                hole=0.4,
                textinfo="label+percent",
                marker=dict(colors=['#3b82f6', '#f59e0b', '#22c55e', '#ef4444', '#8b5cf6'])
            ))
            
            fig.update_layout(
                title="Sales Distribution",
                showlegend=True,
                height=400,
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Item performance metrics
        st.markdown("### 📈 Item Performance Insights")
        
        if not top_10_items.empty:
            total_top10_sales = top_10_items["Sales"].sum()
            total_sales = filtered_data["Sales"].sum()
            top10_percentage = (total_top10_sales / total_sales * 100) if total_sales > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Top 10 Contribution",
                    f"{top10_percentage:.1f}%",
                    f"SAR {total_top10_sales:,.0f}"
                )
            
            with col2:
                avg_margin = top_10_items["Margin_Pct"].mean()
                st.metric(
                    "Avg Margin (Top 10)",
                    f"{avg_margin:.1f}%",
                    "Profitability"
                )
            
            with col3:
                best_seller = top_10_items.iloc[0]["Item"]
                best_sales = top_10_items.iloc[0]["Sales"]
                st.metric(
                    "Best Seller",
                    best_seller[:20] + "..." if len(best_seller) > 20 else best_seller,
                    f"SAR {best_sales:,.0f}"
                )
            
            with col4:
                total_qty = top_10_items["Qty"].sum()
                st.metric(
                    "Units Sold (Top 10)",
                    f"{total_qty:,.0f}",
                    "Quantity"
                )

def render_insights_tab(filtered_data: pd.DataFrame, kpis: Dict[str, Any]):
    """Render the Business Insights tab content"""
    st.markdown("## 💡 Business Intelligence")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🎯 Key Performance Insights")
        
        insights = AnalyticsEngine.generate_insights(filtered_data, kpis)
        if insights:
            for insight in insights:
                st.markdown(f"- {insight}")
        else:
            st.write("No insights available for the current selection")
    
    with col2:
        st.markdown("### 📊 Performance Scorecard")
        
        performance_metrics = [
            ("Sales Target Achievement", f"{kpis['sales_variance_pct']:+.1f}%", 
             "✅" if kpis['sales_variance_pct'] >= 0 else "⚠️"),
            ("Gross Margin", f"{kpis['gross_margin']:.1f}%", 
             "✅" if kpis['gross_margin'] >= 25 else "⚠️"),
            ("Conversion Rate", f"{kpis['conversion_rate']:.1f}%", 
             "✅" if kpis['conversion_rate'] >= config.TARGET_THRESHOLDS["conversion_rate"] else "⚠️"),
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

def render_data_tab(filtered_data: pd.DataFrame, column_mapping: Dict[str, str]):
    """Render the Data Management tab content"""
    st.markdown("## 📊 Data Management")
    
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
    
    with col4:
        st.markdown("**🔗 Export Options**")
        
        # Excel export
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            filtered_data.to_excel(writer, sheet_name='Complete Data', index=False)
        
        st.download_button(
            label="📥 Download Excel",
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
    
    # Raw data preview
    st.markdown("### 📋 Data Preview")
    st.dataframe(filtered_data.head(100), use_container_width=True)

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
                    Comprehensive Business Intelligence & Performance Analytics
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
    
    # Load data
    @st.cache_data(ttl=0, show_spinner=False)
    def load_data(url: str) -> pd.DataFrame:
        csv_url = DataProcessor.convert_gsheet_url(url)
        return pd.read_csv(csv_url)
    
    try:
        with st.spinner("Loading data..."):
            raw_data = load_data(config.GSHEET_URL)
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        st.stop()
    
    # Standardize data
    data, column_mapping = DataProcessor.standardize_dataframe(raw_data)
    
    # Filters in main area (before tabs)
    st.markdown("### 🔍 Data Filters")
    col1, col2, col3 = st.columns([2, 2, 6])
    
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
    
    # Calculate KPIs
    kpis = AnalyticsEngine.calculate_kpis(filtered_data)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", 
        "💰 Financial", 
        "📈 Sales", 
        "⭐ Top Items", 
        "💡 Insights", 
        "📊 Data"
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
        render_data_tab(filtered_data, column_mapping)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.info("Please refresh the page or contact support if the issue persists.")
        st.stop()
