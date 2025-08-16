# app.py — Enhanced Alkhair Family Market Daily Dashboard
# Improved version with better structure, performance, and features

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
    PAGE_TITLE = "Alkhair Family Market — Daily Dashboard"
    LAYOUT = "wide"
    REFRESH_MINUTES = 1
    GSHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSW_ui1m_393ipZv8NAliu1rly6zeifFxMfOWpQF17hjVIDa9Ll8PiGCaz8gTRkMQ/pub?output=csv"
    
    # Column aliases for flexible data mapping
    COLUMN_MAPPINGS = {
        "Date": ["date", "bill_date", "txn_date", "invoice_date", "posting_date"],
        "Store": ["store", "branch", "location", "shop"],
        "Sales": ["sales", "amount", "net_sales", "revenue", "netamount", "net_amount"],
        "COGS": ["cogs", "cost", "purchase_cost", "cost_of_goods"],
        "Qty": ["qty", "quantity", "qty_sold", "quantity_sold", "units", "pcs", "pieces", "units_sold", "qnty", "qnt", "qty."],
        "Tickets": ["tickets", "bills", "invoice_count", "transactions", "bills_count"],
        "Category": ["category", "cat"],
        "Item": ["item", "product", "sku", "item_name"],
        "InventoryQty": ["inventoryqty", "inventory_qty", "stock_qty", "stockqty", "stock_quantity"],
        "InventoryValue": ["inventoryvalue", "inventory_value", "stock_value", "stockvalue"]
    }
    
    # KPI colors for tiles
    KPI_COLORS = {
        "tickets": "#a7f3d0",
        "sales": "#fde68a",
        "profit": "#d9f99d",
        "conversion": "#bae6fd",
        "avg_sales": "#fed7aa"
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
    """Apply custom CSS styling for the dashboard"""
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
        padding: 10px 0;
        margin-bottom: 14px;
        background: #f8fafc;
        border-bottom: 1px solid var(--border);
    }
    
    .kpi-tile {
        border-radius: 14px;
        padding: 14px 16px;
        color: #0b1220;
        font-weight: 800;
        box-shadow: var(--shadow);
        transition: transform 0.2s;
    }
    
    .kpi-tile:hover {
        transform: translateY(-2px);
    }
    
    .metric-trend {
        font-size: 0.9rem;
        margin-top: 4px;
    }
    
    .trend-up { color: #22c55e; }
    .trend-down { color: #ef4444; }
    
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
                elif target_col in ["Sales", "COGS", "Qty", "Tickets", "InventoryQty", "InventoryValue"]:
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
        """Calculate key performance indicators"""
        kpis = {}
        
        # Basic KPIs
        kpis["total_sales"] = float(df["Sales"].sum())
        kpis["total_tickets"] = float(df["Tickets"].sum()) if df["Tickets"].notna().any() else 0
        kpis["total_qty"] = float(df["Qty"].sum())
        
        # Gross profit metrics
        if df["GrossProfit"].notna().any():
            kpis["gross_profit"] = float(df["GrossProfit"].sum())
            kpis["gross_margin"] = (kpis["gross_profit"] / kpis["total_sales"] * 100) if kpis["total_sales"] > 0 else 0
        else:
            kpis["gross_profit"] = 0
            kpis["gross_margin"] = 0
        
        # Average metrics
        kpis["avg_basket"] = kpis["total_sales"] / kpis["total_tickets"] if kpis["total_tickets"] > 0 else 0
        kpis["conversion_rate"] = (kpis["total_qty"] / kpis["total_tickets"] * 100) if kpis["total_tickets"] > 0 else 0
        
        # Date-based metrics
        if df["Date"].notna().any():
            unique_days = df["Date"].dt.date.nunique()
            kpis["days_count"] = unique_days
            kpis["avg_daily_sales"] = kpis["total_sales"] / unique_days if unique_days > 0 else 0
        else:
            kpis["days_count"] = 0
            kpis["avg_daily_sales"] = 0
        
        # Inventory metrics
        if df["InventoryValue"].notna().any():
            kpis["inventory_value"] = float(df["InventoryValue"].sum())
            kpis["inventory_turnover"] = kpis["total_sales"] / kpis["inventory_value"] if kpis.get("inventory_value", 0) > 0 else 0
        
        return kpis
    
    @staticmethod
    def generate_insights(df: pd.DataFrame, kpis: Dict[str, Any]) -> List[str]:
        """Generate intelligent insights from data"""
        insights = []
        
        # Store performance
        if not df.empty and df["Store"].notna().any():
            store_sales = df.groupby("Store")["Sales"].sum().sort_values(ascending=False)
            if not store_sales.empty:
                top_store = store_sales.index[0]
                top_sales = store_sales.iloc[0]
                insights.append(f"🏆 Top performing store: **{top_store}** (SAR {top_sales:,.0f})")
                
                if len(store_sales) > 1:
                    bottom_store = store_sales.index[-1]
                    bottom_sales = store_sales.iloc[-1]
                    insights.append(f"📉 Lowest performing store: **{bottom_store}** (SAR {bottom_sales:,.0f})")
        
        # Time-based trends
        if df["Date"].notna().any() and len(df["Date"].unique()) > 1:
            daily_sales = df.groupby(df["Date"].dt.date)["Sales"].sum()
            if len(daily_sales) >= 2:
                recent_change = ((daily_sales.iloc[-1] - daily_sales.iloc[-2]) / daily_sales.iloc[-2] * 100)
                trend_icon = "📈" if recent_change >= 0 else "📉"
                insights.append(f"{trend_icon} Day-over-day change: **{recent_change:+.1f}%**")
        
        # Category performance
        if df["Category"].notna().any():
            cat_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
            if not cat_sales.empty:
                top_cat = cat_sales.index[0]
                top_cat_sales = cat_sales.iloc[0]
                insights.append(f"🛒 Top category: **{top_cat}** (SAR {top_cat_sales:,.0f})")
        
        # Efficiency metrics
        if kpis.get("gross_margin", 0) > 0:
            insights.append(f"💰 Gross margin: **{kpis['gross_margin']:.1f}%**")
        
        if kpis.get("avg_basket", 0) > 0:
            insights.append(f"🛍️ Average basket size: **SAR {kpis['avg_basket']:.2f}**")
        
        if kpis.get("conversion_rate", 0) > 0:
            insights.append(f"🎯 Sales conversion rate: **{kpis['conversion_rate']:.1f}%**")
        
        return insights

# =========================
# VISUALIZATION ENGINE
# =========================
class VisualizationEngine:
    """Handles all chart and visualization creation"""
    
    @staticmethod
    def create_combo_chart(daily_data: pd.DataFrame) -> go.Figure:
        """Create combination chart with sales and conversion"""
        fig = go.Figure()
        
        # Sales bars
        fig.add_trace(go.Bar(
            x=daily_data["Day"],
            y=daily_data["Sales"],
            name="Sales",
            marker_color="#3b82f6",
            opacity=0.95,
            text=daily_data["Sales"].round(0),
            textposition="outside",
            texttemplate='%{text:,.0f}'
        ))
        
        # Conversion line
        fig.add_trace(go.Scatter(
            x=daily_data["Day"],
            y=daily_data["Conv"],
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
            margin=dict(l=10, r=10, t=50, b=10),
            hovermode='x unified'
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
            margin=dict(l=10, r=100, t=40, b=10)
        )
        
        return fig
    
    @staticmethod
    def create_store_comparison(store_data: pd.DataFrame) -> go.Figure:
        """Create horizontal bar chart for store comparison"""
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=store_data["Store"],
            x=store_data["Sales"],
            orientation='h',
            marker_color='#3b82f6',
            text=store_data["Sales"].round(0),
            textposition='outside',
            texttemplate='SAR %{text:,.0f}'
        ))
        
        fig.update_layout(
            title="Store Performance Comparison",
            xaxis=dict(title="Sales (SAR)"),
            yaxis=dict(title=""),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            margin=dict(l=10, r=10, t=40, b=10),
            height=300
        )
        
        return fig

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
    
    # Header
    st.markdown(f"""
    <div class="brandbar">
        <div style="display:flex;align-items:center;justify-content:space-between;">
            <div style="font-weight:800;font-size:1.2rem;">
                🏪 Alkhair Family Market — Daily Dashboard
            </div>
            <div style="font-size:0.85rem;color:#6b7280;">
                Auto refresh: every {config.REFRESH_MINUTES} min • 
                Last update: {datetime.now().strftime('%H:%M:%S')}
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
    
    # Date range filter
    if data["Date"].notna().any():
        date_min = data["Date"].min().date()
        date_max = data["Date"].max().date()
        
        col1, col2, col3 = st.columns([2, 2, 8])
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
    
    # Display KPI tiles
    st.markdown("<br>", unsafe_allow_html=True)
    kpi_tiles = [
        ("📊 Total Tickets", f"{kpis['total_tickets']:,.0f}", config.KPI_COLORS["tickets"]),
        ("💰 Total Sales", f"SAR {kpis['total_sales']:,.0f}", config.KPI_COLORS["sales"]),
        ("📈 Gross Profit", f"SAR {kpis['gross_profit']:,.0f}", config.KPI_COLORS["profit"]),
        ("🎯 Conversion Rate", f"{kpis['conversion_rate']:.1f}%", config.KPI_COLORS["conversion"]),
        ("📅 Avg Daily Sales", f"SAR {kpis['avg_daily_sales']:,.0f}", config.KPI_COLORS["avg_sales"])
    ]
    
    cols = st.columns(len(kpi_tiles))
    for (label, value, color), col in zip(kpi_tiles, cols):
        col.markdown(f"""
        <div class="kpi-tile" style="background:{color}">
            <div style="font-size:14px; color:#334155; font-weight:700;">{label}</div>
            <div style="font-size:28px; line-height:1; margin-top:4px; color:#0f172a;">{value}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Monthly analysis section
    st.markdown("<br><br>", unsafe_allow_html=True)
    
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
                combo_chart = VisualizationEngine.create_combo_chart(daily_agg)
                st.plotly_chart(combo_chart, use_container_width=True)
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
        st.markdown("<br>", unsafe_allow_html=True)
        store_sales = filtered_data.groupby("Store")["Sales"].sum().sort_values(ascending=True).reset_index()
        store_chart = VisualizationEngine.create_store_comparison(store_sales)
        st.plotly_chart(store_chart, use_container_width=True)
    
    # Insights section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 💡 Key Insights")
    
    insights = AnalyticsEngine.generate_insights(filtered_data, kpis)
    if insights:
        for insight in insights:
            st.write(insight)
    else:
        st.write("No insights available for the current selection")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Top items table
    if filtered_data["Item"].notna().any():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### ⭐ Top Performing Items")
        
        top_items = (filtered_data.groupby("Item")
                    .agg({"Sales": "sum", "Qty": "sum", "GrossProfit": "sum"})
                    .sort_values("Sales", ascending=False)
                    .head(15)
                    .reset_index())
        
        # Format for display
        top_items["Sales"] = top_items["Sales"].apply(lambda x: f"SAR {x:,.0f}")
        top_items["Qty"] = top_items["Qty"].astype(int)
        if top_items["GrossProfit"].notna().any():
            top_items["GrossProfit"] = top_items["GrossProfit"].apply(lambda x: f"SAR {x:,.0f}")
        
        st.dataframe(top_items, use_container_width=True, height=400)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Export functionality
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([8, 1, 1])
    
    with col2:
        # Excel export
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            filtered_data.to_excel(writer, sheet_name='Data', index=False)
            
            # Add summary sheets
            if not daily_agg.empty:
                daily_agg.to_excel(writer, sheet_name='Daily Summary', index=False)
            
            if filtered_data["Store"].notna().any():
                store_summary = filtered_data.groupby("Store").agg({
                    "Sales": "sum",
                    "GrossProfit": "sum",
                    "Tickets": "sum"
                }).reset_index()
                store_summary.to_excel(writer, sheet_name='Store Summary', index=False)
            
            if filtered_data["Category"].notna().any():
                category_summary = filtered_data.groupby("Category").agg({
                    "Sales": "sum",
                    "Qty": "sum"
                }).reset_index()
                category_summary.to_excel(writer, sheet_name='Category Summary', index=False)
        
        st.download_button(
            label="📥 Excel",
            data=buffer.getvalue(),
            file_name=f"Alkhair_Dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col3:
        # CSV export
        csv_buffer = io.StringIO()
        filtered_data.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📄 CSV",
            data=csv_buffer.getvalue(),
            file_name=f"Alkhair_Data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Footer with metadata
    st.markdown("<br><hr>", unsafe_allow_html=True)
    with st.expander("📊 Data Information"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Data Range**")
            if filtered_data["Date"].notna().any():
                st.write(f"From: {filtered_data['Date'].min().strftime('%Y-%m-%d')}")
                st.write(f"To: {filtered_data['Date'].max().strftime('%Y-%m-%d')}")
                st.write(f"Days: {filtered_data['Date'].dt.date.nunique()}")
            else:
                st.write("No date information available")
        
        with col2:
            st.markdown("**Data Quality**")
            total_rows = len(filtered_data)
            st.write(f"Total Records: {total_rows:,}")
            st.write(f"Stores: {filtered_data['Store'].nunique()}")
            st.write(f"Categories: {filtered_data['Category'].nunique()}")
        
        with col3:
            st.markdown("**Column Mapping**")
            mapping_df = pd.DataFrame({
                "Target": list(column_mapping.keys())[:5],
                "Source": list(column_mapping.values())[:5]
            })
            st.dataframe(mapping_df, hide_index=True)

# =========================
# ERROR HANDLING
# =========================
class ErrorHandler:
    """Centralized error handling for the application"""
    
    @staticmethod
    def handle_data_error(error: Exception) -> None:
        """Handle data loading errors"""
        st.error(f"❌ Data Loading Error: {str(error)}")
        st.info("Please check:")
        st.write("• The Google Sheets URL is correct and accessible")
        st.write("• The sheet is published to the web")
        st.write("• Your internet connection is stable")
        
    @staticmethod
    def handle_processing_error(error: Exception) -> None:
        """Handle data processing errors"""
        st.warning(f"⚠️ Processing Warning: {str(error)}")
        st.info("The dashboard will continue with available data")

# =========================
# PERFORMANCE MONITORING
# =========================
class PerformanceMonitor:
    """Monitor and log application performance"""
    
    @staticmethod
    def log_metrics(data_size: int, processing_time: float) -> None:
        """Log performance metrics to console"""
        if st.secrets.get("debug_mode", False):
            st.sidebar.markdown("### Debug Info")
            st.sidebar.write(f"Data rows: {data_size:,}")
            st.sidebar.write(f"Processing time: {processing_time:.2f}s")
            st.sidebar.write(f"Memory usage: {get_memory_usage():.2f} MB")

def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    import psutil
    import os
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        ErrorHandler.handle_data_error(e)
        st.stop()
