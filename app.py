import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ===== SETTINGS =====
REFRESH_INTERVAL_MINUTES = 5  # Auto-refresh every X minutes
st.set_page_config(page_title="Sales Dashboard", layout="wide")

# ===== CSS for white background & Power BI style cards =====
st.markdown("""
    <style>
    .reportview-container {
        background-color: white;
    }
    .card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f9f9f9;
        box-shadow: 0px 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        font-family: Arial, sans-serif;
    }
    .card h3 {
        margin: 0;
        font-size: 18px;
        color: #555;
    }
    .card h2 {
        margin: 0;
        font-size: 28px;
        font-weight: bold;
        color: #000;
    }
    </style>
""", unsafe_allow_html=True)

# ===== Auto refresh =====
st_autorefresh = st.experimental_rerun  # Keeps code shorter

# ===== File Upload =====
uploaded_file = st.file_uploader("Upload your sales data (Excel/CSV)", type=["xlsx", "csv"])
if uploaded_file is not None:
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    # Ensure Date column
    if "Date" not in df.columns:
        st.error("Your file must contain a 'Date' column.")
        st.stop()
    
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # ===== Date Range Filter =====
    dmin = df["Date"].min().date()
    dmax = df["Date"].max().date()
    start_date, end_date = st.date_input(
        "Select date range",
        value=[dmin, dmax],
        min_value=dmin,
        max_value=dmax
    )
    if not isinstance(start_date, datetime):
        start_date = datetime.combine(start_date, datetime.min.time())
    if not isinstance(end_date, datetime):
        end_date = datetime.combine(end_date, datetime.max.time())

    df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

    # ===== KPI CARDS =====
    total_sales = df["Sales"].sum() if "Sales" in df.columns else 0
    total_calls = df["Calls"].sum() if "Calls" in df.columns else 0
    sales_conversion = (total_sales / total_calls * 100) if total_calls > 0 else 0
    avg_sales_day = total_sales / df["Date"].nunique()

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        st.markdown(f"<div class='card'><h3># of Calls</h3><h2>{int(total_calls):,}</h2></div>", unsafe_allow_html=True)
    with kpi_cols[1]:
        st.markdown(f"<div class='card'><h3>Sales</h3><h2>{int(total_sales):,}</h2></div>", unsafe_allow_html=True)
    with kpi_cols[2]:
        st.markdown(f"<div class='card'><h3>Sales Conversion</h3><h2>{sales_conversion:.1f}%</h2></div>", unsafe_allow_html=True)
    with kpi_cols[3]:
        st.markdown(f"<div class='card'><h3>Avg Sales Per Day</h3><h2>{int(avg_sales_day):,}</h2></div>", unsafe_allow_html=True)

    # ===== SALES CHART =====
    daily = df.groupby("Date").agg({"Sales": "sum"}).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=daily["Date"], y=daily["Sales"], name="Sales", marker_color="#4F8CFF"))
    fig.update_layout(
        title="Daily Sales",
        xaxis_title="Date",
        yaxis_title="Sales",
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please upload an Excel or CSV file with at least a 'Date' and 'Sales' column.")
