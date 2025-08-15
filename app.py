import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time

# ===================== CONFIG =====================
st.set_page_config(
    page_title="Sales Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====== WHITE BACKGROUND THEME ======
st.markdown("""
    <style>
    .stApp {
        background-color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ====== FILE UPLOAD ======
uploaded_file = st.sidebar.file_uploader("Upload your Excel file", type=["xlsx"])

# ====== AUTO REFRESH ======
REFRESH_INTERVAL = 60  # seconds
st_autorefresh = st.sidebar.checkbox("Auto Refresh", value=True)
if st_autorefresh:
    st_autorefresh_counter = st.experimental_rerun

# ====== DATE FILTER ======
today = datetime.today()
default_start = today - timedelta(days=30)
start_date, end_date = st.sidebar.date_input(
    "Select date range",
    value=[default_start, today]
)

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    # ====== DATE PARSING ======
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]

    # ====== KPIs ======
    total_calls = df['Calls'].sum() if 'Calls' in df.columns else 0
    total_sales = df['Sales'].sum() if 'Sales' in df.columns else 0
    sales_conversion = (total_sales / total_calls * 100) if total_calls > 0 else 0
    avg_sales_per_day = df.groupby('Date')['Sales'].sum().mean() if 'Sales' in df.columns else 0

    # ====== KPI CARDS ======
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("# of Calls", f"{total_calls:,}")
    kpi2.metric("Sales", f"{total_sales:,}")
    kpi3.metric("Sales Conversion", f"{sales_conversion:.1f}%")
    kpi4.metric("Average Sales Per Day", f"{avg_sales_per_day:.0f}")

    st.markdown("---")

    # ====== DAILY SALES CHART ======
    if 'Sales' in df.columns:
        daily = df.groupby('Date', as_index=False).sum()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=daily["Date"],
            y=daily["Sales"],
            name="Sales",
            marker_color="#4F8CFF",
            opacity=0.9
        ))
        if 'Conversion' in daily.columns:
            fig.add_trace(go.Scatter(
                x=daily["Date"],
                y=daily["Conversion"],
                name="Sales Conversion",
                yaxis="y2",
                mode="lines+markers",
                line=dict(color="#FFAA00", width=2)
            ))

        fig.update_layout(
            title="Daily Sales and Sales Conversion",
            xaxis_title="Date",
            yaxis_title="Sales",
            yaxis2=dict(
                title="Conversion %",
                overlaying="y",
                side="right"
            ),
            template="simple_white",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    # ====== MONTHLY SALES PIE ======
    if 'Sales' in df.columns:
        monthly = df.groupby(df['Date'].dt.strftime('%B'), as_index=False)['Sales'].sum()
        pie_chart = px.pie(
            monthly,
            names="Date",
            values="Sales",
            title="Monthly Sales",
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(pie_chart, use_container_width=True)

else:
    st.warning("Please upload an Excel file to view the dashboard.")
