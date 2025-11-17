import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Daily MTD Dashboard", layout="wide")
st.title("📊 Daily MTD Dashboard")

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file, sheet_name="Daily Input", engine="openpyxl")
    df = df.dropna(subset=["Brand"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date

    # Sidebar Filters
    st.sidebar.header("Filters")
    min_date, max_date = df["Date"].min(), df["Date"].max()
    date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])

    # Handle single or range selection
    if isinstance(date_range, tuple) or len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range

    brands = st.sidebar.multiselect("Select Brands", df["Brand"].unique(), default=df["Brand"].unique())

    # Apply filters
    filtered_df = df[
        (df["Date"] >= start_date) & (df["Date"] <= end_date) &
        (df["Brand"].isin(brands))
    ]

    # KPI Calculations
    sales_pct = (filtered_df["Sales Achieved"].sum() / filtered_df["Sales Target"].sum() * 100) if filtered_df["Sales Target"].sum() > 0 else 0
    abv_pct = (filtered_df["ABV Achieved"].sum() / filtered_df["ABV Target"].sum() * 100) if filtered_df["ABV Target"].sum() > 0 else 0
    nob_pct = (filtered_df["NOB Achieved"].sum() / filtered_df["NOB Target"].sum() * 100) if filtered_df["NOB Target"].sum() > 0 else 0

    def color_for_kpi(value):
        if value >= 90:
            return "✅"
        elif value >= 70:
            return "🟠"
        else:
            return "🔴"

    st.subheader("📈 Key Performance Indicators")
    col1, col2, col3 = st.columns(3)
    col1.metric(f"{color_for_kpi(sales_pct)} Sales Achieved", f"{filtered_df['Sales Achieved'].sum():,.0f}", f"{sales_pct:.1f}%")
    col2.metric(f"{color_for_kpi(abv_pct)} ABV Achieved", f"{filtered_df['ABV Achieved'].sum():,.0f}", f"{abv_pct:.1f}%")
    col3.metric(f"{color_for_kpi(nob_pct)} NOB Achieved", f"{filtered_df['NOB Achieved'].sum():,.0f}", f"{nob_pct:.1f}%")

    st.markdown("---")

    # Charts
    st.plotly_chart(px.bar(filtered_df, x="Brand", y=["Sales Target", "Sales Achieved"], barmode="group",
                           title="Sales Target vs Achieved", color_discrete_sequence=["#636EFA", "#EF553B"]), use_container_width=True)
    st.plotly_chart(px.bar(filtered_df, x="Brand", y=["ABV Target", "ABV Achieved"], barmode="group",
                           title="ABV Target vs Achieved", color_discrete_sequence=["#00CC96", "#AB63FA"]), use_container_width=True)
    st.plotly_chart(px.bar(filtered_df, x="Brand", y=["NOB Target", "NOB Achieved"], barmode="group",
                           title="NOB Target vs Achieved", color_discrete_sequence=["#FFA15A", "#19D3F3"]), use_container_width=True)

    # Trend Line Chart
    st.subheader("📅 Daily Performance Trend")
    trend_df = filtered_df.groupby("Date")[["Sales Achieved", "ABV Achieved", "NOB Achieved"]].sum().reset_index()
    fig_trend = px.line(trend_df, x="Date", y=["Sales Achieved", "ABV Achieved", "NOB Achieved"],
                        title="Daily Achieved Trend", markers=True)
    fig_trend.update_xaxes(dtick="D")
    st.plotly_chart(fig_trend, use_container_width=True)

    st.download_button("Download Filtered Data", filtered_df.to_csv(index=False), "filtered_data.csv", "text/csv")
else:
    st.info("Please upload an Excel file to view the dashboard.")
