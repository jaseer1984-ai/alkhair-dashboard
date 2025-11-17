import streamlit as st
import pandas as pd

# Page configuration
st.set_page_config(page_title="Daily MTD Dashboard", layout="wide")
st.title("📊 Daily MTD Dashboard")

# Upload Excel file
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
if uploaded_file:
    # Read and clean data
    df = pd.read_excel(uploaded_file, sheet_name="Daily Input", engine="openpyxl")
    df = df.dropna(subset=["Brand"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date  # Remove time

    # Sidebar Filters
    st.sidebar.header("Filters")
    min_date, max_date = df["Date"].min(), df["Date"].max()
    date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])

    # ✅ Handle single or range selection safely
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range  # Single date selected

    brands = st.sidebar.multiselect("Select Brands", df["Brand"].unique(), default=df["Brand"].unique())

    # Apply filters
    filtered_df = df[
        (df["Date"] >= start_date) & (df["Date"] <= end_date) &
        (df["Brand"].isin(brands))
    ]

    # KPI Calculations
    sales_target = filtered_df["Sales Target"].sum()
    sales_achieved = filtered_df["Sales Achieved"].sum()
    abv_target = filtered_df["ABV Target"].sum()
    abv_achieved = filtered_df["ABV Achieved"].sum()
    nob_target = filtered_df["NOB Target"].sum()
    nob_achieved = filtered_df["NOB Achieved"].sum()

    sales_pct = (sales_achieved / sales_target * 100) if sales_target > 0 else 0
    abv_pct = (abv_achieved / abv_target * 100) if abv_target > 0 else 0
    nob_pct = (nob_achieved / nob_target * 100) if nob_target > 0 else 0

    # Function for color indicator
    def color_for_kpi(value):
        if value >= 90:
            return "✅"
        elif value >= 70:
            return "🟠"
        else:
            return "🔴"

    # KPI Cards
    st.subheader("📈 Key Performance Indicators")
    col1, col2, col3 = st.columns(3)
    col1.metric(f"{color_for_kpi(sales_pct)} Sales Achieved", f"{sales_achieved:,.0f}", f"{sales_pct:.1f}%")
    col2.metric(f"{color_for_kpi(abv_pct)} ABV Achieved", f"{abv_achieved:,.0f}", f"{abv_pct:.1f}%")
    col3.metric(f"{color_for_kpi(nob_pct)} NOB Achieved", f"{nob_achieved:,.0f}", f"{nob_pct:.1f}%")

    st.markdown("---")

    # Progress Bars
    st.subheader("📊 Progress Toward Targets")
    st.write("Sales Progress")
    st.progress(int(sales_pct))
    st.write("ABV Progress")
    st.progress(int(abv_pct))
    st.write("NOB Progress")
    st.progress(int(nob_pct))

    st.markdown("---")

    # Styled Table with Conditional Formatting
    st.subheader("📋 Detailed Performance Table")
    styled_df = filtered_df.style.format({
        "Sales %": "{:.1%}",
        "ABV %": "{:.1%}",
        "NOB %": "{:.1%}"
    }).applymap(lambda v: 'color: green' if isinstance(v, (int, float)) and v >= 0.9 else 'color: red', subset=['Sales %','ABV %','NOB %'])
    st.dataframe(styled_df)

    # Download filtered data
    st.download_button("Download Filtered Data", filtered_df.to_csv(index=False), "filtered_data.csv", "text/csv")
else:
    st.info("Please upload an Excel file to view the dashboard.")
