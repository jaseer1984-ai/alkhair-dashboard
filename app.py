import streamlit as st
import pandas as pd

# Page configuration
st.set_page_config(page_title="Daily & MTD Dashboard", layout="wide")
st.title("📊 Daily & MTD Dashboard")

# Upload Excel file
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
if uploaded_file:
    # Read and clean data
    df = pd.read_excel(uploaded_file, sheet_name="Daily Input", engine="openpyxl")
    df = df.dropna(subset=["Brand"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date

    # Sidebar Filters
    st.sidebar.header("Filters")
    min_date, max_date = df["Date"].min(), df["Date"].max()
    date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])

    # Handle single or range selection safely
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range

    brands = st.sidebar.multiselect("Select Brands", df["Brand"].unique(), default=df["Brand"].unique())

    # Apply filters
    filtered_df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date) & (df["Brand"].isin(brands))]

    # DAILY DATA (last selected date)
    daily_df = df[(df["Date"] == end_date) & (df["Brand"].isin(brands))]

    # MTD DATA (month of end_date)
    mtd_df = df[(pd.to_datetime(df["Date"]).month == pd.to_datetime(end_date).month) & (df["Brand"].isin(brands))]

    # KPI Calculation function
    def calc_kpis(data):
        sales_target = data["Sales Target"].sum()
        sales_achieved = data["Sales Achieved"].sum()
        abv_target = data["ABV Target"].sum()
        abv_achieved = data["ABV Achieved"].sum()
        nob_target = data["NOB Target"].sum()
        nob_achieved = data["NOB Achieved"].sum()

        sales_pct = (sales_achieved / sales_target * 100) if sales_target > 0 else 0
        abv_pct = (abv_achieved / abv_target * 100) if abv_target > 0 else 0
        nob_pct = (nob_achieved / nob_target * 100) if nob_target > 0 else 0

        return sales_achieved, sales_pct, abv_achieved, abv_pct, nob_achieved, nob_pct

    # Color indicator
    def color_for_kpi(value):
        if value >= 90:
            return "✅"
        elif value >= 70:
            return "🟠"
        else:
            return "🔴"

    # Combined Summary Section
    st.subheader("📌 Summary: Daily vs MTD")
    d_sales, d_sales_pct, d_abv, d_abv_pct, d_nob, d_nob_pct = calc_kpis(daily_df)
    m_sales, m_sales_pct, m_abv, m_abv_pct, m_nob, m_nob_pct = calc_kpis(mtd_df)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### Daily KPIs ({end_date})")
        st.metric(f"{color_for_kpi(d_sales_pct)} Sales", f"{d_sales:,.0f}", f"{d_sales_pct:.1f}%")
        st.metric(f"{color_for_kpi(d_abv_pct)} ABV", f"{d_abv:,.0f}", f"{d_abv_pct:.1f}%")
        st.metric(f"{color_for_kpi(d_nob_pct)} NOB", f"{d_nob:,.0f}", f"{d_nob_pct:.1f}%")
    with col2:
        st.markdown(f"### MTD KPIs (Month of {end_date})")
        st.metric(f"{color_for_kpi(m_sales_pct)} Sales", f"{m_sales:,.0f}", f"{m_sales_pct:.1f}%")
        st.metric(f"{color_for_kpi(m_abv_pct)} ABV", f"{m_abv:,.0f}", f"{m_abv_pct:.1f}%")
        st.metric(f"{color_for_kpi(m_nob_pct)} NOB", f"{m_nob:,.0f}", f"{m_nob_pct:.1f}%")

    st.markdown("---")

    # Tabs for detailed view
    tab1, tab2 = st.tabs(["📅 Daily Details", "📆 MTD Details"])

    with tab1:
        st.subheader(f"Daily Progress ({end_date})")
        st.progress(int(d_sales_pct))
        st.progress(int(d_abv_pct))
        st.progress(int(d_nob_pct))
        st.dataframe(daily_df)

    with tab2:
        st.subheader(f"MTD Progress (Month of {end_date})")
        st.progress(int(m_sales_pct))
        st.progress(int(m_abv_pct))
        st.progress(int(m_nob_pct))
        st.dataframe(mtd_df)

    # Download filtered data
    st.download_button("Download Filtered Data", filtered_df.to_csv(index=False), "filtered_data.csv", "text/csv")
else:
    st.info("Please upload an Excel file to view the dashboard.")
