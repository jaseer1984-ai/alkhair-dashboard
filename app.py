import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Daily MTD Dashboard")

# Upload Excel file
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file, sheet_name="Daily Input", engine="openpyxl")
    df = df.dropna(subset=["Brand"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Summary metrics
    st.subheader("Summary Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sales Achieved", f"{df['Sales Achieved'].sum():,.0f}")
    col2.metric("Total ABV Achieved", f"{df['ABV Achieved'].sum():,.0f}")
    col3.metric("Total NOB Achieved", f"{df['NOB Achieved'].sum():,.0f}")

    # Filters
    brands = st.multiselect("Select Brands", df["Brand"].unique(), default=df["Brand"].unique())
    filtered_df = df[df["Brand"].isin(brands)]

    # Charts
    st.plotly_chart(px.bar(filtered_df, x="Brand", y=["Sales Target", "Sales Achieved"], barmode="group", title="Sales Target vs Achieved"))
    st.plotly_chart(px.bar(filtered_df, x="Brand", y=["ABV Target", "ABV Achieved"], barmode="group", title="ABV Target vs Achieved"))
    st.plotly_chart(px.bar(filtered_df, x="Brand", y=["NOB Target", "NOB Achieved"], barmode="group", title="NOB Target vs Achieved"))
else:
    st.info("Please upload an Excel file to view the dashboard.")
