# app.py — Sales Dashboard for INPUT DATA.xlsx
# -------------------------------------------
# ✅ Features:
#    - Date range filter
#    - Branch filter
#    - Daily report (by DATE & BRANCH)
#    - Monthly report (by Month & BRANCH)
#    - Actual vs Target (tables + charts)
#    - Quick Insights section
#    - Numbers formatted with commas
#
# 📂 Files required in same folder:
#    - app.py
#    - INPUT DATA.xlsx
#
# How to run locally:
#    1) pip install streamlit pandas altair
#    2) streamlit run app.py

import pandas as pd
import streamlit as st
import altair as alt

# ---------- CONFIG ----------
st.set_page_config(
    page_title="Sales Dashboard",
    layout="wide",
)

# ---------- DATA LOADER ----------
@st.cache_data
def load_data():
    # Excel file must be in same directory as this app
    df = pd.read_excel("INPUT DATA.xlsx")
    # Clean up
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["BRANCH"] = df["BRANCH"].astype(str).str.strip()
    df["Month"] = df["DATE"].dt.to_period("M").astype(str)  # e.g. '2025-11'
    return df

df = load_data()

if df.empty:
    st.error("No data found in INPUT DATA.xlsx")
    st.stop()

# ---------- SIDEBAR FILTERS ----------
st.sidebar.title("Filters")

min_date = df["DATE"].min().date()
max_date = df["DATE"].max().date()

date_range = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, tuple) or isinstance(date_range, list):
    start_date, end_date = date_range
else:
    # If user selects single date
    start_date = date_range
    end_date = date_range

branches = sorted(df["BRANCH"].dropna().unique())
selected_branches = st.sidebar.multiselect(
    "Select branch(es)",
    options=branches,
    default=branches,
)

report_type = st.sidebar.radio(
    "Report Type",
    options=["Daily", "Monthly"],
    index=0,
)

# ---------- APPLY FILTERS ----------
mask = (
    (df["DATE"].dt.date >= start_date)
    & (df["DATE"].dt.date <= end_date)
    & (df["BRANCH"].isin(selected_branches))
)

df_filtered = df.loc[mask].copy()

if df_filtered.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# ---------- HELPER FUNCTIONS ----------
def safe_sum(series):
    return series.fillna(0).sum()

def format_number(x, decimals=2):
    if pd.isna(x):
        return "-"
    return f"{x:,.{decimals}f}"

def build_kpis(df_):
    total_sales_target = safe_sum(df_["Sales Target"])
    total_sales_ach = safe_sum(df_["Sales Achieved"])
    total_nob_target = safe_sum(df_["NOB Target"])
    total_nob_ach = safe_sum(df_["NOB Achieved"])
    total_mtd_target = safe_sum(df_["MTD Sales Target"])
    total_mtd_ach = safe_sum(df_["MTD Sales Achieved"])

    sales_ach_pct = (total_sales_ach / total_sales_target * 100) if total_sales_target > 0 else 0
    nob_ach_pct = (total_nob_ach / total_nob_target * 100) if total_nob_target > 0 else 0
    mtd_ach_pct = (total_mtd_ach / total_mtd_target * 100) if total_mtd_target > 0 else 0

    return {
        "total_sales_target": total_sales_target,
        "total_sales_ach": total_sales_ach,
        "sales_ach_pct": sales_ach_pct,
        "total_nob_target": total_nob_target,
        "total_nob_ach": total_nob_ach,
        "nob_ach_pct": nob_ach_pct,
        "total_mtd_target": total_mtd_target,
        "total_mtd_ach": total_mtd_ach,
        "mtd_ach_pct": mtd_ach_pct,
    }

def daily_summary(df_):
    # Group by DATE & BRANCH
    grp = df_.groupby(["DATE", "BRANCH"], as_index=False).agg(
        {
            "Sales Target": "sum",
            "Sales Achieved": "sum",
            "NOB Target": "sum",
            "NOB Achieved": "sum",
            "ABV Target": "mean",
            "ABV Achieved": "mean",
        }
    )
    grp["Sales Gap"] = grp["Sales Achieved"] - grp["Sales Target"]
    grp["Sales Achievement %"] = grp.apply(
        lambda r: (r["Sales Achieved"] / r["Sales Target"] * 100) if r["Sales Target"] > 0 else 0,
        axis=1,
    )
    grp["NOB Achievement %"] = grp.apply(
        lambda r: (r["NOB Achieved"] / r["NOB Target"] * 100) if r["NOB Target"] > 0 else 0,
        axis=1,
    )
    # Sort by DATE then BRANCH
    grp = grp.sort_values(["DATE", "BRANCH"])
    return grp

def monthly_summary(df_):
    grp = df_.groupby(["Month", "BRANCH"], as_index=False).agg(
        {
            "Sales Target": "sum",
            "Sales Achieved": "sum",
            "NOB Target": "sum",
            "NOB Achieved": "sum",
            "ABV Target": "mean",
            "ABV Achieved": "mean",
        }
    )
    grp["Sales Gap"] = grp["Sales Achieved"] - grp["Sales Target"]
    grp["Sales Achievement %"] = grp.apply(
        lambda r: (r["Sales Achieved"] / r["Sales Target"] * 100) if r["Sales Target"] > 0 else 0,
        axis=1,
    )
    grp["NOB Achievement %"] = grp.apply(
        lambda r: (r["NOB Achieved"] / r["NOB Target"] * 100) if r["NOB Target"] > 0 else 0,
        axis=1,
    )
    grp = grp.sort_values(["Month", "BRANCH"])
    return grp

def quick_insights(df_):
    insights = []

    kpis = build_kpis(df_)

    # Overall sales performance
    diff = kpis["total_sales_ach"] - kpis["total_sales_target"]
    if diff >= 0:
        insights.append(
            f"Overall sales are ABOVE target by {format_number(diff)} "
            f"({format_number(kpis['sales_ach_pct'])}%)."
        )
    else:
        insights.append(
            f"Overall sales are BELOW target by {format_number(abs(diff))} "
            f"({format_number(kpis['sales_ach_pct'])}%)."
        )

    # NOB performance
    nob_diff = kpis["total_nob_ach"] - kpis["total_nob_target"]
    if nob_diff >= 0:
        insights.append(
            f"Number of bills (NOB) is ABOVE target by {format_number(nob_diff)} "
            f"({format_number(kpis['nob_ach_pct'])}%)."
        )
    else:
        insights.append(
            f"Number of bills (NOB) is BELOW target by {format_number(abs(nob_diff))} "
            f"({format_number(kpis['nob_ach_pct'])}%)."
        )

    # Branch wise best/worst
    if len(df_["BRANCH"].unique()) > 1:
        branch_perf = df_.groupby("BRANCH", as_index=False).agg(
            {
                "Sales Target": "sum",
                "Sales Achieved": "sum",
            }
        )
        branch_perf["Sales Achievement %"] = branch_perf.apply(
            lambda r: (r["Sales Achieved"] / r["Sales Target"] * 100)
            if r["Sales Target"] > 0 else 0,
            axis=1,
        )
        best = branch_perf.sort_values("Sales Achievement %", ascending=False).iloc[0]
        worst = branch_perf.sort_values("Sales Achievement %", ascending=True).iloc[0]

        insights.append(
            f"Best performing branch: **{best['BRANCH']}** "
            f"with {format_number(best['Sales Achievement %'])}% achievement."
        )
        insights.append(
            f"Lowest performing branch: **{worst['BRANCH']}** "
            f"with {format_number(worst['Sales Achievement %'])}% achievement."
        )

    return insights

def show_table(df_, title):
    st.subheader(title)

    # Prepare formatted copy for display
    df_display = df_.copy()
    # Format dates as yyyy-mm-dd
    if "DATE" in df_display.columns:
        df_display["DATE"] = df_display["DATE"].dt.strftime("%Y-%m-%d")

    num_cols_2d = [
        c
        for c in df_display.columns
        if any(
            key in c
            for key in [
                "Sales Target",
                "Sales Achieved",
                "Sales Gap",
                "NOB Target",
                "NOB Achieved",
                "ABV Target",
                "ABV Achieved",
            ]
        )
    ]

    num_cols_pct = [c for c in df_display.columns if "Achievement %" in c]

    for c in num_cols_2d:
        df_display[c] = df_display[c].apply(lambda x: format_number(x, 2))
    for c in num_cols_pct:
        df_display[c] = df_display[c].apply(lambda x: format_number(x, 2))

    st.dataframe(df_display, use_container_width=True)

# ---------- TITLE ----------
st.title("📊 Sales Dashboard — Daily & Monthly Reports")

st.caption(
    f"Filtered from **{start_date}** to **{end_date}** "
    f"for branch(es): {', '.join(selected_branches)}"
)

# ---------- KPI SECTION ----------
st.markdown("### 🔑 Key Metrics (Filtered Period)")

kpis = build_kpis(df_filtered)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        "Total Sales Target",
        format_number(kpis["total_sales_target"]),
    )
    st.metric(
        "Total Sales Achieved",
        format_number(kpis["total_sales_ach"]),
        delta=format_number(kpis["total_sales_ach"] - kpis["total_sales_target"]),
    )
with col2:
    st.metric(
        "Sales Achievement %",
        f"{format_number(kpis['sales_ach_pct'])}%",
    )
    st.metric(
        "Total NOB Target",
        format_number(kpis["total_nob_target"], 0),
    )
with col3:
    st.metric(
        "Total NOB Achieved",
        format_number(kpis["total_nob_ach"], 0),
        delta=format_number(kpis["total_nob_ach"] - kpis["total_nob_target"], 0),
    )
    st.metric(
        "MTD Sales Achievement %",
        f"{format_number(kpis['mtd_ach_pct'])}%",
    )

# ---------- QUICK INSIGHTS ----------
st.markdown("### 💡 Quick Insights")
for line in quick_insights(df_filtered):
    st.write("• " + line)

st.markdown("---")

# ---------- REPORTS (DAILY / MONTHLY) ----------
if report_type == "Daily":
    summary_df = daily_summary(df_filtered)
    show_table(summary_df, "🗓 Daily Report (by Date & Branch)")

    # Actual vs Target chart (Daily)
    st.markdown("#### 📈 Daily Sales — Actual vs Target")
    chart_df = summary_df[["DATE", "BRANCH", "Sales Target", "Sales Achieved"]].copy()
    chart_df = chart_df.melt(
        id_vars=["DATE", "BRANCH"],
        value_vars=["Sales Target", "Sales Achieved"],
        var_name="Type",
        value_name="Amount",
    )
    chart_df["Amount"] = chart_df["Amount"].fillna(0)

    daily_chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x="DATE:T",
            y="Amount:Q",
            color="Type:N",
            column="BRANCH:N",
            tooltip=["DATE:T", "BRANCH:N", "Type:N", "Amount:Q"],
        )
        .interactive()
    )
    st.altair_chart(daily_chart, use_container_width=True)

else:
    summary_df = monthly_summary(df_filtered)
    show_table(summary_df, "📆 Monthly Report (by Month & Branch)")

    # Actual vs Target chart (Monthly)
    st.markdown("#### 📈 Monthly Sales — Actual vs Target")
    chart_df = summary_df[["Month", "BRANCH", "Sales Target", "Sales Achieved"]].copy()
    chart_df = chart_df.melt(
        id_vars=["Month", "BRANCH"],
        value_vars=["Sales Target", "Sales Achieved"],
        var_name="Type",
        value_name="Amount",
    )
    chart_df["Amount"] = chart_df["Amount"].fillna(0)

    monthly_chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x="Month:N",
            y="Amount:Q",
            color="Type:N",
            column="BRANCH:N",
            tooltip=["Month:N", "BRANCH:N", "Type:N", "Amount:Q"],
        )
        .interactive()
    )
    st.altair_chart(monthly_chart, use_container_width=True)
