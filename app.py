# app.py — Sales Dashboard with File Upload
# ========================================
# ✅ Features:
#   - Upload Excel file manually (no fixed path)
#   - Date range filter
#   - Branch filter
#   - Daily report (by DATE & BRANCH)
#   - Monthly report (by Month & BRANCH)
#   - Actual vs Target (Sales & NOB) with charts
#   - Quick Insights section
#   - Amounts formatted with commas
#
# 🧪 Run locally:
#   1) pip install streamlit pandas altair
#   2) streamlit run app.py
#
# Later you can replace the file_uploader part with Google Drive link.

import pandas as pd
import streamlit as st
import altair as alt

# ----------------- CONFIG -----------------
st.set_page_config(
    page_title="Sales Dashboard",
    layout="wide",
)

st.title("📊 Sales Dashboard — Daily & Monthly Reports")

st.write(
    "Upload your Excel file (e.g. **INPUT DATA.xlsx**) with columns like "
    "`DATE`, `BRANCH`, `Sales Target`, `Sales Achieved`, `NOB Target`, `NOB Achieved`, etc."
)

# ----------------- FILE UPLOAD -----------------
uploaded_file = st.file_uploader(
    "📁 Upload Excel file",
    type=["xlsx", "xls"],
    accept_multiple_files=False,
)

if not uploaded_file:
    st.info("Please upload an Excel file to see the dashboard.")
    st.stop()


@st.cache_data
def load_data(file_obj):
    df = pd.read_excel(file_obj)

    # Ensure DATE column
    if "DATE" not in df.columns:
        raise ValueError("Column 'DATE' not found in uploaded file.")

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"])

    # Ensure BRANCH column
    if "BRANCH" not in df.columns:
        raise ValueError("Column 'BRANCH' not found in uploaded file.")

    df["BRANCH"] = df["BRANCH"].astype(str).str.strip()
    df["Month"] = df["DATE"].dt.to_period("M").astype(str)  # e.g. '2025-11'
    return df


try:
    df = load_data(uploaded_file)
except Exception as e:
    st.error(f"❌ Error while reading file: {e}")
    st.stop()

if df.empty:
    st.error("Uploaded file has no usable data.")
    st.stop()

# ----------------- HELPERS -----------------
def safe_sum(series):
    return series.fillna(0).sum()


def format_number(x, decimals=2):
    if pd.isna(x):
        return "-"
    return f"{x:,.{decimals}f}"


def build_kpis(df_):
    cols = df_.columns

    sales_tgt = safe_sum(df_["Sales Target"]) if "Sales Target" in cols else 0
    sales_ach = safe_sum(df_["Sales Achieved"]) if "Sales Achieved" in cols else 0
    nob_tgt = safe_sum(df_["NOB Target"]) if "NOB Target" in cols else 0
    nob_ach = safe_sum(df_["NOB Achieved"]) if "NOB Achieved" in cols else 0
    mtd_tgt = safe_sum(df_["MTD Sales Target"]) if "MTD Sales Target" in cols else 0
    mtd_ach = safe_sum(df_["MTD Sales Achieved"]) if "MTD Sales Achieved" in cols else 0

    sales_ach_pct = (sales_ach / sales_tgt * 100) if sales_tgt > 0 else 0
    nob_ach_pct = (nob_ach / nob_tgt * 100) if nob_tgt > 0 else 0
    mtd_ach_pct = (mtd_ach / mtd_tgt * 100) if mtd_tgt > 0 else 0

    return {
        "total_sales_target": sales_tgt,
        "total_sales_ach": sales_ach,
        "sales_ach_pct": sales_ach_pct,
        "total_nob_target": nob_tgt,
        "total_nob_ach": nob_ach,
        "nob_ach_pct": nob_ach_pct,
        "total_mtd_target": mtd_tgt,
        "total_mtd_ach": mtd_ach,
        "mtd_ach_pct": mtd_ach_pct,
    }


def daily_summary(df_):
    cols = df_.columns

    agg_dict = {}
    if "Sales Target" in cols:
        agg_dict["Sales Target"] = "sum"
    if "Sales Achieved" in cols:
        agg_dict["Sales Achieved"] = "sum"
    if "NOB Target" in cols:
        agg_dict["NOB Target"] = "sum"
    if "NOB Achieved" in cols:
        agg_dict["NOB Achieved"] = "sum"
    if "ABV Target" in cols:
        agg_dict["ABV Target"] = "mean"
    if "ABV Achieved" in cols:
        agg_dict["ABV Achieved"] = "mean"

    grp = df_.groupby(["DATE", "BRANCH"], as_index=False).agg(agg_dict)

    # Sales Gap + Achievement %
    if {"Sales Target", "Sales Achieved"}.issubset(grp.columns):
        grp["Sales Gap"] = grp["Sales Achieved"] - grp["Sales Target"]
        grp["Sales Achievement %"] = grp.apply(
            lambda r: (r["Sales Achieved"] / r["Sales Target"] * 100)
            if r["Sales Target"] > 0 else 0,
            axis=1,
        )
    else:
        grp["Sales Gap"] = 0
        grp["Sales Achievement %"] = 0

    # NOB Achievement %
    if {"NOB Target", "NOB Achieved"}.issubset(grp.columns):
        grp["NOB Achievement %"] = grp.apply(
            lambda r: (r["NOB Achieved"] / r["NOB Target"] * 100)
            if r["NOB Target"] > 0 else 0,
            axis=1,
        )
    else:
        grp["NOB Achievement %"] = 0

    grp = grp.sort_values(["DATE", "BRANCH"])
    return grp


def monthly_summary(df_):
    cols = df_.columns

    agg_dict = {}
    if "Sales Target" in cols:
        agg_dict["Sales Target"] = "sum"
    if "Sales Achieved" in cols:
        agg_dict["Sales Achieved"] = "sum"
    if "NOB Target" in cols:
        agg_dict["NOB Target"] = "sum"
    if "NOB Achieved" in cols:
        agg_dict["NOB Achieved"] = "sum"
    if "ABV Target" in cols:
        agg_dict["ABV Target"] = "mean"
    if "ABV Achieved" in cols:
        agg_dict["ABV Achieved"] = "mean"

    grp = df_.groupby(["Month", "BRANCH"], as_index=False).agg(agg_dict)

    if {"Sales Target", "Sales Achieved"}.issubset(grp.columns):
        grp["Sales Gap"] = grp["Sales Achieved"] - grp["Sales Target"]
        grp["Sales Achievement %"] = grp.apply(
            lambda r: (r["Sales Achieved"] / r["Sales Target"] * 100)
            if r["Sales Target"] > 0 else 0,
            axis=1,
        )
    else:
        grp["Sales Gap"] = 0
        grp["Sales Achievement %"] = 0

    if {"NOB Target", "NOB Achieved"}.issubset(grp.columns):
        grp["NOB Achievement %"] = grp.apply(
            lambda r: (r["NOB Achieved"] / r["NOB Target"] * 100)
            if r["NOB Target"] > 0 else 0,
            axis=1,
        )
    else:
        grp["NOB Achievement %"] = 0

    grp = grp.sort_values(["Month", "BRANCH"])
    return grp


def quick_insights(df_):
    insights = []
    kpis = build_kpis(df_)

    # Sales overall
    diff = kpis["total_sales_ach"] - kpis["total_sales_target"]
    if diff >= 0:
        insights.append(
            f"Overall sales are **ABOVE** target by {format_number(diff)} "
            f"({format_number(kpis['sales_ach_pct'])}%)."
        )
    else:
        insights.append(
            f"Overall sales are **BELOW** target by {format_number(abs(diff))} "
            f"({format_number(kpis['sales_ach_pct'])}%)."
        )

    # NOB overall
    nob_diff = kpis["total_nob_ach"] - kpis["total_nob_target"]
    if nob_diff >= 0:
        insights.append(
            f"NOB is **ABOVE** target by {format_number(nob_diff)} "
            f"({format_number(kpis['nob_ach_pct'])}%)."
        )
    else:
        insights.append(
            f"NOB is **BELOW** target by {format_number(abs(nob_diff))} "
            f"({format_number(kpis['nob_ach_pct'])}%)."
        )

    # Best/Worst branch by achievement %
    if df_["BRANCH"].nunique() > 1 and {
        "Sales Target",
        "Sales Achieved",
    }.issubset(df_.columns):
        branch_perf = df_.groupby("BRANCH", as_index=False).agg(
            {"Sales Target": "sum", "Sales Achieved": "sum"}
        )
        branch_perf["Sales Achievement %"] = branch_perf.apply(
            lambda r: (r["Sales Achieved"] / r["Sales Target"] * 100)
            if r["Sales Target"] > 0 else 0,
            axis=1,
        )
        best = branch_perf.sort_values("Sales Achievement %", ascending=False).iloc[0]
        worst = branch_perf.sort_values("Sales Achievement %", ascending=True).iloc[0]

        insights.append(
            f"Best branch: **{best['BRANCH']}** with "
            f"{format_number(best['Sales Achievement %'])}% achievement."
        )
        insights.append(
            f"Lowest branch: **{worst['BRANCH']}** with "
            f"{format_number(worst['Sales Achievement %'])}% achievement."
        )

    return insights


def show_table(df_, title):
    st.subheader(title)

    df_display = df_.copy()

    # Format DATE if present
    if "DATE" in df_display.columns:
        df_display["DATE"] = df_display["DATE"].dt.strftime("%Y-%m-%d")

    # Identify numeric columns to format
    num_cols_money = [
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
                "MTD Sales Target",
                "MTD Sales Achieved",
            ]
        )
    ]
    num_cols_pct = [c for c in df_display.columns if "Achievement %" in c]

    for c in num_cols_money:
        df_display[c] = df_display[c].apply(lambda x: format_number(x, 2))
    for c in num_cols_pct:
        df_display[c] = df_display[c].apply(lambda x: format_number(x, 2))

    st.dataframe(df_display, use_container_width=True)


# ----------------- SIDEBAR FILTERS -----------------
st.sidebar.header("🔎 Filters")

min_date = df["DATE"].min().date()
max_date = df["DATE"].max().date()

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, (list, tuple)):
    start_date, end_date = date_range
else:
    start_date = end_date = date_range

branches = sorted(df["BRANCH"].dropna().unique())
selected_branches = st.sidebar.multiselect(
    "Branch(es)",
    options=branches,
    default=branches,
)

report_type = st.sidebar.radio(
    "Report Type",
    options=["Daily", "Monthly"],
    index=0,
)

# ----------------- APPLY FILTERS -----------------
mask = (
    (df["DATE"].dt.date >= start_date)
    & (df["DATE"].dt.date <= end_date)
    & (df["BRANCH"].isin(selected_branches))
)

df_filtered = df.loc[mask].copy()

if df_filtered.empty:
    st.warning("No data for selected filters.")
    st.stop()

st.caption(
    f"Showing data from **{start_date}** to **{end_date}** "
    f"for branch(es): {', '.join(selected_branches)}"
)

# ----------------- KPI SECTION -----------------
st.markdown("### 🔑 Key Metrics (Filtered Period)")

kpis = build_kpis(df_filtered)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Sales Target", format_number(kpis["total_sales_target"]))
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
    st.metric("Total NOB Target", format_number(kpis["total_nob_target"], 0))
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

# ----------------- QUICK INSIGHTS -----------------
st.markdown("### 💡 Quick Insights")
for line in quick_insights(df_filtered):
    st.write("• " + line)

st.markdown("---")

# ----------------- REPORTS -----------------
if report_type == "Daily":
    summary_df = daily_summary(df_filtered)
    show_table(summary_df, "🗓 Daily Report (by Date & Branch)")

    if {"Sales Target", "Sales Achieved"}.issubset(summary_df.columns):
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

    if {"Sales Target", "Sales Achieved"}.issubset(summary_df.columns):
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
