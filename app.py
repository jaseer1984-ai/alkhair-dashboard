# app.py — Enhanced Professional Sales Dashboard
# ==============================================
# ✨ Modern UI, KPI cards, gauges, alerts, daily/monthly reports
# 🧪 Run: streamlit run app.py

import pandas as pd
import streamlit as st
import altair as alt
from io import BytesIO
import datetime

# ===================== CONFIG =====================
st.set_page_config(
    page_title="Sales Dashboard Pro",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_URL = "https://docs.google.com/spreadsheets/d/1c_Ihaydv2kwGWg12MYhFmIxpXTuCKUT5/edit?usp=sharing&ouid=116672375799622827695&rtpof=true&sd=true"

# ===================== CUSTOM CSS =====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .insight-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .alert-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
    .kpi-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 0.9rem 1.0rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        border: 1px solid #f0f0f0;
        height: 120px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.75rem;
    }
    .kpi-left {
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .kpi-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #555555;
        margin-bottom: 0.3rem;
    }
    .kpi-main {
        font-size: 1.5rem;
        font-weight: 700;
    }
    .kpi-right {
        text-align: right;
        font-size: 0.8rem;
        color: #777777;
    }
    .kpi-highlight {
        font-weight: 600;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# ===================== TITLE =====================
st.markdown('<h1 class="main-header">📊 Sales Performance Dashboard</h1>', unsafe_allow_html=True)
st.caption("Data source: Google Sheet ➜ published XLSX link")

# ===================== DATA LOADING (from Google Sheet URL) =====================
@st.cache_data
def load_data():
    df = pd.read_excel(DATA_URL)

    if "DATE" not in df.columns:
        raise ValueError("Column 'DATE' not found")

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"])

    if "BRANCH" not in df.columns:
        raise ValueError("Column 'BRANCH' not found")

    df["BRANCH"] = df["BRANCH"].astype(str).str.strip()
    df["Month"] = df["DATE"].dt.to_period("M").astype(str)
    df["YearMonth"] = df["DATE"].dt.strftime("%Y-%m")
    df["DayOfWeek"] = df["DATE"].dt.day_name()

    return df

try:
    with st.spinner("📊 Loading and processing data from Google Sheets..."):
        df = load_data()
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

if df.empty:
    st.error("No valid data found")
    st.stop()

# ===================== HELPER FUNCTIONS =====================
def safe_sum(series):
    return series.fillna(0).sum()

def format_number(x, decimals=0):
    if pd.isna(x):
        return "-"
    return f"{x:,.{decimals}f}"

def format_percent(x):
    if pd.isna(x):
        return "-"
    return f"{x:.1f}%"

def build_kpis(df_):
    cols = df_.columns

    sales_tgt = safe_sum(df_["Sales Target"]) if "Sales Target" in cols else 0
    sales_ach = safe_sum(df_["Sales Achieved"]) if "Sales Achieved" in cols else 0
    nob_tgt = safe_sum(df_["NOB Target"]) if "NOB Target" in cols else 0
    nob_ach = safe_sum(df_["NOB Achieved"]) if "NOB Achieved" in cols else 0

    sales_ach_pct = (sales_ach / sales_tgt * 100) if sales_tgt > 0 else 0
    nob_ach_pct = (nob_ach / nob_tgt * 100) if nob_tgt > 0 else 0
    sales_gap = sales_ach - sales_tgt
    nob_gap = nob_ach - nob_tgt

    overall_pct = (sales_ach_pct + nob_ach_pct) / 2 if (sales_tgt > 0 and nob_tgt > 0) else 0

    return {
        "sales_target": sales_tgt,
        "sales_achieved": sales_ach,
        "sales_ach_pct": sales_ach_pct,
        "sales_gap": sales_gap,
        "nob_target": nob_tgt,
        "nob_achieved": nob_ach,
        "nob_ach_pct": nob_ach_pct,
        "nob_gap": nob_gap,
        "overall_pct": overall_pct,
    }

# arrow html helper
def arrow_html(pct_diff: float) -> str:
    """
    Returns HTML for green up arrow or red down arrow with % vs target.
    pct_diff = achievement% - 100
    """
    if pd.isna(pct_diff):
        return ""
    if pct_diff >= 0:
        return f'<span style="color:#1a9c5b; font-weight:600;">▲ {pct_diff:.1f}% vs target</span>'
    else:
        return f'<span style="color:#e5534b; font-weight:600;">▼ {abs(pct_diff):.1f}% vs target</span>'

# 🔧 Gauge with center text
def create_gauge_chart(value, target):
    """
    Create a donut-style gauge with percentage text in the center.
    """
    pct = (value / target * 100) if target > 0 else 0
    pct_for_gauge = max(0, min(pct, 100))

    if pct >= 100:
        color = "#38ef7d"
    elif pct < 80:
        color = "#f5576c"
    else:
        color = "#ffc107"

    gauge_data = pd.DataFrame([
        {"category": "Achieved", "value": pct_for_gauge},
        {"category": "Remaining", "value": max(0, 100 - pct_for_gauge)},
    ])

    base = alt.Chart(gauge_data).mark_arc(innerRadius=55).encode(
        theta=alt.Theta("value:Q", stack=True),
        color=alt.Color(
            "category:N",
            scale=alt.Scale(
                domain=["Achieved", "Remaining"],
                range=[color, "#e0e0e0"],
            ),
            legend=None,
        ),
        tooltip=["category:N", "value:Q"],
    )

    text_df = pd.DataFrame({"label": [f"{pct:.1f}%"]})
    text = alt.Chart(text_df).mark_text(
        fontSize=18,
        fontWeight="bold"
    ).encode(
        text="label:N",
        x=alt.value(80),
        y=alt.value(80),
    )

    chart = (base + text).properties(
        width=160,
        height=160,
    )

    return chart

# ===================== SIDEBAR FILTERS =====================
st.sidebar.header("🔎 Filters & Settings")

# ---- 2 separate date pickers ----
min_date = df["DATE"].min().date()
max_date = df["DATE"].max().date()

st.sidebar.subheader("📅 Date Range")
start_date = st.sidebar.date_input(
    "From",
    value=min_date,
    min_value=min_date,
    max_value=max_date,
)
end_date = st.sidebar.date_input(
    "To",
    value=max_date,
    min_value=min_date,
    max_value=max_date,
)

if start_date > end_date:
    st.sidebar.error("From date cannot be after To date")
    st.stop()

# ---- Branch radio selection ----
branches_all = sorted(df["BRANCH"].dropna().unique())
st.sidebar.subheader("🏢 Branch")

branch_choice = st.sidebar.radio(
    "Select branch",
    options=["All Branches"] + branches_all,
    index=0,
)

if branch_choice == "All Branches":
    selected_branches = branches_all
else:
    selected_branches = [branch_choice]

show_alerts = st.sidebar.checkbox("🚨 Show Performance Alerts", value=True)
show_trends = st.sidebar.checkbox("📈 Show Trend Analysis", value=True)

if st.sidebar.button("🔄 Reset Filters"):
    st.rerun()

# ===================== APPLY FILTERS =====================
mask = (
    (df["DATE"].dt.date >= start_date) &
    (df["DATE"].dt.date <= end_date) &
    (df["BRANCH"].isin(selected_branches))
)

df_filtered = df.loc[mask].copy()

if df_filtered.empty:
    st.warning("⚠️ No data matches your filters")
    st.stop()

# ===================== KPI SECTION =====================
st.markdown("### 🎯 Performance Overview")

kpis = build_kpis(df_filtered)

sales_delta_pct = kpis["sales_ach_pct"] - 100
nob_delta_pct = kpis["nob_ach_pct"] - 100

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-left">
                <div class="kpi-title">Sales Performance</div>
                <div class="kpi-main">{format_number(kpis['sales_achieved'])}</div>
            </div>
            <div class="kpi-right">
                Target: <span class="kpi-highlight">{format_number(kpis['sales_target'])}</span><br>
                Achievement: <span class="kpi-highlight">{format_percent(kpis['sales_ach_pct'])}</span><br>
                {arrow_html(sales_delta_pct)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-left">
                <div class="kpi-title">NOB Performance</div>
                <div class="kpi-main">{format_number(kpis['nob_achieved'], 0)}</div>
            </div>
            <div class="kpi-right">
                Target: <span class="kpi-highlight">{format_number(kpis['nob_target'], 0)}</span><br>
                Achievement: <span class="kpi-highlight">{format_percent(kpis['nob_ach_pct'])}</span><br>
                {arrow_html(nob_delta_pct)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col3:
    sales_gap_label = "Above Target" if kpis["sales_gap"] >= 0 else "Below Target"
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-left">
                <div class="kpi-title">Sales Gap</div>
                <div class="kpi-main">{format_number(abs(kpis['sales_gap']))}</div>
            </div>
            <div class="kpi-right">
                Status: <span class="kpi-highlight">{sales_gap_label}</span><br>
                Total Sales: <span class="kpi-highlight">{format_number(kpis['sales_achieved'])}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-left">
                <div class="kpi-title">Overall Achievement</div>
                <div class="kpi-main">{format_percent(kpis['overall_pct'])}</div>
            </div>
            <div class="kpi-right">
                Sales %: <span class="kpi-highlight">{format_percent(kpis['sales_ach_pct'])}</span><br>
                NOB %: <span class="kpi-highlight">{format_percent(kpis['nob_ach_pct'])}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ===================== GAUGE CHARTS =====================
st.markdown("### 📊 Achievement Gauges")
gauge_col1, gauge_col2, gauge_col3 = st.columns(3)

with gauge_col1:
    st.markdown("**Sales**")
    gauge1 = create_gauge_chart(kpis["sales_achieved"], kpis["sales_target"])
    st.altair_chart(gauge1, use_container_width=True)

with gauge_col2:
    st.markdown("**NOB**")
    gauge2 = create_gauge_chart(kpis["nob_achieved"], kpis["nob_target"])
    st.altair_chart(gauge2, use_container_width=True)

with gauge_col3:
    st.markdown("**Overall Score**")
    st.markdown(f"### {format_percent(kpis['overall_pct'])}")
    st.progress(min(max(kpis["overall_pct"] / 100, 0), 1.0))

# ===================== PERFORMANCE ALERTS =====================
if show_alerts:
    st.markdown("### 🚨 Performance Alerts")

    alerts = []

    branch_perf = df_filtered.groupby("BRANCH").agg({
        "Sales Target": "sum",
        "Sales Achieved": "sum",
        "NOB Target": "sum",
        "NOB Achieved": "sum"
    }).reset_index()

    branch_perf["Sales Ach %"] = (branch_perf["Sales Achieved"] / branch_perf["Sales Target"] * 100).fillna(0)
    branch_perf["NOB Ach %"] = (branch_perf["NOB Achieved"] / branch_perf["NOB Target"] * 100).fillna(0)

    for _, row in branch_perf.iterrows():
        if row["Sales Ach %"] < 80:
            alerts.append(f"⚠️ **{row['BRANCH']}** - Sales achievement critically low at {row['Sales Ach %']:.1f}%")
        if row["NOB Ach %"] < 80:
            alerts.append(f"⚠️ **{row['BRANCH']}** - NOB achievement critically low at {row['NOB Ach %']:.1f}%")

    if kpis["sales_ach_pct"] < 100:
        alerts.append(f"📊 Overall sales gap: {format_number(abs(kpis['sales_gap']))} below target")

    if alerts:
        for alert in alerts[:5]:
            st.markdown(f'<div class="alert-box">{alert}</div>', unsafe_allow_html=True)
    else:
        st.success("✅ No critical alerts! All metrics are on track.")

# ===================== TOP/BOTTOM PERFORMERS =====================
st.markdown("### 🏆 Branch Performance Rankings")

perf_col1, perf_col2 = st.columns(2)

branch_summary = df_filtered.groupby("BRANCH").agg({
    "Sales Target": "sum",
    "Sales Achieved": "sum",
    "NOB Target": "sum",
    "NOB Achieved": "sum"
}).reset_index()

branch_summary["Sales Achievement %"] = (branch_summary["Sales Achieved"] / branch_summary["Sales Target"] * 100).fillna(0)
branch_summary["NOB Achievement %"] = (branch_summary["NOB Achieved"] / branch_summary["NOB Target"] * 100).fillna(0)
branch_summary = branch_summary.sort_values("Sales Achievement %", ascending=False)

with perf_col1:
    st.markdown("#### 🥇 Top 3 Performers (Sales)")
    top3 = branch_summary.head(3)
    for _, row in top3.iterrows():
        st.markdown(f"""
        <div class="insight-box">
        <strong>{row['BRANCH']}</strong><br>
        Achievement: {row['Sales Achievement %']:.1f}% | 
        Sales: {format_number(row['Sales Achieved'])}
        </div>
        """, unsafe_allow_html=True)

with perf_col2:
    st.markdown("#### 📉 Bottom 3 Performers (Sales)")
    bottom3 = branch_summary.tail(3)
    for _, row in bottom3.iterrows():
        st.markdown(f"""
        <div class="alert-box">
        <strong>{row['BRANCH']}</strong><br>
        Achievement: {row['Sales Achievement %']:.1f}% | 
        Gap: {format_number(row['Sales Target'] - row['Sales Achieved'])}
        </div>
        """, unsafe_allow_html=True)

# ===================== TREND ANALYSIS =====================
if show_trends:
    st.markdown("### 📈 Trend Analysis")

    daily_trend = df_filtered.groupby("DATE").agg({
        "Sales Target": "sum",
        "Sales Achieved": "sum"
    }).reset_index()

    daily_trend["Achievement %"] = (daily_trend["Sales Achieved"] / daily_trend["Sales Target"] * 100).fillna(0)

    trend_chart = alt.Chart(daily_trend).mark_line(point=True).encode(
        x=alt.X("DATE:T", title="Date"),
        y=alt.Y("Achievement %:Q", title="Sales Achievement %"),
        tooltip=["DATE:T", "Achievement %:Q", "Sales Achieved:Q", "Sales Target:Q"]
    ).properties(
        height=300,
        title="Daily Sales Achievement Trend"
    ).interactive()

    target_line = alt.Chart(pd.DataFrame({'y': [100]})).mark_rule(
        color='red',
        strokeDash=[5, 5]
    ).encode(y='y:Q')

    st.altair_chart(trend_chart + target_line, use_container_width=True)

# ===================== DETAILED REPORTS =====================
st.markdown("---")
st.markdown("### 📑 Daily Detailed Report")

daily_data = df_filtered.groupby(["DATE", "BRANCH"]).agg({
    "Sales Target": "sum",
    "Sales Achieved": "sum",
    "NOB Target": "sum",
    "NOB Achieved": "sum"
}).reset_index()

daily_data["Sales Gap"] = daily_data["Sales Achieved"] - daily_data["Sales Target"]
daily_data["Sales Ach %"] = (daily_data["Sales Achieved"] / daily_data["Sales Target"] * 100).fillna(0)
daily_data["NOB Ach %"] = (daily_data["NOB Achieved"] / daily_data["NOB Target"] * 100).fillna(0)

display_df_daily = daily_data.copy()
display_df_daily["DATE"] = display_df_daily["DATE"].dt.strftime("%Y-%m-%d")
display_df_daily["Sales Target"] = display_df_daily["Sales Target"].apply(lambda x: format_number(x))
display_df_daily["Sales Achieved"] = display_df_daily["Sales Achieved"].apply(lambda x: format_number(x))
display_df_daily["Sales Gap"] = display_df_daily["Sales Gap"].apply(lambda x: format_number(x))
display_df_daily["Sales Ach %"] = display_df_daily["Sales Ach %"].apply(lambda x: format_percent(x))
display_df_daily["NOB Target"] = display_df_daily["NOB Target"].apply(lambda x: format_number(x, 0))
display_df_daily["NOB Achieved"] = display_df_daily["NOB Achieved"].apply(lambda x: format_number(x, 0))
display_df_daily["NOB Ach %"] = display_df_daily["NOB Ach %"].apply(lambda x: format_percent(x))

st.dataframe(display_df_daily, use_container_width=True, height=350)

st.markdown("### 📆 Monthly Detailed Report")

monthly_data = df_filtered.groupby(["Month", "BRANCH"]).agg({
    "Sales Target": "sum",
    "Sales Achieved": "sum",
    "NOB Target": "sum",
    "NOB Achieved": "sum"
}).reset_index()

monthly_data["Sales Gap"] = monthly_data["Sales Achieved"] - monthly_data["Sales Target"]
monthly_data["Sales Ach %"] = (monthly_data["Sales Achieved"] / monthly_data["Sales Target"] * 100).fillna(0)
monthly_data["NOB Ach %"] = (monthly_data["NOB Achieved"] / monthly_data["NOB Target"] * 100).fillna(0)

display_df_monthly = monthly_data.copy()
display_df_monthly["Sales Target"] = display_df_monthly["Sales Target"].apply(lambda x: format_number(x))
display_df_monthly["Sales Achieved"] = display_df_monthly["Sales Achieved"].apply(lambda x: format_number(x))
display_df_monthly["Sales Gap"] = display_df_monthly["Sales Gap"].apply(lambda x: format_number(x))
display_df_monthly["Sales Ach %"] = display_df_monthly["Sales Ach %"].apply(lambda x: format_percent(x))
display_df_monthly["NOB Target"] = display_df_monthly["NOB Target"].apply(lambda x: format_number(x, 0))
display_df_monthly["NOB Achieved"] = display_df_monthly["NOB Achieved"].apply(lambda x: format_number(x, 0))
display_df_monthly["NOB Ach %"] = display_df_monthly["NOB Ach %"].apply(lambda x: format_percent(x))

st.dataframe(display_df_monthly, use_container_width=True, height=350)

st.markdown("### 🧭 Branch Comparison (Achievement %)")

branch_comp = df_filtered.groupby("BRANCH").agg({
    "Sales Target": "sum",
    "Sales Achieved": "sum",
    "NOB Target": "sum",
    "NOB Achieved": "sum"
}).reset_index()

branch_comp["Sales Ach %"] = (branch_comp["Sales Achieved"] / branch_comp["Sales Target"] * 100).fillna(0)
branch_comp["NOB Ach %"] = (branch_comp["NOB Achieved"] / branch_comp["NOB Target"] * 100).fillna(0)
branch_comp = branch_comp.sort_values("Sales Ach %", ascending=False)

heatmap_data = branch_comp[["BRANCH", "Sales Ach %", "NOB Ach %"]].melt(
    id_vars=["BRANCH"],
    var_name="Metric",
    value_name="Achievement %"
)

heatmap = alt.Chart(heatmap_data).mark_rect().encode(
    x=alt.X("Metric:N", title=None),
    y=alt.Y("BRANCH:N", title="Branch"),
    color=alt.Color(
        "Achievement %:Q",
        scale=alt.Scale(scheme="redyellowgreen", domain=[0, 150]),
        legend=alt.Legend(title="Achievement %")
    ),
    tooltip=["BRANCH:N", "Metric:N", "Achievement %:Q"]
).properties(
    height=max(300, len(selected_branches) * 40),
    title="Branch Performance Heatmap"
)

st.altair_chart(heatmap, use_container_width=True)

display_df_branch = branch_comp.copy()
display_df_branch["Sales Target"] = display_df_branch["Sales Target"].apply(lambda x: format_number(x))
display_df_branch["Sales Achieved"] = display_df_branch["Sales Achieved"].apply(lambda x: format_number(x))
display_df_branch["Sales Ach %"] = display_df_branch["Sales Ach %"].apply(lambda x: format_percent(x))
display_df_branch["NOB Target"] = display_df_branch["NOB Target"].apply(lambda x: format_number(x, 0))
display_df_branch["NOB Achieved"] = display_df_branch["NOB Achieved"].apply(lambda x: format_number(x, 0))
display_df_branch["NOB Ach %"] = display_df_branch["NOB Ach %"].apply(lambda x: format_percent(x))

st.dataframe(display_df_branch, use_container_width=True, height=350)

# ===================== EXPORT FUNCTIONALITY =====================
st.markdown("---")
st.markdown("### 💾 Export Data")

export_col1, export_col2 = st.columns([3, 1])

with export_col1:
    st.info("📊 Download current view as Excel file")

with export_col2:
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_filtered.to_excel(writer, sheet_name='Raw Data', index=False)
        branch_summary.to_excel(writer, sheet_name='Branch Summary', index=False)

    output.seek(0)

    st.download_button(
        label="📥 Download Excel",
        data=output,
        file_name=f"sales_report_{datetime.date.today()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# ===================== FOOTER =====================
st.markdown("---")
st.caption("📊 Sales Performance Dashboard | Built with Streamlit")

