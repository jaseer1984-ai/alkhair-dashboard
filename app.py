# app.py — Enhanced Professional Sales Dashboard
# ==============================================
# ✨ Premium Features:
#   - Modern UI with custom styling
#   - Advanced visualizations (gauges, trends, heatmaps)
#   - Export functionality
#   - Top/Bottom performers
#   - Performance alerts
#   - Enhanced insights
#
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
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-metric {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .warning-metric {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
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
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ===================== TITLE =====================
st.markdown('<h1 class="main-header">📊 Sales Performance Dashboard</h1>', unsafe_allow_html=True)
# (Subtitle removed as requested)

# ===================== FILE UPLOAD =====================
uploaded_file = st.file_uploader(
    "📁 Upload Excel file (DATE, BRANCH, Sales Target, Sales Achieved, NOB Target, NOB Achieved)",
    type=["xlsx", "xls"],
    help="Upload your sales data file to generate insights"
)

if not uploaded_file:
    st.info("👆 Upload your Excel file to unlock powerful analytics")
    st.stop()

# ===================== DATA LOADING =====================
@st.cache_data
def load_data(file_obj):
    df = pd.read_excel(file_obj)
    
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
    with st.spinner("📊 Loading and processing data..."):
        df = load_data(uploaded_file)
except Exception as e:
    st.error(f"❌ Error: {e}")
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

def get_trend_icon(current, previous):
    if pd.isna(current) or pd.isna(previous) or previous == 0:
        return "→"
    return "📈" if current > previous else "📉"

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
    
    return {
        "sales_target": sales_tgt,
        "sales_achieved": sales_ach,
        "sales_ach_pct": sales_ach_pct,
        "sales_gap": sales_gap,
        "nob_target": nob_tgt,
        "nob_achieved": nob_ach,
        "nob_ach_pct": nob_ach_pct,
        "nob_gap": nob_gap,
    }

# 🔧 Gauge with center text
def create_gauge_chart(value, target, title):
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
        title=title
    )

    return chart

# ===================== SIDEBAR FILTERS =====================
st.sidebar.header("🔎 Filters & Settings")

min_date = df["DATE"].min().date()
max_date = df["DATE"].max().date()

date_range = st.sidebar.date_input(
    "📅 Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range if not isinstance(date_range, tuple) else date_range[0]

branches = sorted(df["BRANCH"].dropna().unique())
selected_branches = st.sidebar.multiselect(
    "🏢 Branch(es)",
    options=branches,
    default=branches,
)

report_type = st.sidebar.radio(
    "📊 Report Type",
    options=["Daily", "Monthly", "Branch Comparison"],
    index=0,
)

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

# (Caption with "4 records | ..." removed as requested)

# ===================== KPI SECTION =====================
st.markdown("### 🎯 Performance Overview")

kpis = build_kpis(df_filtered)

col1, col2, col3, col4 = st.columns(4)

with col1:
    delta_color = "normal" if kpis["sales_gap"] >= 0 else "inverse"
    st.metric(
        "Sales Achieved",
        format_number(kpis["sales_achieved"]),
        delta=format_number(kpis["sales_gap"]),
        delta_color=delta_color
    )
    st.progress(min(max(kpis["sales_ach_pct"] / 100, 0), 1.0))
    st.caption(f"Target: {format_number(kpis['sales_target'])}")

with col2:
    st.metric(
        "Sales Achievement",
        format_percent(kpis["sales_ach_pct"]),
    )
    status = "✅ On Track" if kpis["sales_ach_pct"] >= 100 else "⚠️ Below Target" if kpis["sales_ach_pct"] >= 80 else "🔴 Critical"
    st.markdown(f"**{status}**")

with col3:
    delta_color = "normal" if kpis["nob_gap"] >= 0 else "inverse"
    st.metric(
        "NOB Achieved",
        format_number(kpis["nob_achieved"], 0),
        delta=format_number(kpis["nob_gap"], 0),
        delta_color=delta_color
    )
    st.progress(min(max(kpis["nob_ach_pct"] / 100, 0), 1.0))
    st.caption(f"Target: {format_number(kpis['nob_target'], 0)}")

with col4:
    st.metric(
        "NOB Achievement",
        format_percent(kpis["nob_ach_pct"]),
    )
    status = "✅ On Track" if kpis["nob_ach_pct"] >= 100 else "⚠️ Below Target" if kpis["nob_ach_pct"] >= 80 else "🔴 Critical"
    st.markdown(f"**{status}**")

# ===================== GAUGE CHARTS =====================
st.markdown("### 📊 Achievement Gauges")
gauge_col1, gauge_col2, gauge_col3 = st.columns(3)

with gauge_col1:
    gauge1 = create_gauge_chart(kpis["sales_achieved"], kpis["sales_target"], "Sales")
    st.altair_chart(gauge1, use_container_width=True)

with gauge_col2:
    gauge2 = create_gauge_chart(kpis["nob_achieved"], kpis["nob_target"], "NOB")
    st.altair_chart(gauge2, use_container_width=True)

with gauge_col3:
    overall_pct = (kpis["sales_ach_pct"] + kpis["nob_ach_pct"]) / 2
    st.metric("Overall Score", format_percent(overall_pct))
    st.progress(min(max(overall_pct / 100, 0), 1.0))
    
    if overall_pct >= 100:
        st.success("🎉 Excellent Performance!")
    elif overall_pct >= 90:
        st.info("👍 Good Performance")
    elif overall_pct >= 80:
        st.warning("⚠️ Needs Improvement")
    else:
        st.error("🔴 Critical Attention Required")

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
st.markdown(f"### 📑 Detailed {report_type} Report")

if report_type == "Daily":
    daily_data = df_filtered.groupby(["DATE", "BRANCH"]).agg({
        "Sales Target": "sum",
        "Sales Achieved": "sum",
        "NOB Target": "sum",
        "NOB Achieved": "sum"
    }).reset_index()
    
    daily_data["Sales Gap"] = daily_data["Sales Achieved"] - daily_data["Sales Target"]
    daily_data["Sales Ach %"] = (daily_data["Sales Achieved"] / daily_data["Sales Target"] * 100).fillna(0)
    daily_data["NOB Ach %"] = (daily_data["NOB Achieved"] / daily_data["NOB Target"] * 100).fillna(0)
    
    display_df = daily_data.copy()
    display_df["DATE"] = display_df["DATE"].dt.strftime("%Y-%m-%d")
    display_df["Sales Target"] = display_df["Sales Target"].apply(lambda x: format_number(x))
    display_df["Sales Achieved"] = display_df["Sales Achieved"].apply(lambda x: format_number(x))
    display_df["Sales Gap"] = display_df["Sales Gap"].apply(lambda x: format_number(x))
    display_df["Sales Ach %"] = display_df["Sales Ach %"].apply(lambda x: format_percent(x))
    display_df["NOB Target"] = display_df["NOB Target"].apply(lambda x: format_number(x, 0))
    display_df["NOB Achieved"] = display_df["NOB Achieved"].apply(lambda x: format_number(x, 0))
    display_df["NOB Ach %"] = display_df["NOB Ach %"].apply(lambda x: format_percent(x))
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
elif report_type == "Monthly":
    monthly_data = df_filtered.groupby(["Month", "BRANCH"]).agg({
        "Sales Target": "sum",
        "Sales Achieved": "sum",
        "NOB Target": "sum",
        "NOB Achieved": "sum"
    }).reset_index()
    
    monthly_data["Sales Gap"] = monthly_data["Sales Achieved"] - monthly_data["Sales Target"]
    monthly_data["Sales Ach %"] = (monthly_data["Sales Achieved"] / monthly_data["Sales Target"] * 100).fillna(0)
    monthly_data["NOB Ach %"] = (monthly_data["NOB Achieved"] / monthly_data["NOB Target"] * 100).fillna(0)
    
    display_df = monthly_data.copy()
    display_df["Sales Target"] = display_df["Sales Target"].apply(lambda x: format_number(x))
    display_df["Sales Achieved"] = display_df["Sales Achieved"].apply(lambda x: format_number(x))
    display_df["Sales Gap"] = display_df["Sales Gap"].apply(lambda x: format_number(x))
    display_df["Sales Ach %"] = display_df["Sales Ach %"].apply(lambda x: format_percent(x))
    display_df["NOB Target"] = display_df["NOB Target"].apply(lambda x: format_number(x, 0))
    display_df["NOB Achieved"] = display_df["NOB Achieved"].apply(lambda x: format_number(x, 0))
    display_df["NOB Ach %"] = display_df["NOB Ach %"].apply(lambda x: format_percent(x))
    
    st.dataframe(display_df, use_container_width=True, height=400)

else:  # Branch Comparison
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
    
    display_df = branch_comp.copy()
    display_df["Sales Target"] = display_df["Sales Target"].apply(lambda x: format_number(x))
    display_df["Sales Achieved"] = display_df["Sales Achieved"].apply(lambda x: format_number(x))
    display_df["Sales Ach %"] = display_df["Sales Ach %"].apply(lambda x: format_percent(x))
    display_df["NOB Target"] = display_df["NOB Target"].apply(lambda x: format_number(x, 0))
    display_df["NOB Achieved"] = display_df["NOB Achieved"].apply(lambda x: format_number(x, 0))
    display_df["NOB Ach %"] = display_df["NOB Ach %"].apply(lambda x: format_percent(x))
    
    st.dataframe(display_df, use_container_width=True)

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
