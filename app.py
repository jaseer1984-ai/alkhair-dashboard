"""
Alkhair Family Market — Multi-Branch Executive Dashboard (Clean Fixed)
- Upload Excel (multiple sheets = branches) OR paste a Google Sheets *published* URL
- Beautiful KPIs, Branch Comparison, Radar, Daily Trends, Export
- Plotly charts, Inter font styling, glassmorphism UI
- GitHub/Streamlit Cloud ready

Author: ChatGPT (for Jaseer)
"""
from __future__ import annotations

import io
import re
import math
from datetime import datetime, date, timedelta
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# =========================
# CONFIGURATION
# =========================
class Config:
    PAGE_TITLE = "Alkhair Family Market — Branch Analytics Dashboard"
    LAYOUT = "wide"
    BRANCHES = {
        "Al khair - 102": {"name": "Al Khair", "code": "102", "color": "#3b82f6"},
        "Noora - 104": {"name": "Noora", "code": "104", "color": "#10b981"},
        "Hamra - 109": {"name": "Hamra", "code": "109", "color": "#f59e0b"},
        "Magnus - 107": {"name": "Magnus", "code": "107", "color": "#8b5cf6"},
    }
    TARGETS = {
        "sales_achievement": 95.0,
        "nob_achievement": 90.0,
        "abv_achievement": 90.0,
    }

config = Config()

st.set_page_config(page_title=config.PAGE_TITLE, layout=config.LAYOUT, initial_sidebar_state="collapsed")

# =========================
# STYLES (CSS)
# =========================

def apply_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
        html, body, [data-testid="stAppViewContainer"] { font-family: 'Inter', sans-serif; }
        .main .block-container { padding: 2rem; max-width: 1500px; }

        /* Hero */
        .hero { background: rgba(255,255,255,0.15); border: 1px solid rgba(255,255,255,0.2);
                backdrop-filter: blur(20px); border-radius: 24px; padding: 28px; color: #fff; }

        /* Metric cards */
        [data-testid="metric-container"] { background: rgba(255,255,255,0.95) !important; border-radius: 16px !important;
            border: 1px solid rgba(0,0,0,0.05) !important; padding: 18px !important; }
        [data-testid="metric-container"] [data-testid="metric-value"] { font-weight: 900 !important; }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"]{ gap:6px; background: #fff; border-radius: 14px; padding:8px; border:1px solid rgba(0,0,0,0.05)}
        .stTabs [data-baseweb="tab"]{ border-radius:12px; }

        /* Chart container */
        .stPlotlyChart > div { background: rgba(255,255,255,0.98); border:1px solid rgba(0,0,0,0.05); border-radius: 16px; padding: 12px; }

        /* Cards */
        .branch-card { background: rgba(255,255,255,0.98); border:1px solid rgba(0,0,0,0.06); border-radius:16px; padding:18px; margin:12px 0; }
        .branch-header { font-weight: 800; font-size: 1.1rem; display:flex; align-items:center; gap:10px; }
        .badge { padding: 4px 10px; border-radius: 999px; font-size: .8rem; font-weight:700; }
        .excellent { background: rgba(16,185,129,.15); color:#059669; }
        .good { background: rgba(59,130,246,.15); color:#2563eb; }
        .warn { background: rgba(245,158,11,.15); color:#d97706; }
        .danger { background: rgba(239,68,68,.15); color:#dc2626; }
        .progress { height:10px; background:#f1f5f9; border-radius: 999px; overflow:hidden; }
        .progress > div { height:100%; transition: width .6s ease; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# =========================
# HELPERS
# =========================

def _human(n: float | int | None) -> str:
    if n is None or (isinstance(n, float) and (math.isnan(n) or math.isinf(n))):
        return "–"
    return f"{float(n):,.2f}"


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# =========================
# DATA LOADING
# =========================

def _gs_published_to_xlsx(url: str) -> str:
    # converts .../pubhtml to .../pub?output=xlsx
    if "docs.google.com/spreadsheets/d/e/" in url:
        return re.sub(r"/pubhtml(.*)$", "/pub?output=xlsx", url.strip())
    return url

@st.cache_data(show_spinner=False)
def load_workbook_from_gsheet(published_url: str) -> Dict[str, pd.DataFrame]:
    dl = _gs_published_to_xlsx(published_url)
    r = requests.get(dl, timeout=60)
    r.raise_for_status()
    xls = pd.ExcelFile(io.BytesIO(r.content))
    sheets: Dict[str, pd.DataFrame] = {}
    for sn in xls.sheet_names:
        df = xls.parse(sn)
        df = df.dropna(how='all').dropna(axis=1, how='all')
        if not df.empty:
            sheets[sn] = df
    return sheets

@st.cache_data(show_spinner=False)
def load_workbook_from_upload(file) -> Dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(file)
    sheets: Dict[str, pd.DataFrame] = {}
    for sn in xls.sheet_names:
        df = xls.parse(sn)
        df = df.dropna(how='all').dropna(axis=1, how='all')
        if not df.empty:
            sheets[sn] = df
    return sheets

# =========================
# PROCESSING & ANALYTICS
# =========================

def process_branch_data(excel_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    combined = []
    column_mapping = {
        "Date": "Date",
        "Day": "Day",
        "Sales Target": "SalesTarget",
        "Sales Achivement": "SalesActual",
        "Sales Achievement": "SalesActual",
        "Sales %": "SalesPercent",
        "NOB Target": "NOBTarget",
        "NOB Achievemnet": "NOBActual",
        "NOB Achievement": "NOBActual",
        "NOB %": "NOBPercent",
        " NOB %": "NOBPercent",
        "ABV Target": "ABVTarget",
        "ABV Achievement": "ABVActual",
        " ABV Achievement": "ABVActual",
        "ABV %": "ABVPercent",
    }
    for sheet_name, df in excel_data.items():
        if df.empty:
            continue
        d = df.copy()
        d.columns = d.columns.astype(str).str.strip()
        for old, new in column_mapping.items():
            if old in d.columns:
                d.rename(columns={old: new}, inplace=True)
        d['Branch'] = sheet_name
        d['BranchName'] = config.BRANCHES.get(sheet_name, {}).get('name', sheet_name)
        d['BranchCode'] = config.BRANCHES.get(sheet_name, {}).get('code', '000')
        if 'Date' in d.columns:
            d['Date'] = pd.to_datetime(d['Date'], errors='coerce')
        for col in ['SalesTarget','SalesActual','SalesPercent','NOBTarget','NOBActual','NOBPercent','ABVTarget','ABVActual','ABVPercent']:
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors='coerce')
        combined.append(d)
    return pd.concat(combined, ignore_index=True) if combined else pd.DataFrame()


def calc_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {}
    k = {}
    k['total_sales_target'] = float(df.get('SalesTarget', pd.Series(dtype=float)).sum())
    k['total_sales_actual'] = float(df.get('SalesActual', pd.Series(dtype=float)).sum())
    k['total_sales_variance'] = k['total_sales_actual'] - k['total_sales_target']
    k['overall_sales_percent'] = (k['total_sales_actual'] / k['total_sales_target'] * 100) if k['total_sales_target']>0 else 0
    k['total_nob_target'] = float(df.get('NOBTarget', pd.Series(dtype=float)).sum())
    k['total_nob_actual'] = float(df.get('NOBActual', pd.Series(dtype=float)).sum())
    k['overall_nob_percent'] = (k['total_nob_actual'] / k['total_nob_target'] * 100) if k['total_nob_target']>0 else 0
    k['avg_abv_target'] = float(df.get('ABVTarget', pd.Series(dtype=float)).mean()) if 'ABVTarget' in df else 0
    k['avg_abv_actual'] = float(df.get('ABVActual', pd.Series(dtype=float)).mean()) if 'ABVActual' in df else 0
    k['overall_abv_percent'] = (k['avg_abv_actual'] / k['avg_abv_target'] * 100) if k['avg_abv_target']>0 else 0
    if 'BranchName' in df.columns:
        k['branch_performance'] = df.groupby('BranchName').agg({
            'SalesTarget':'sum','SalesActual':'sum','SalesPercent':'mean',
            'NOBTarget':'sum','NOBActual':'sum','NOBPercent':'mean',
            'ABVTarget':'mean','ABVActual':'mean','ABVPercent':'mean'
        }).round(2)
    if 'Date' in df.columns and df['Date'].notna().any():
        k['date_range'] = {
            'start': df['Date'].min(),
            'end': df['Date'].max(),
            'days': int(df['Date'].dt.date.nunique()),
        }
    # Weighted performance score
    score = 0; w=0
    if k.get('overall_sales_percent',0)>0: score += min(k['overall_sales_percent']/100,1.2)*40; w+=40
    if k.get('overall_nob_percent',0)>0: score += min(k['overall_nob_percent']/100,1.2)*35; w+=35
    if k.get('overall_abv_percent',0)>0: score += min(k['overall_abv_percent']/100,1.2)*25; w+=25
    k['performance_score'] = (score/w*100) if w>0 else 0
    return k

# =========================
# VISUALS
# =========================

def chart_branch_comparison(branch_data: pd.DataFrame) -> go.Figure:
    if branch_data.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=branch_data.index,
        y=branch_data['SalesPercent'],
        name='Sales Achievement %',
        marker=dict(color=branch_data['SalesPercent'], colorscale='RdYlGn', cmin=0, cmax=120, showscale=True,
                    colorbar=dict(title="Achievement %", x=1.02)),
        text=[f"{v:.1f}%" for v in branch_data['SalesPercent']], textposition='outside',
        customdata=branch_data['SalesTarget'],
        hovertemplate='<b>%{x}</b><br>Achievement: %{y:.1f}%<br>Target: SAR %{customdata:,.0f}<extra></extra>'
    ))
    fig.add_hline(y=100, line_dash='dash', line_color='#ef4444', line_width=3, annotation_text='💯 Target', annotation_position='top right')
    fig.add_hline(y=95, line_dash='dot', line_color='#10b981', line_width=2, annotation_text='🌟 Excellence', annotation_position='bottom right')
    fig.update_layout(title='Branch Sales Performance Comparison', xaxis_title='Branch', yaxis_title='Achievement (%)',
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=460, showlegend=False)
    return fig


def chart_radar(branch_data: pd.DataFrame) -> go.Figure:
    if branch_data.empty:
        return go.Figure()
    fig = go.Figure()
    metrics = ['SalesPercent','NOBPercent','ABVPercent']
    labels = ['Sales Achievement','NOB Achievement','ABV Achievement']
    colors = ['#3b82f6','#10b981','#f59e0b','#8b5cf6','#ef4444']
    for i, branch in enumerate(branch_data.index):
        vals = [float(branch_data.loc[branch, m]) if m in branch_data.columns else 0 for m in metrics]
        vals.append(vals[0])
        base = colors[i % len(colors)]
        fig.add_trace(go.Scatterpolar(r=vals, theta=labels+[labels[0]], name=branch, fill='toself',
                                      line=dict(color=base, width=3), fillcolor=_hex_to_rgba(base, 0.12)))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,120], dtick=20)),
                      title='Branch Performance Radar', showlegend=True, height=480,
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    return fig


def chart_daily_trends(df: pd.DataFrame) -> go.Figure:
    if df.empty or 'Date' not in df or df['Date'].isna().all():
        return go.Figure()
    daily = df.groupby(['Date','BranchName']).agg({'SalesActual':'sum','NOBActual':'sum','ABVActual':'mean'}).reset_index()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=('Daily Sales','Number of Baskets','Average Basket Value'), vertical_spacing=0.08)
    color_cycle = ['#3b82f6','#10b981','#f59e0b','#8b5cf6','#ef4444']
    for i, br in enumerate(daily['BranchName'].unique()):
        d = daily[daily['BranchName']==br]
        col = color_cycle[i % len(color_cycle)]
        fig.add_trace(go.Scatter(x=d['Date'], y=d['SalesActual'], name=f"{br} — Sales", line=dict(color=col, width=3), mode='lines+markers'), row=1, col=1)
        fig.add_trace(go.Scatter(x=d['Date'], y=d['NOBActual'], name=f"{br} — NOB", line=dict(color=col, width=3), mode='lines+markers', showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=d['Date'], y=d['ABVActual'], name=f"{br} — ABV", line=dict(color=col, width=3), mode='lines+markers', showlegend=False), row=3, col=1)
    fig.update_layout(height=800, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation='h', y=1.05, x=0.5, xanchor='center'))
    return fig

# =========================
# RENDER SECTIONS
# =========================

def render_overview(df: pd.DataFrame, k: Dict[str, Any]):
    st.markdown("### 🏆 Overall Performance")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("💰 Total Sales", f"SAR {k.get('total_sales_actual',0):,.0f}", delta=f"vs Target: {k.get('overall_sales_percent',0):.1f}%")
    variance_color = "normal" if k.get('total_sales_variance',0) >= 0 else "inverse"
    c2.metric("📊 Sales Variance", f"SAR {k.get('total_sales_variance',0):,.0f}", delta=f"{k.get('overall_sales_percent',0)-100:+.1f}%", delta_color=variance_color)
    nob_color = "normal" if k.get('overall_nob_percent',0) >= 90 else "inverse"
    c3.metric("🛍️ Total Baskets", f"{k.get('total_nob_actual',0):,.0f}", delta=f"Achievement: {k.get('overall_nob_percent',0):.1f}%", delta_color=nob_color)
    abv_color = "normal" if k.get('overall_abv_percent',0) >= 90 else "inverse"
    c4.metric("💎 Avg Basket Value", f"SAR {k.get('avg_abv_actual',0):,.2f}", delta=f"vs Target: {k.get('overall_abv_percent',0):.1f}%", delta_color=abv_color)
    score_color = "normal" if k.get('performance_score',0) >= 80 else "off"
    c5.metric("⭐ Performance Score", f"{k.get('performance_score',0):.0f}/100", delta="Weighted Score", delta_color=score_color)

    st.markdown("### 🏪 Branch Performance Dashboard")
    if 'branch_performance' in k:
        bp = k['branch_performance']
        for br, row in bp.iterrows():
            info = None
            for key, meta in config.BRANCHES.items():
                if meta['name'] == br:
                    info = meta; break
            if info is None:
                info = {"color":"#6b7280","code":"000"}
            avg_pct = np.nanmean([row.get('SalesPercent',0), row.get('NOBPercent',0), row.get('ABVPercent',0)])
            if avg_pct >= 95: badge, cls = "🏆 EXCELLENT", "excellent"
            elif avg_pct >= 85: badge, cls = "✅ GOOD", "good"
            elif avg_pct >= 75: badge, cls = "⚠️ NEEDS FOCUS", "warn"
            else: badge, cls = "🔴 ACTION REQUIRED", "danger"
            st.markdown(f"""
            <div class='branch-card'>
              <div class='branch-header'>
                <span style='color:{info['color']}'>🏪</span> {br} (Code: {info.get('code','')})
                <span class='badge {cls}' style='margin-left:8px'>{badge}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Sales", f"SAR {row.get('SalesActual',0):,.0f}", delta=f"{row.get('SalesPercent',0):.1f}% of target")
            c2.metric("Baskets (NOB)", f"{row.get('NOBActual',0):,.0f}", delta=f"{row.get('NOBPercent',0):.1f}% of target")
            c3.metric("Avg Basket Value", f"SAR {row.get('ABVActual',0):,.2f}", delta=f"{row.get('ABVPercent',0):.1f}% of target")
            c4.metric("Overall Score", f"{avg_pct:.1f}%", delta="Average Performance")
            # Progress bars
            for label, pct, grad_to in [("Sales", row.get('SalesPercent',0), '#22d3ee'), ("Baskets (NOB)", row.get('NOBPercent',0), '#10b981'), ("Basket Value (ABV)", row.get('ABVPercent',0), '#f59e0b')]:
                pct_c = min(float(pct if pd.notna(pct) else 0), 120.0)
                st.markdown(f"""
                <div style='margin:10px 0;'>
                  <div style='display:flex;justify-content:space-between;margin-bottom:6px;'>
                    <span style='font-weight:600;color:#64748b'>{label}</span>
                    <span style='font-weight:800;color:{info['color']}'>{pct_c:.1f}%</span>
                  </div>
                  <div class='progress'><div style='width:{min(pct_c,100)}%;background:linear-gradient(90deg,{info['color']},{grad_to});border-radius:999px'></div></div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("### 📊 Comparisons")
    c1, c2 = st.columns([2,1])
    if 'branch_performance' in k:
        with c1:
            st.plotly_chart(chart_branch_comparison(k['branch_performance']), use_container_width=True, config={"displayModeBar": False})
        with c2:
            st.plotly_chart(chart_radar(k['branch_performance']), use_container_width=True, config={"displayModeBar": False})


def render_daily(df: pd.DataFrame):
    st.markdown("## 📈 Daily Performance Trends")
    if df.empty:
        st.info("No data available for trend analysis.")
        return
    if 'Date' in df and df['Date'].notna().any():
        opts = ["Last 7 Days","Last 30 Days","Last 3 Months","All Time"]
        choice = st.selectbox("Time Period", opts, index=1)
        today = (df['Date'].max() if df['Date'].notna().any() else pd.Timestamp.today()).date()
        if choice == "Last 7 Days":
            start = today - timedelta(days=7)
        elif choice == "Last 30 Days":
            start = today - timedelta(days=30)
        elif choice == "Last 3 Months":
            start = today - timedelta(days=90)
        else:
            start = df['Date'].min().date()
        mask = (df['Date'].dt.date >= start) & (df['Date'].dt.date <= today)
        f = df.loc[mask].copy()
    else:
        f = df.copy()
    st.plotly_chart(chart_daily_trends(f), use_container_width=True, config={"displayModeBar": False})
    if 'Date' in f and f['Date'].notna().any():
        st.markdown("### 📊 Weekly Summary (Sales Achievement %)")
        wk = f.copy()
        wk['Week'] = wk['Date'].dt.isocalendar().week
        wk['Year'] = wk['Date'].dt.year
        g = wk.groupby(['Year','Week','BranchName']).agg({'SalesActual':'sum','SalesTarget':'sum','NOBActual':'sum','ABVActual':'mean'}).reset_index()
        g['SalesAchievement'] = (np.where(g['SalesTarget']>0, g['SalesActual']/g['SalesTarget']*100, 0)).round(1)
        g['WeekLabel'] = g['Year'].astype(str)+"-W"+g['Week'].astype(str)
        pivot = g.pivot(index='WeekLabel', columns='BranchName', values='SalesAchievement').fillna(0)
        display = pivot.applymap(lambda x: f"{x:.1f}%")
        st.dataframe(display, use_container_width=True)


def render_export(df: pd.DataFrame, k: Dict[str, Any]):
    st.markdown("## 📥 Data Export & Analysis")
    if df.empty:
        st.info("No data to export.")
        return
    c1,c2,c3 = st.columns(3)
    with c1:
        st.metric("Total Records", f"{len(df):,}")
        st.metric("Branches", f"{df['BranchName'].nunique() if 'BranchName' in df else 0}")
        st.metric("Dates", f"{df['Date'].dt.date.nunique() if 'Date' in df else 0}")
    with c2:
        st.metric("Total Sales", f"SAR {k.get('total_sales_actual',0):,.0f}")
        st.metric("Target Sales", f"SAR {k.get('total_sales_target',0):,.0f}")
        st.metric("Variance", f"SAR {k.get('total_sales_variance',0):,.0f}")
    with c3:
        # Excel export
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='All Branches', index=False)
            if 'branch_performance' in k:
                k['branch_performance'].to_excel(writer, sheet_name='Branch Summary')
            if 'Date' in df and df['Date'].notna().any():
                daily = df.groupby(['Date','BranchName']).agg({'SalesActual':'sum','SalesTarget':'sum','NOBActual':'sum','ABVActual':'mean'}).reset_index()
                daily.to_excel(writer, sheet_name='Daily Summary', index=False)
        st.download_button("📊 Download Excel Report", buf.getvalue(), f"Alkhair_Branch_Analytics_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        # CSV
        st.download_button("📄 Download CSV", df.to_csv(index=False).encode('utf-8'), f"Alkhair_Branch_Data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv", use_container_width=True)

# =========================
# MAIN
# =========================

def main():
    apply_css()
    st.markdown(
        f"""
        <div class='hero'>
          <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
            <div>
              <h1 style='margin:0'>{config.PAGE_TITLE}</h1>
              <div style='opacity:.8'>Multi-Branch Performance • Real-time Analytics</div>
            </div>
            <div style='text-align:right;opacity:.9'>
              <div style='font-weight:700'>📈 Executive Analytics</div>
              <div>Updated: {datetime.now().strftime('%H:%M:%S')}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Source chooser
    st.sidebar.header("Data Source")
    choice = st.sidebar.radio("Choose input method", ["Google Sheet (published)", "Upload Excel"], index=0)
    sheets_map: Dict[str, pd.DataFrame] = {}
    if choice == "Google Sheet (published)":
        url = st.sidebar.text_input("Published Google Sheets URL", value="")
        if not url:
            st.info("Paste your *Published to web* Google Sheets URL in the sidebar.")
            st.stop()
        try:
            with st.spinner("Downloading workbook from Google Sheets…"):
                sheets_map = load_workbook_from_gsheet(url)
        except Exception as e:
            st.error(f"Failed to download Google Sheet. Ensure it's published to web.\n{e}")
            st.stop()
    else:
        file = st.sidebar.file_uploader("Upload Excel", type=["xlsx","xls"])
        if not file:
            st.info("Upload an Excel file with one sheet per branch.")
            st.stop()
        try:
            with st.spinner("Reading Excel…"):
                sheets_map = load_workbook_from_upload(file)
        except Exception as e:
            st.error(f"Failed to read Excel: {e}")
            st.stop()

    if not sheets_map:
        st.warning("No non-empty sheets found.")
        st.stop()

    with st.spinner("Processing branch data…"):
        df = process_branch_data(sheets_map)

    if df.empty:
        st.error("Could not process data. Check column names and sheet structure.")
        st.stop()

    st.success(f"Loaded data for {df['BranchName'].nunique()} branches • {len(df):,} rows")

    # Optional overall date filter
    if 'Date' in df and df['Date'].notna().any():
        c1,c2,_ = st.columns([2,2,6])
        with c1:
            start_d = st.date_input("Start Date", value=df['Date'].min().date())
        with c2:
            end_d = st.date_input("End Date", value=df['Date'].max().date())
        mask = (df['Date'].dt.date >= start_d) & (df['Date'].dt.date <= end_d)
        df = df.loc[mask].copy()

    k = calc_kpis(df)

    t1,t2,t3,t4 = st.tabs(["🏠 Branch Overview","📈 Daily Trends","🏆 Competition Analysis","📥 Export"])
    with t1:
        render_overview(df, k)
    with t2:
        render_daily(df)
    with t3:
        if 'branch_performance' in k and not k['branch_performance'].empty:
            # Champions cards
            bp = k['branch_performance']
            c1,c2,c3 = st.columns(3)
            with c1:
                br = bp['SalesPercent'].idxmax(); sc = bp.loc[br,'SalesPercent']
                st.markdown(f"<div class='branch-card'><div class='branch-header'>🥇 Sales Champion</div><div style='text-align:center'><h2 style='color:#059669;margin:0'>{br}</h2><h1 style='margin:8px 0'>{sc:.1f}%</h1><div>Target Achievement</div></div></div>", unsafe_allow_html=True)
            with c2:
                br = bp['NOBPercent'].idxmax(); sc = bp.loc[br,'NOBPercent']
                st.markdown(f"<div class='branch-card'><div class='branch-header'>🥇 Basket Champion</div><div style='text-align:center'><h2 style='color:#2563eb;margin:0'>{br}</h2><h1 style='margin:8px 0'>{sc:.1f}%</h1><div>NOB Achievement</div></div></div>", unsafe_allow_html=True)
            with c3:
                br = bp['ABVPercent'].idxmax(); sc = bp.loc[br,'ABVPercent']
                st.markdown(f"<div class='branch-card'><div class='branch-header'>🥇 Value Champion</div><div style='text-align:center'><h2 style='color:#d97706;margin:0'>{br}</h2><h1 style='margin:8px 0'>{sc:.1f}%</h1><div>ABV Achievement</div></div></div>", unsafe_allow_html=True)
            st.markdown("### 📋 Detailed Branch Metrics")
            disp = bp.copy()
            if 'SalesTarget' in disp: disp['SalesTarget'] = disp['SalesTarget'].apply(lambda x: f"SAR {x:,.0f}")
            if 'SalesActual' in disp: disp['SalesActual'] = disp['SalesActual'].apply(lambda x: f"SAR {x:,.0f}")
            for col in ['SalesPercent','NOBPercent','ABVPercent']:
                if col in disp: disp[col] = disp[col].apply(lambda x: f"{x:.1f}%")
            if 'NOBTarget' in disp: disp['NOBTarget'] = disp['NOBTarget'].apply(lambda x: f"{x:,.0f}")
            if 'NOBActual' in disp: disp['NOBActual'] = disp['NOBActual'].apply(lambda x: f"{x:,.0f}")
            if 'ABVTarget' in disp: disp['ABVTarget'] = disp['ABVTarget'].apply(lambda x: f"SAR {x:.2f}")
            if 'ABVActual' in disp: disp['ABVActual'] = disp['ABVActual'].apply(lambda x: f"SAR {x:.2f}")
            disp = disp.rename(columns={'SalesTarget':'Sales Target','SalesActual':'Sales Actual','SalesPercent':'Sales %','NOBTarget':'NOB Target','NOBActual':'NOB Actual','NOBPercent':'NOB %','ABVTarget':'ABV Target','ABVActual':'ABV Actual','ABVPercent':'ABV %'})
            st.dataframe(disp, use_container_width=True)
        else:
            st.info("No branch performance data available.")
    with t4:
        render_export(df, k)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {e}")
        st.stop()
