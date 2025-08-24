import io
import re
from datetime import datetime, timedelta
from typing import Dict, Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================
# CONFIGURATION
# =========================
SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQG7boLWl2bNLCPR05NXv6EFpPPcFfXsiXPQ7rAGYr3q8Nkc2Ijg8BqEwVofcMLSg/pub?output=xlsx"

st.set_page_config(page_title="Al Khair Business Performance", layout="wide", initial_sidebar_state="collapsed")

# =========================
# STYLES (hide alerts + top KPI cards)
# =========================
CARD_CSS = """
<style>
  .kpi-grid {display:grid; grid-template-columns:repeat(4, 1fr); gap:16px; margin: 10px 0 18px 0;}
  .kpi-card {background:#fff;border:1px solid rgba(0,0,0,.06);border-radius:16px;padding:16px 18px;}
  .kpi-title {font-weight:800;color:#334155;font-size:.95rem;margin-bottom:6px}
  .kpi-value {font-weight:900;font-size:1.8rem;line-height:1;margin-bottom:8px}
  .pill {display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:999px;font-weight:700;font-size:.80rem}
  .pill.ok {background:rgba(16,185,129,.15); color:#059669;}
  .pill.warn {background:rgba(245,158,11,.15); color:#b45309;}
  .pill.bad {background:rgba(239,68,68,.15); color:#b91c1c;}
  .bar {height:10px;background:#eef2f7;border-radius:999px;overflow:hidden;margin-top:10px}
  .bar>div {height:100%;border-radius:999px}
  /* Hide any Streamlit alert banners (success/info/warn) */
  .stAlert {display:none !important}
</style>
"""

st.markdown(CARD_CSS, unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
@st.cache_data(ttl=300)
def load_data(url: str) -> Dict[str, pd.DataFrame]:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    xls = pd.ExcelFile(io.BytesIO(r.content))
    data: Dict[str, pd.DataFrame] = {}
    for sn in xls.sheet_names:
        df = xls.parse(sn)
        df = df.dropna(how='all').dropna(axis=1, how='all')
        if not df.empty:
            data[sn] = df
    return data

# =========================
# HELPERS
# =========================

def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    k: Dict[str, Any] = {}
    sales_a = float(df.get('SalesActual', pd.Series(dtype=float)).sum())
    sales_t = float(df.get('SalesTarget', pd.Series(dtype=float)).sum())
    k['sales_actual'] = sales_a
    k['sales_target'] = sales_t
    k['sales_pct'] = (sales_a / sales_t * 100) if sales_t > 0 else 0.0

    nob_a = float(df.get('NOBActual', pd.Series(dtype=float)).sum()) if 'NOBActual' in df else 0.0
    nob_t = float(df.get('NOBTarget', pd.Series(dtype=float)).sum()) if 'NOBTarget' in df else 0.0
    k['nob_actual'] = nob_a
    k['nob_target'] = nob_t
    k['nob_pct'] = (nob_a / nob_t * 100) if nob_t > 0 else 0.0

    abv_a = float(df.get('ABVActual', pd.Series(dtype=float)).mean()) if 'ABVActual' in df else 0.0
    abv_t = float(df.get('ABVTarget', pd.Series(dtype=float)).mean()) if 'ABVTarget' in df else 0.0
    k['abv_actual'] = abv_a
    k['abv_target'] = abv_t
    k['abv_pct'] = (abv_a / abv_t * 100) if abv_t > 0 else 0.0

    comps = [p for p in [k['sales_pct'], k['nob_pct'], k['abv_pct']] if p > 0]
    k['overall_score'] = float(np.mean(comps)) if comps else 0.0
    return k


def pill_class(pct: float, target: float) -> str:
    if pct >= target: return 'ok'
    if pct >= target * 0.9: return 'warn'
    return 'bad'


def render_top_cards(k: Dict[str, Any]):
    # targets used for color thresholds
    tgt_sales, tgt_nob, tgt_abv = 95.0, 90.0, 90.0
    st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)

    # Sales
    st.markdown(
        f"""
        <div class='kpi-card'>
          <div class='kpi-title'>Sales</div>
          <div class='kpi-value'>SAR {k['sales_actual']:,.0f}</div>
          <span class='pill {pill_class(k['sales_pct'], tgt_sales)}'>↑ {k['sales_pct']:.1f}% of target</span>
          <div class='bar'><div style='width:{min(k['sales_pct'],100):.1f}%;background:linear-gradient(90deg,#60a5fa,#22d3ee)'></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # NOB
    st.markdown(
        f"""
        <div class='kpi-card'>
          <div class='kpi-title'>Baskets (NOB)</div>
          <div class='kpi-value'>{k['nob_actual']:,.0f}</div>
          <span class='pill {pill_class(k['nob_pct'], tgt_nob)}'>↑ {k['nob_pct']:.1f}% of target</span>
          <div class='bar'><div style='width:{min(k['nob_pct'],100):.1f}%;background:linear-gradient(90deg,#34d399,#10b981)'></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ABV
    st.markdown(
        f"""
        <div class='kpi-card'>
          <div class='kpi-title'>Avg Basket Value</div>
          <div class='kpi-value'>SAR {k['abv_actual']:,.2f}</div>
          <span class='pill {pill_class(k['abv_pct'], tgt_abv)}'>↑ {k['abv_pct']:.1f}% of target</span>
          <div class='bar'><div style='width:{min(k['abv_pct'],100):.1f}%;background:linear-gradient(90deg,#f59e0b,#fbbf24)'></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Overall score
    st.markdown(
        f"""
        <div class='kpi-card'>
          <div class='kpi-title'>Overall Score</div>
          <div class='kpi-value'>{k['overall_score']:.1f}%</div>
          <span class='pill ok'>↑ Average Performance</span>
          <div class='bar'><div style='width:{min(k['overall_score'],100):.1f}%;background:linear-gradient(90deg,#818cf8,#60a5fa)'></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# CHARTS
# =========================

def branch_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty: return go.Figure()
    g = df.groupby('BranchName').agg({'SalesActual':'sum','SalesTarget':'sum'}).reset_index()
    g['Achievement'] = np.where(g['SalesTarget']>0, g['SalesActual']/g['SalesTarget']*100, 0)
    fig = go.Figure(go.Bar(x=g['BranchName'], y=g['Achievement'], text=g['Achievement'].round(1).astype(str)+"%", textposition='outside'))
    fig.update_layout(title="Branch Sales Achievement %", yaxis_title="%", xaxis_title="Branch")
    return fig


def daily_trends_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty or 'Date' not in df or df['Date'].isna().all():
        return go.Figure()
    daily = df.groupby(['Date','BranchName']).agg({'SalesActual':'sum'}).reset_index()
    fig = go.Figure()
    for br in daily['BranchName'].unique():
        d = daily[daily['BranchName']==br]
        fig.add_trace(go.Scatter(x=d['Date'], y=d['SalesActual'], mode='lines+markers', name=br))
    fig.update_layout(title='Daily Sales by Branch', xaxis_title='Date', yaxis_title='Sales')
    return fig


def competition_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    g = df.groupby('BranchName').agg({'SalesActual':'sum','SalesTarget':'sum'}).reset_index()
    g['Achievement'] = np.where(g['SalesTarget']>0, g['SalesActual']/g['SalesTarget']*100, 0)
    return g.sort_values('Achievement', ascending=False)

# =========================
# MAIN
# =========================

def main():
    st.title("🏪 Al Khair Business Performance")

    # Hidden source (URL not shown). Refresh only.
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()

    with st.spinner("Loading data from Google Sheets…"):
        data = load_data(SHEET_URL)

    if not data:
        st.error("No data loaded from Google Sheets.")
        return

    # Combine all sheets
    frames = []
    for sn, df in data.items():
        d = df.copy()
        d['BranchName'] = sn
        if 'Date' in d.columns:
            d['Date'] = pd.to_datetime(d['Date'], errors='coerce')
        # Ensure numeric types if present
        for c in ['SalesActual','SalesTarget','NOBActual','NOBTarget','ABVActual','ABVTarget']:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors='coerce')
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    # --- Branch filter (before KPIs so cards react to selection) ---
    branches = df['BranchName'].dropna().unique().tolist()
    selected = st.multiselect("Select Branches", options=branches, default=branches)
    df = df[df['BranchName'].isin(selected)]

    # --- Top KPI cards ---
    k = compute_kpis(df)
    render_top_cards(k)

    # Tabs
    t1, t2, t3 = st.tabs(["📊 Overview", "📈 Daily Trends", "🏆 Competition"])
    with t1:
        st.plotly_chart(branch_chart(df), use_container_width=True, key="overview_chart")
    with t2:
        st.plotly_chart(daily_trends_chart(df), use_container_width=True, key="daily_chart")
    with t3:
        comp = competition_summary(df)
        if comp.empty:
            st.info("No competition data available")
        else:
            st.dataframe(
                comp[['BranchName','SalesActual','SalesTarget','Achievement']]
                    .assign(Achievement=lambda x: x['Achievement'].round(1).astype(str)+'%'),
                use_container_width=True
            )

if __name__ == "__main__":
    main()
