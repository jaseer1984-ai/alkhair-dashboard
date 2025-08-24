from __future__ import annotations

import io
import re
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

# =========================================
# CONFIG
# =========================================
class Config:
    PAGE_TITLE = "Al Khair Business Performance"
    LAYOUT = "wide"
    AUTO_REFRESH_URL = (
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vQG7boLWl2bNLCPR05NXv6EFpPPcFfXsiXPQ7rAGYr3q8Nkc2Ijg8BqEwVofcMLSg/pubhtml"
    )
    BRANCHES = {
        "Al khair - 102": {"name": "Al Khair", "code": "102", "color": "#6366f1", "icon": "🏢"},
        "Noora - 104": {"name": "Noora", "code": "104", "color": "#10b981", "icon": "🌟"},
        "Hamra - 109": {"name": "Hamra", "code": "109", "color": "#ef4444", "icon": "🏪"},
        "Magnus - 107": {"name": "Magnus", "code": "107", "color": "#8b5cf6", "icon": "⭐"},
    }
    REFRESH_INTERVAL = 300  # seconds

config = Config()
st.set_page_config(page_title=config.PAGE_TITLE, layout=config.LAYOUT, initial_sidebar_state="collapsed")

# =========================================
# AUTO REFRESH
# =========================================
def auto_refresh(seconds: int = 300):
    now = time.time()
    if "last_autorefresh_ts" not in st.session_state:
        st.session_state.last_autorefresh_ts = now
    elif now - st.session_state.last_autorefresh_ts >= seconds:
        st.session_state.last_autorefresh_ts = now
        st.rerun()

# =========================================
# MODERN CSS DESIGN (background fixed to white)
# =========================================
def apply_modern_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

        html, body, [data-testid="stAppViewContainer"] { 
            font-family: 'Poppins', sans-serif !important;
            background: #ffffff !important;   /* White background */
            color: #1a202c;
        }

        .stApp > header { background-color: transparent !important; }
        .main .block-container { padding: 0.5rem !important; max-width: 100% !important; margin: 0 !important; }

        .dashboard-container {
            background: #ffffff;
            border-radius: 25px;
            padding: 2rem;
            margin: 1rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        }

        .modern-header {
            text-align: center;
            padding: 2rem 0;
            background: #6366f1;
            border-radius: 20px;
            color: white;
            margin-bottom: 2rem;
        }
        .modern-header .header-title { font-size: 2.5rem; font-weight: 800; margin-bottom: 0.25rem; }
        .modern-header .header-subtitle { font-size: 1rem; opacity: 0.95; font-weight: 400; }

        .kpi-card {
            background: #ffffff;
            border-radius: 20px; padding: 1.5rem; text-align: center;
            border: 1px solid rgba(0,0,0,0.05);
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }

        .insights-container { background: #f9fafb; border-radius: 20px; padding: 1.25rem; border: 1px solid rgba(0,0,0,0.05); }
        .insight-item { padding: .6rem .8rem; margin: .35rem 0; background: #ffffff; border-radius: 12px; border: 1px solid rgba(0,0,0,0.05); }

        #MainMenu {visibility: hidden;} footer {visibility: hidden;} .stDeployButton {display: none;}
        </style>
        """,
        unsafe_allow_html=True,
    )

# =========================================
# DATA FUNCTIONS
# =========================================
def _pubhtml_to_xlsx(url: str) -> str:
    if "docs.google.com/spreadsheets/d/e/" in url:
        return re.sub(r"/pubhtml(.*)$", "/pub?output=xlsx", url.strip())
    return url

@st.cache_data(ttl=config.REFRESH_INTERVAL, show_spinner=False)
def auto_load_data() -> Dict[str, pd.DataFrame]:
    url = _pubhtml_to_xlsx(config.AUTO_REFRESH_URL)
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        xls = pd.ExcelFile(io.BytesIO(r.content))
        sheets: Dict[str, pd.DataFrame] = {}
        for sn in xls.sheet_names:
            df = xls.parse(sn)
            df = df.dropna(how="all").dropna(axis=1, how="all")
            if not df.empty:
                sheets[sn] = df
        return sheets
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return {}

# =========================================
# MAIN APPLICATION
# =========================================
def main():
    apply_modern_css()
    auto_refresh(config.REFRESH_INTERVAL)

    st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)

    # Refresh button
    if st.button("🔄 Refresh", key="refresh", type="primary"):
        auto_load_data.clear()
        st.rerun()

    # Header
    st.markdown(f"""
        <div class="modern-header">
            <div class="header-title">🏢 Al Khair Business Performance</div>
            <div class="header-subtitle">Last Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
        </div>
    """, unsafe_allow_html=True)

    # Load data
    with st.spinner("🔄 Loading latest data..."):
        sheets_map = auto_load_data()
    if not sheets_map:
        st.error("❌ Unable to load data. Please check your data source.")
        st.stop()

    # Example: show one sheet
    for name, df in sheets_map.items():
        st.subheader(f"📄 {name}")
        st.dataframe(df.head())
        break

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.info("Please refresh the page if the issue persists.")
