# ABEER BLUESTAR SOCCER FEST 2K25 — Complete Streamlit Dashboard
# Author: AI Assistant | Updated: 2025-08-28
# What’s new in this build:
# - Tabs bar is STICKY (frozen) — content scrolls beneath it
# - Sidebar toggle kept visible
# - Removed "Minimum goals per player" quick filter
# - Player Search is a type-to-search multiselect
# - Removed every "avg goals per player" display
# - Fixed Altair TitleParams (fontWeight)
# - Title shows football emoji correctly
# - Robust World Cup trophy watermark background

from __future__ import annotations

import base64
from datetime import datetime
from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import altair as alt
import pandas as pd
import requests
import streamlit as st

# Optional imports with fallbacks
PLOTLY_AVAILABLE = False
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    px = None
    go = None

# ====================== CONFIGURATION =============================
st.set_page_config(
    page_title="ABEER BLUESTAR SOCCER FEST 2K25",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ====================== STYLING ===================================
def inject_advanced_css():
    st.markdown(
        """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root{
          /* Adjust this if the sticky tabs should sit a bit lower/higher */
          --sticky-tabs-top: 52px;
        }

        .stApp {
            font-family: 'Poppins', system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .block-container {
            padding-top: 0.5rem;
            padding-bottom: 2rem;
            max-width: 98vw;
            width: 98vw;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(15px);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.15);
            margin: 1rem auto;
            position: relative; /* for z layering */
            z-index: 1;
        }

        /* Keep header/toolbar visible so sidebar toggle shows */
        #MainMenu, footer, .stDeployButton,
        div[data-testid="stDecoration"],
        div[data-testid="stStatusWidget"] {
            display: none !important;
        }

        /* ----- STICKY TABS (freeze the pane just below the tabs) ----- */
        .block-container [data-testid="stTabs"]:first-of-type{
            position: sticky;
            top: var(--sticky-tabs-top);
            z-index: 6;                           /* above content, below dialogs */
            background: rgba(255,255,255,0.96);   /* frosted background */
            backdrop-filter: blur(8px);
            border-bottom: 1px solid #e2e8f0;
            padding-top: .25rem;
            padding-bottom: .25rem;
            margin-top: .25rem;
        }

        /* App title */
        .app-title{
            display:flex; align-items:center; justify-content:center; gap:12px;
            margin: .75rem 0 1.0rem;
        }
        .app-title .ball{
            font-size: 32px; line-height:1;
            filter: drop-shadow(0 2px 4px rgba(0,0,0,.15));
        }
        .app-title .title{
            font-weight:700; letter-spacing:.05em;
            font-size: clamp(22px, 3.5vw, 36px);
            background: linear-gradient(45deg, #0ea5e9, #1e40af, #7c3aed);
            -webkit-background-clip: text; background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        /* Buttons */
        .stButton > button, .stDownloadButton > button {
            background: linear-gradient(135deg, #0ea5e9, #3b82f6) !important;
            color: white !important;
            border: 0 !important;
            border-radius: 12px !important;
            padding: 0.6rem 1.2rem !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(14, 165, 233, 0.3) !important;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(14, 165, 233, 0.4) !important;
            filter: brightness(1.05) !important;
        }

        /* Dataframes */
        .stDataFrame {
            border-radius: 15px !important;
            overflow: hidden !important;
            box-shadow: 0 8px 32px rgba(0,0,0,0.08) !important;
        }

        /* Metric cards */
        .metric-container {
            background: linear-gradient(135deg, rgba(14,165,233,.1), rgba(59,130,246,.05));
            border-radius: 15px;
            padding: 1.5rem;
            border-left: 4px solid #0ea5e9;
            box-shadow: 0 4px 20px rgba(14,165,233,.1);
            transition: transform .2s ease;
        }
        .metric-container:hover { transform: translateY(-3px); }

        /* Sidebar status pill */
        .status-pill { padding:.5rem .75rem; border-radius:.6rem; font-size:.85rem; margin-top:.5rem; }
        .status-ok  { background:#ecfeff; border-left:4px solid #06b6d4; color:#155e75; }
        .status-warn{ background:#fef9c3; border-left:4px solid #f59e0b; color:#713f12; }
        .status-err { background:#fee2e2; border-left:4px solid #ef4444; color:#7f1d1d; }

        @media (max-width: 768px) {
            .block-container { padding: 1rem .5rem; margin: .5rem; width: 95vw; max-width: 95vw; }
            .app-title .ball{font-size:24px;}
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Altair theme
    alt.themes.register(
        "tournament_theme",
        lambda: {
            "config": {
                "view": {"stroke": "transparent", "fill": "white"},
                "background": "white",
                "title": {"font": "Poppins", "fontSize": 18, "color": "#1e293b", "fontWeight": 600},
                "axis": {
                    "labelColor": "#64748b",
                    "titleColor": "#374151",
                    "gridColor": "#f1f5f9",
                    "labelFont": "Poppins",
                    "titleFont": "Poppins",
                },
                "legend": {
                    "labelColor": "#64748b",
                    "titleColor": "#374151",
                    "labelFont": "Poppins",
                    "titleFont": "Poppins",
                },
                "range": {
                    "category": [
                        "#0ea5e9","#34d399","#60a5fa","#f59e0b",
                        "#f87171","#a78bfa","#fb7185","#4ade80",
                    ]
                },
            }
        },
    )
    alt.themes.enable("tournament_theme")

def notify(msg: str, kind: str = "ok"):
    cls = {"ok": "status-ok", "warn": "status-warn", "err": "status-err"}.get(kind, "status-ok")
    st.markdown(f'<div class="status-pill {cls}">{msg}</div>', unsafe_allow_html=True)

# ---------- Robust Trophy watermark (DOM element, not :before) ----------
def add_world_cup_watermark(*, image_path: str | None = None,
                            image_url: str | None = None,
                            opacity: float = 0.08,
                            size: str = "68vmin",
                            y_offset: str = "6vh"):
    """Shows a big, faint trophy behind the whole app."""
    if image_path:
        ext = "svg+xml" if image_path.lower().endswith(".svg") else "png"
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        bg = f"url('data:image/{ext};base64,{b64}')"
    elif image_url:
        bg = f"url('{image_url}')"
    else:
        bg = "url('https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/1f3c6.svg')"

    st.markdown(
        f"""
    <style>
      #wc-trophy {{
        position: fixed;
        inset: 0;
        background-image: {bg};
        background-repeat: no-repeat;
        background-position: center {y_offset};
        background-size: {size};
        opacity: {opacity};
        pointer-events: none;
        z-index: 0; /* below content; .block-container has z-index:1 */
      }}
    </style>
    <div id="wc-trophy"></div>
    """,
        unsafe_allow_html=True,
    )

# ====================== DATA PROCESSING ===========================
def parse_xlsx_without_dependencies(file_bytes: bytes) -> pd.DataFrame:
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(BytesIO(file_bytes)) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            with z.open("xl/sharedStrings.xml") as f:
                root = ET.parse(f).getroot()
                for si in root.findall(".//main:si", ns):
                    text = "".join(t.text or "" for t in si.findall(".//main:t", ns))
                    shared.append(text)

        if "xl/worksheets/sheet1.xml" not in z.namelist():
            return pd.DataFrame()
        with z.open("xl/worksheets/sheet1.xml") as f:
            root = ET.parse(f).getroot()
            sheet = root.find("main:sheetData", ns)
            if sheet is None:
                return pd.DataFrame()
            rows, max_col = [], 0
            for row in sheet.findall("main:row", ns):
                rd = {}
                for cell in row.findall("main:c", ns):
                    ref = cell.attrib.get("r", "A1")
                    col_letters = "".join(ch for ch in ref if ch.isalpha())
                    col_idx = 0
                    for ch in col_letters:
                        col_idx = col_idx * 26 + (ord(ch) - 64)
                    col_idx -= 1
                    ctype = cell.attrib.get("t")
                    v = cell.find("main:v", ns)
                    val = v.text if v is not None else None
                    if ctype == "s" and val is not None:
                        i = int(val)
                        if 0 <= i < len(shared):
                            val = shared[i]
                    rd[col_idx] = val
                    max_col = max(max_col, col_idx)
                rows.append(rd)

    if not rows:
        return pd.DataFrame()
    matrix = [[r.get(i) for i in range(max_col + 1)] for r in rows]
    return pd.DataFrame(matrix)

def safe_read_excel(file_source) -> pd.DataFrame:
    if isinstance(file_source, (str, Path)):
        with open(file_source, "rb") as f:
            file_bytes = f.read()
    elif isinstance(file_source, bytes):
        file_bytes = file_source
    else:
        file_bytes = file_source.read()

    try:
        return pd.read_excel(BytesIO(file_bytes), header=None)
    except Exception:
        return parse_xlsx_without_dependencies(file_bytes)

def find_division_columns(raw_df: pd.DataFrame):
    b_col, a_col = None, None
    for row_idx in range(min(2, len(raw_df))):
        row = raw_df.iloc[row_idx].astype(str).str.strip().str.lower()
        for col_idx, cell in row.items():
            if "b division" in cell and b_col is None:
                b_col = col_idx
            elif "a division" in cell and a_col is None:
                a_col = col_idx
    if b_col is None and a_col is None:
        b_col = 0
        a_col = 5 if raw_df.shape[1] >= 8 else (4 if raw_df.shape[1] >= 7 else None)
    return b_col, a_col

def process_tournament_data(xlsx_bytes: bytes) -> pd.DataFrame:
    raw_df = safe_read_excel(xlsx_bytes)
    if raw_df.empty:
        return pd.DataFrame(columns=["Division", "Team", "Player", "Goals"])

    b_start, a_start = find_division_columns(raw_df)
    header_row = 1 if len(raw_df) > 1 else 0
    data_start_row = header_row + 1

    processed = []

    def extract_div(start_col: int | None, name: str):
        if start_col is None or start_col + 2 >= raw_df.shape[1]:
            return
        df = raw_df.iloc[data_start_row:, start_col:start_col + 3].copy()
        df.columns = ["Team", "Player", "Goals"]
        df = df.dropna(subset=["Team", "Player", "Goals"])
        df["Goals"] = pd.to_numeric(df["Goals"], errors="coerce")
        df = df.dropna(subset=["Goals"])
        df["Goals"] = df["Goals"].astype(int)
        df["Division"] = name
        processed.extend(df.to_dict("records"))

    extract_div(b_start, "B Division")
    extract_div(a_start, "A Division")

    if not processed:
        return pd.DataFrame(columns=["Division", "Team", "Player", "Goals"])
    out = pd.DataFrame(processed)
    return out[["Division", "Team", "Player", "Goals"]]

@st.cache_data(ttl=300, show_spinner=False)
def fetch_tournament_data(url: str) -> pd.DataFrame:
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        if not r.content:
            raise ValueError("Downloaded file is empty")
        return process_tournament_data(r.content)
    except Exception:
        return pd.DataFrame(columns=["Division", "Team", "Player", "Goals"])

# ====================== ANALYTICS FUNCTIONS =======================
def calculate_tournament_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "total_goals": 0,
            "total_players": 0,
            "total_teams": 0,
            "divisions": 0,
            "avg_goals_per_team": 0,
            "top_scorer_goals": 0,
            "competitive_balance": 0,
        }

    player_totals = df.groupby(["Player", "Team", "Division"])["Goals"].sum().reset_index()
    team_totals = df.groupby(["Team", "Division"])["Goals"].sum().reset_index()

    return {
        "total_goals": int(df["Goals"].sum()),
        "total_players": len(player_totals),
        "total_teams": len(team_totals),
        "divisions": df["Division"].nunique(),
        "avg_goals_per_team": round(df["Goals"].sum() / max(1, len(team_totals)), 2),
        "top_scorer_goals": int(player_totals["Goals"].max()) if not player_totals.empty else 0,
        "competitive_balance": round(team_totals["Goals"].std(), 2) if len(team_totals) > 1 else 0,
    }

def get_top_performers(df: pd.DataFrame, top_n: int = 10) -> dict:
    if df.empty:
        return {"players": pd.DataFrame(), "teams": pd.DataFrame()}
    top_players = (
        df.groupby(["Player", "Team", "Division"])["Goals"]
        .sum()
        .reset_index()
        .sort_values(["Goals", "Player"], ascending=[False, True])
        .head(top_n)
    )
    top_teams = (
        df.groupby(["Team", "Division"])["Goals"]
        .sum()
        .reset_index()
        .sort_values("Goals", ascending=False)
        .head(top_n)
    )
    return {"players": top_players, "teams": top_teams}

def create_division_comparison(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    division_stats = (
        df.groupby("Division")
        .agg(Goals_sum=("Goals", "sum"), Goals_mean=("Goals", "mean"), Records=("Goals", "count"),
             Teams=("Team", "nunique"), Players=("Player", "nunique"))
        .round(2)
        .reset_index()
        .rename(columns={"Goals_sum": "Total_Goals", "Goals_mean": "Avg_Goals", "Records": "Total_Records"})
    )
    total_goals = division_stats["Total_Goals"].sum()
    division_stats["Goal_Share_Pct"] = (division_stats["Total_Goals"] / total_goals * 100).round(1) if total_goals else 0
    return division_stats

# ====================== VISUALIZATION FUNCTIONS ===================
def create_horizontal_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, color_scheme: str = "blues") -> alt.Chart:
    if df.empty:
        return alt.Chart(pd.DataFrame({"note": ["No data available"]})).mark_text().encode(text="note:N")
    max_val = int(df[x_col].max()) if not df.empty else 1
    tick_values = list(range(0, max_val + 1)) if max_val <= 50 else None
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, opacity=0.85, stroke="white", strokeWidth=1)
        .encode(
            x=alt.X(f"{x_col}:Q", title="Goals", axis=alt.Axis(format="d", tickMinStep=1, values=tick_values, gridOpacity=0.3), scale=alt.Scale(domainMin=0, nice=False)),
            y=alt.Y(f"{y_col}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=200)),
            color=alt.Color(f"{x_col}:Q", scale=alt.Scale(scheme=color_scheme), legend=None),
            tooltip=[alt.Tooltip(f"{y_col}:N", title=y_col), alt.Tooltip(f"{x_col}:Q", title="Goals", format="d")],
        )
        .properties(height=max(300, min(600, len(df) * 25)), title=alt.TitleParams(text=title, fontSize=16, anchor="start", fontWeight=600))
        .resolve_scale(color="independent")
    )

def create_division_donut_chart(df: pd.DataFrame) -> alt.Chart:
    if df.empty:
        return alt.Chart(pd.DataFrame({"note": ["No data available"]})).mark_text().encode(text="note:N")

    division_data = df.groupby("Division")["Goals"].sum().reset_index()
    sel = alt.selection_single(fields=["Division"], empty="none")

    base = (
        alt.Chart(division_data)
        .add_selection(sel)
        .properties(
            width=300,
            height=300,
            title=alt.TitleParams(text="Goals Distribution by Division", fontSize=16, fontWeight=600),
        )
    )

    outer = (
        base.mark_arc(innerRadius=60, outerRadius=120, stroke="white", strokeWidth=2)
        .encode(
            theta=alt.Theta("Goals:Q", title="Goals"),
            color=alt.Color("Division:N", scale=alt.Scale(range=["#0ea5e9", "#f59e0b"]), title="Division"),
            opacity=alt.condition(sel, alt.value(1.0), alt.value(0.8)),
            tooltip=["Division:N", alt.Tooltip("Goals:Q", format="d")],
        )
    )

    center_text = (
        base.mark_text(align="center", baseline="middle", fontSize=18, fontWeight="bold", color="#1e293b")
        .encode(text=alt.value(f"Total\n{int(division_data['Goals'].sum())}"))
    )

    return outer + center_text

def create_advanced_scatter_plot(df: pd.DataFrame):
    if df.empty:
        if PLOTLY_AVAILABLE:
            fig = go.Figure(); fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False); return fig
        return alt.Chart(pd.DataFrame({"note": ["No data available"]})).mark_text().encode(text="note:N")

    team_stats = df.groupby(["Team", "Division"]).agg(Players=("Player", "nunique"), Goals=("Goals", "sum")).reset_index()

    if PLOTLY_AVAILABLE:
        fig = px.scatter(
            team_stats, x="Players", y="Goals", color="Division", size="Goals",
            hover_name="Team", hover_data={"Players": True, "Goals": True},
            title="Team Performance: Players vs Total Goals",
        )
        fig.update_traces(marker=dict(sizemode="diameter", sizemin=8, sizemax=30, line=dict(width=2, color="white"), opacity=0.85))
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Poppins", size=12),
            title=dict(font=dict(size=16, color="#1e293b")),
            xaxis=dict(title="Number of Players in Team", gridcolor="#f1f5f9", zeroline=False),
            yaxis=dict(title="Total Goals", gridcolor="#f1f5f9", zeroline=False),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            height=400,
        )
        return fig

    return (
        alt.Chart(team_stats)
        .mark_circle(size=100, opacity=0.85)
        .encode(
            x=alt.X("Players:Q", title="Number of Players in Team"),
            y=alt.Y("Goals:Q", title="Total Goals"),
            color=alt.Color("Division:N", scale=alt.Scale(range=["#0ea5e9", "#f59e0b"]), title="Division"),
            size=alt.Size("Goals:Q", legend=None),
            tooltip=["Team:N", "Division:N", "Players:Q", "Goals:Q"],
        )
        .properties(title="Team Performance: Players vs Total Goals", height=400)
    )

def create_goals_distribution_histogram(df: pd.DataFrame):
    if df.empty:
        if PLOTLY_AVAILABLE:
            fig = go.Figure(); fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False); return fig
        return alt.Chart(pd.DataFrame({"note": ["No data available"]})).mark_text().encode(text="note:N")

    player_goals = df.groupby(["Player", "Team"])["Goals"].sum().values
    if PLOTLY_AVAILABLE:
        fig = go.Figure(
            data=[go.Histogram(x=player_goals, nbinsx=max(1, len(set(player_goals))), marker_line_color="white", marker_line_width=1.5, opacity=0.85, hovertemplate="<b>%{x} Goals</b><br>%{y} Players<extra></extra>")]
        )
        fig.update_layout(
            title="Distribution of Goals per Player",
            xaxis_title="Goals per Player",
            yaxis_title="Number of Players",
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Poppins", size=12),
            title_font=dict(size=16, color="#1e293b"),
            xaxis=dict(gridcolor="#f1f5f9", zeroline=False),
            yaxis=dict(gridcolor="#f1f5f9", zeroline=False),
            height=400,
        )
        return fig

    hist_df = pd.DataFrame({"Goals": player_goals})
    return (
        alt.Chart(hist_df)
        .mark_bar(opacity=0.85)
        .encode(
            x=alt.X("Goals:Q", bin=alt.Bin(maxbins=10), title="Goals per Player"),
            y=alt.Y("count():Q", title="Number of Players"),
            tooltip=[alt.Tooltip("Goals:Q", bin=True, title="Goals"), alt.Tooltip("count():Q", title="Players")],
        )
        .properties(title="Distribution of Goals per Player", height=400)
    )

# ====================== UI COMPONENTS =============================
def display_metric_cards(stats: dict):
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""<div class="metric-container"><div style="font-size:2.5rem;font-weight:700;color:#0ea5e9;margin-bottom:.5rem;">{stats['total_goals']}</div><div style="color:#64748b;font-weight:500;text-transform:uppercase;letter-spacing:.05em;">TOTAL GOALS</div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="metric-container"><div style="font-size:2.5rem;font-weight:700;color:#0ea5e9;margin-bottom:.5rem;">{stats['total_players']}</div><div style="color:#64748b;font-weight:500;text-transform:uppercase;letter-spacing:.05em;">PLAYERS</div></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="metric-container"><div style="font-size:2.5rem;font-weight:700;color:#0ea5e9;margin-bottom:.5rem;">{stats['total_teams']}</div><div style="color:#64748b;font-weight:500;text-transform:uppercase;letter-spacing:.05em;">TEAMS</div></div>""", unsafe_allow_html=True)

def display_insights_cards(df: pd.DataFrame, scope: str = "Tournament"):
    if df.empty:
        st.info("📊 No data available for insights.")
        return

    stats = calculate_tournament_stats(df)
    top_perf = get_top_performers(df, 5)
    division_cmp = create_division_comparison(df)

    if not top_perf["teams"].empty:
        tt = top_perf["teams"].iloc[0]
        top_team_name, top_team_goals = tt["Team"], int(tt["Goals"])
    else:
        top_team_name, top_team_goals = "—", 0

    if not top_perf["players"].empty:
        tp = top_perf["players"].iloc[0]
        top_player_name, top_player_team, top_player_goals = tp["Player"], tp["Team"], int(tp["Goals"])
    else:
        top_player_name, top_player_team, top_player_goals = "—", "—", 0

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
        <div style="background:white;padding:1.5rem;border-radius:15px;box-shadow:0 4px 20px rgba(0,0,0,.05);border-left:4px solid #0ea5e9;margin-bottom:1rem;">
            <div style="font-weight:600;color:#0ea5e9;margin-bottom:.5rem;font-size:1.1rem;">🏆 Top Performing Team</div>
            <div style="color:#374151;line-height:1.5;"><strong>{top_team_name}</strong> leads with <strong>{top_team_goals} goals</strong>.</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        if not division_cmp.empty and len(division_cmp) > 1:
            def get_vals(div):
                s = division_cmp[division_cmp["Division"] == div]
                return (int(s["Total_Goals"].iloc[0]), s["Goal_Share_Pct"].iloc[0]) if not s.empty else (0, 0)
            b_goals, b_pct = get_vals("B Division")
            a_goals, a_pct = get_vals("A Division")

            st.markdown(
                f"""
            <div style="background:white;padding:1.5rem;border-radius:15px;box-shadow:0 4px 20px rgba(0,0,0,.05);border-left:4px solid #f59e0b;margin-bottom:1rem;">
                <div style="font-weight:600;color:#f59e0b;margin-bottom:.5rem;font-size:1.1rem;">🎯 Division Performance</div>
                <div style="color:#374151;line-height:1.5;">
                    <strong>B Division:</strong> {b_goals} goals ({b_pct}%)<br>
                    <strong>A Division:</strong> {a_goals} goals ({a_pct}%)<br>
                    {"B Division shows higher scoring activity" if b_goals > a_goals else ("A Division leads in goal production" if a_goals > b_goals else "Both divisions are evenly matched")}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown(
            f"""
        <div style="background:white;padding:1.5rem;border-radius:15px;box-shadow:0 4px 20px rgba(0,0,0,.05);border-left:4px solid #34d399;margin-bottom:1rem;">
            <div style="font-weight:600;color:#34d399;margin-bottom:.5rem;font-size:1.1rem;">⚽ Leading Scorer</div>
            <div style="color:#374151;line-height:1.5;"><strong>{top_player_name}</strong> ({top_player_team}) with <strong>{top_player_goals} goals</strong>.</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        player_goals = df.groupby(["Player", "Team"])["Goals"].sum()
        goals_1 = int((player_goals == 1).sum())
        goals_2_plus = int((player_goals >= 2).sum())

        st.markdown(
            f"""
        <div style="background:white;padding:1.5rem;border-radius:15px;box-shadow:0 4px 20px rgba(0,0,0,.05);border-left:4px solid #a78bfa;margin-bottom:1rem;">
            <div style="font-weight:600;color:#a78bfa;margin-bottom:.5rem;font-size:1.1rem;">📊 Competition Balance</div>
            <div style="color:#374151;line-height:1.5;">
                <strong>{goals_1} players</strong> scored 1 goal, <strong>{goals_2_plus} players</strong> scored 2+ goals<br>
                {
                    "Well-balanced competition" if stats['competitive_balance'] < 2
                    else ("Some teams dominating" if stats['competitive_balance'] > 4 else "Moderate competition spread")
                }
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

def create_enhanced_data_table(df: pd.DataFrame, table_type: str = "records"):
    if df.empty:
        st.info(f"📋 No {table_type} data available with current filters.")
        return

    if table_type == "records":
        display_df = df.sort_values("Goals", ascending=False).reset_index(drop=True)
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Division": st.column_config.TextColumn("Division", width="small"),
                "Team": st.column_config.TextColumn("Team", width="medium"),
                "Player": st.column_config.TextColumn("Player", width="large"),
                "Goals": st.column_config.NumberColumn("Goals", format="%d", width="small"),
            },
        )

    elif table_type == "teams":
        teams_summary = df.groupby(["Team", "Division"]).agg(Players=("Player", "nunique"), Total_Goals=("Goals", "sum")).reset_index()
        top_rows = []
        for team in teams_summary["Team"].unique():
            team_data = df[df["Team"] == team]
            if team_data.empty:
                continue
            s = team_data.groupby("Player")["Goals"].sum()
            top_rows.append({"Team": team, "Top_Scorer": s.idxmax(), "Top_Scorer_Goals": int(s.max())})
        top_df = pd.DataFrame(top_rows)
        teams_display = teams_summary.merge(top_df, on="Team", how="left").sort_values("Total_Goals", ascending=False)

        st.dataframe(
            teams_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Division": st.column_config.TextColumn("Division", width="small"),
                "Team": st.column_config.TextColumn("Team", width="medium"),
                "Players": st.column_config.NumberColumn("Players", format="%d", width="small"),
                "Total_Goals": st.column_config.NumberColumn("Total Goals", format="%d", width="small"),
                "Top_Scorer": st.column_config.TextColumn("Top Scorer", width="large"),
                "Top_Scorer_Goals": st.column_config.NumberColumn("Goals", format="%d", width="small"),
            },
        )

    elif table_type == "players":
        players_summary = df.groupby(["Player", "Team", "Division"])["Goals"].sum().reset_index()
        players_summary = players_summary.sort_values(["Goals", "Player"], ascending=[False, True])
        players_summary.insert(0, "Rank", range(1, len(players_summary) + 1))
        st.dataframe(
            players_summary,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
                "Player": st.column_config.TextColumn("Player", width="large"),
                "Team": st.column_config.TextColumn("Team", width="medium"),
                "Division": st.column_config.TextColumn("Division", width="small"),
                "Goals": st.column_config.NumberColumn("Goals", format="%d", width="small"),
            },
        )

def create_comprehensive_zip_report(full_df: pd.DataFrame, filtered_df: pd.DataFrame) -> bytes:
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as z:
        if not full_df.empty:
            z.writestr("01_full_tournament_data.csv", full_df.to_csv(index=False))
        if not filtered_df.empty:
            z.writestr("02_filtered_tournament_data.csv", filtered_df.to_csv(index=False))
            teams = (
                filtered_df.groupby(["Team", "Division"])
                .agg(Unique_Players=("Player", "nunique"), Total_Records=("Goals", "count"), Total_Goals=("Goals", "sum"),
                     Avg_Goals=("Goals", "mean"), Max_Goals=("Goals", "max"))
                .round(2)
                .reset_index()
            )
            z.writestr("03_teams_detailed_analysis.csv", teams.to_csv(index=False))
            players = filtered_df.groupby(["Player", "Team", "Division"])["Goals"].sum().reset_index()
            players = players.sort_values(["Goals", "Player"], ascending=[False, True])
            players.insert(0, "Rank", range(1, len(players) + 1))
            z.writestr("04_players_ranking.csv", players.to_csv(index=False))
            div_cmp = create_division_comparison(filtered_df)
            if not div_cmp.empty:
                z.writestr("05_division_comparison.csv", div_cmp.to_csv(index=False))
            stats = calculate_tournament_stats(filtered_df)
            z.writestr("06_tournament_statistics.csv", pd.DataFrame([stats]).to_csv(index=False))
        z.writestr("README.txt", f"ABEER BLUESTAR SOCCER FEST 2K25 - Data Package\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def create_download_section(full_df: pd.DataFrame, filtered_df: pd.DataFrame):
    st.subheader("📥 Download Reports")
    st.caption("**Full** = all data ignoring filters, **Filtered** = current view with active filters")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📊 Data Exports")
        if not full_df.empty:
            st.download_button(
                label="⬇️ Download FULL Dataset (CSV)",
                data=full_df.to_csv(index=False),
                file_name=f"tournament_full_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help="Download complete tournament data",
            )
        if not filtered_df.empty:
            st.download_button(
                label="⬇️ Download FILTERED Dataset (CSV)",
                data=filtered_df.to_csv(index=False),
                file_name=f"tournament_filtered_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help="Download data with current filters applied",
            )

    with col2:
        st.subheader("🏆 Summary Reports")
        if not filtered_df.empty:
            teams_summary = (
                filtered_df.groupby(["Team", "Division"]).agg(Players_Count=("Player", "nunique"), Total_Goals=("Goals", "sum")).reset_index()
            ).sort_values("Total_Goals", ascending=False)
            st.download_button(
                label="⬇️ Download TEAMS Summary (CSV)",
                data=teams_summary.to_csv(index=False),
                file_name=f"teams_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help="Download team performance summary",
            )

            players_summary = filtered_df.groupby(["Player", "Team", "Division"])["Goals"].sum().reset_index()
            players_summary = players_summary.sort_values(["Goals", "Player"], ascending=[False, True])
            players_summary.insert(0, "Rank", range(1, len(players_summary) + 1))
            st.download_button(
                label="⬇️ Download PLAYERS Summary (CSV)",
                data=players_summary.to_csv(index=False),
                file_name=f"players_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help="Download top scorers summary",
            )

    with col3:
        st.subheader("📦 Complete Package")
        if st.button("📦 Generate Complete Report Package", help="Generate ZIP with all reports and analytics"):
            z = create_comprehensive_zip_report(full_df, filtered_df)
            st.download_button(
                label="⬇️ Download Complete Package (ZIP)",
                data=z,
                file_name=f"tournament_complete_package_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                mime="application/zip",
                help="Download ZIP containing all data, summaries, and analytics",
            )

# ====================== MAIN APPLICATION ==========================
def main():
    inject_advanced_css()

    # Title
    st.markdown("""
<div class="app-title">
  <span class="ball">⚽</span>
  <span class="title">ABEER BLUESTAR SOCCER FEST 2K25</span>
</div>
""", unsafe_allow_html=True)

    # Trophy background (URL or local file)
    add_world_cup_watermark(
        image_url="https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/1f3c6.svg",
        opacity=0.10,
        size="68vmin",
        y_offset="6vh"
    )

    GOOGLE_SHEETS_URL = (
        "https://docs.google.com/spreadsheets/d/e/"
        "2PACX-1vRpCD-Wh_NnGQjJ1Mh3tuU5Mdcl8TK41JopMUcSnfqww8wkPXKKgRyg7v4sC_vuUw/pub?output=xlsx"
    )

    # Sidebar
    with st.sidebar:
        st.header("🎛️ Dashboard Controls")

        if st.button("🔄 Refresh Tournament Data", use_container_width=True):
            st.cache_data.clear()
            st.session_state["last_refresh"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            notify("Data cache cleared. Reloading…", "ok")
            st.rerun()

        last_refresh = st.session_state.get("last_refresh", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        st.caption(f"🕒 Last refreshed: {last_refresh}")
        st.divider()

        # Load data
        with st.spinner("📡 Loading tournament data…"):
            tournament_data = fetch_tournament_data(GOOGLE_SHEETS_URL)

        if tournament_data.empty:
            notify("No tournament data available. Check the published sheet link/permissions.", "err")
            st.stop()

        full_tournament_data = tournament_data.copy()

        st.header("🔍 Data Filters")

        # Division filter
        division_options = ["All Divisions"] + sorted(tournament_data["Division"].unique().tolist())
        selected_division = st.selectbox("📊 Division", division_options, key="division_filter")
        if selected_division != "All Divisions":
            tournament_data = tournament_data[tournament_data["Division"] == selected_division]

        # Team filter
        available_teams = sorted(tournament_data["Team"].unique().tolist())
        selected_teams = st.multiselect(
            "🏆 Teams (optional)",
            available_teams,
            key="teams_filter",
            help="Select specific teams to focus on",
            placeholder="Type to search teams…",
        )
        if selected_teams:
            tournament_data = tournament_data[tournament_data["Team"].isin(selected_teams)]

        # Player search (type-to-search list)
        st.subheader("👤 Player Search")
        player_names = sorted(tournament_data["Player"].dropna().astype(str).unique().tolist())
        selected_players = st.multiselect(
            "Type to search and select players",
            options=player_names,
            default=[],
            key="players_filter",
            placeholder="Start typing a player name…",
            help="Type to search. You can select one or multiple players.",
        )
        if selected_players:
            tournament_data = tournament_data[tournament_data["Player"].isin(selected_players)]

    # Tabs (this first tabs block is sticky via CSS above)
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["📊 OVERVIEW", "⚡ QUICK INSIGHTS", "🏆 TEAMS", "👤 PLAYERS", "📈 ANALYTICS", "📥 DOWNLOADS"]
    )

    current_stats = calculate_tournament_stats(tournament_data)
    top_performers = get_top_performers(tournament_data, 10)

    # TAB 1
    with tab1:
        st.header("📊 Tournament Overview")
        display_metric_cards(current_stats)
        st.divider()
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("🎯 Goal Scoring Records")
            create_enhanced_data_table(tournament_data, "records")
        with col2:
            if not tournament_data.empty:
                st.subheader("🏁 Division Distribution")
                st.altair_chart(create_division_donut_chart(tournament_data), use_container_width=True)
        if not tournament_data.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("🏆 Goals by Team")
                team_goals = tournament_data.groupby("Team")["Goals"].sum().reset_index().sort_values("Goals", ascending=False).head(10)
                if not team_goals.empty:
                    st.altair_chart(create_horizontal_bar_chart(team_goals, "Goals", "Team", "Top 10 Teams by Goals", "blues"), use_container_width=True)
            with c2:
                st.subheader("⚽ Top Scorers")
                if not top_performers["players"].empty:
                    ts = top_performers["players"].head(10).copy()
                    ts["Display_Name"] = ts["Player"] + " (" + ts["Team"] + ")"
                    st.altair_chart(create_horizontal_bar_chart(ts, "Goals", "Display_Name", "Top 10 Players by Goals", "greens"), use_container_width=True)

    # TAB 2
    with tab2:
        st.header("⚡ Quick Tournament Insights")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🎯 Total Goals", current_stats["total_goals"])
        c2.metric("👥 Active Players", current_stats["total_players"])
        c3.metric("🏆 Teams", current_stats["total_teams"])
        c4.metric("📊 Divisions", current_stats["divisions"])
        st.divider()
        display_insights_cards(tournament_data, "Current View" if len(tournament_data) < len(full_tournament_data) else "Tournament")
        if tournament_data["Division"].nunique() > 1:
            st.subheader("🔄 Division Comparison")
            division_comparison = create_division_comparison(tournament_data)
            if not division_comparison.empty:
                st.dataframe(
                    division_comparison,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Total_Goals": st.column_config.NumberColumn("Total Goals", format="%d"),
                        "Avg_Goals": st.column_config.NumberColumn("Avg Goals", format="%.2f"),
                        "Total_Records": st.column_config.NumberColumn("Records", format="%d"),
                        "Teams": st.column_config.NumberColumn("Teams", format="%d"),
                        "Players": st.column_config.NumberColumn("Players", format="%d"),
                        "Goal_Share_Pct": st.column_config.NumberColumn("Share %", format="%.1f%%"),
                    },
                )

    # TAB 3
    with tab3:
        st.header("🏆 Teams Analysis")
        if tournament_data.empty:
            st.info("🔍 No teams match your current filters.")
        else:
            st.subheader("📋 Teams Summary")
            create_enhanced_data_table(tournament_data, "teams")
            st.divider()
            st.subheader("📊 Team Performance Analysis")
            team_analysis = tournament_data.groupby(["Team", "Division"]).agg(Players=("Player", "nunique"), Goals=("Goals", "sum")).reset_index().sort_values("Goals", ascending=False)
            if not team_analysis.empty:
                st.altair_chart(create_horizontal_bar_chart(team_analysis.head(15), "Goals", "Team", "Team Goals Distribution", "viridis"), use_container_width=True)

    # TAB 4
    with tab4:
        st.header("👤 Players Analysis")
        if tournament_data.empty:
            st.info("🔍 No players match your current filters.")
        else:
            st.subheader("📋 Players Ranking")
            create_enhanced_data_table(tournament_data, "players")
            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("🥇 Top Scorers by Division")
                for division in tournament_data["Division"].unique():
                    div_data = tournament_data[tournament_data["Division"] == division]
                    div_top = div_data.groupby(["Player", "Team"])["Goals"].sum().reset_index().sort_values("Goals", ascending=False).head(5)
                    st.write(f"**{division}**")
                    if not div_top.empty:
                        for _, row in div_top.iterrows():
                            st.write(f"• {row['Player']} ({row['Team']}) — {int(row['Goals'])} goals")
                    else:
                        st.write("• No players found")
                    st.write("")
            with c2:
                st.subheader("📊 Player Statistics")
                player_goals = tournament_data.groupby(["Player", "Team"])["Goals"].sum()
                st.metric("🎯 Highest Individual Score", int(player_goals.max()) if not player_goals.empty else 0)
                st.metric("👥 Players with 2+ Goals", int((player_goals >= 2).sum()))
                st.metric("⚽ Single Goal Scorers", int((player_goals == 1).sum()))

    # TAB 5
    with tab5:
        st.header("📈 Advanced Analytics")
        if tournament_data.empty:
            st.info("🔍 No data available for analytics with current filters.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("📊 Goals Distribution")
                dist = create_goals_distribution_histogram(tournament_data)
                (st.plotly_chart if PLOTLY_AVAILABLE else st.altair_chart)(dist, use_container_width=True)
            with c2:
                st.subheader("🎯 Team Performance Matrix")
                scatter = create_advanced_scatter_plot(tournament_data)
                (st.plotly_chart if PLOTLY_AVAILABLE else st.altair_chart)(scatter, use_container_width=True)
            st.divider()
            st.subheader("🔍 Detailed Performance Metrics")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**🏆 Top Teams (Total Goals)**")
                tg = tournament_data.groupby("Team")["Goals"].sum().sort_values(ascending=False).head(5)
                for team, g in tg.items():
                    st.write(f"• **{team}**: {int(g)} goals")
            with c2:
                st.markdown("**⚽ Scoring Patterns**")
                counts = tournament_data["Goals"].value_counts().sort_index()
                for g, c in counts.items():
                    pct = (c / len(tournament_data) * 100)
                    st.write(f"• **{int(g)} goal{'s' if g != 1 else ''}**: {int(c)} records ({pct:.1f}%)")
            with c3:
                st.markdown("**🎯 Division Insights**")
                for division in tournament_data["Division"].unique():
                    div_data = tournament_data[tournament_data["Division"] == division]
                    total_goals = int(div_data["Goals"].sum())
                    unique_players = int(div_data["Player"].nunique())
                    st.write(f"• **{division}**:")
                    st.write(f"  - {total_goals} total goals")
                    st.write(f"  - {unique_players} unique players")

    # TAB 6
    with tab6:
        create_download_section(full_tournament_data, tournament_data)

# ====================== ENTRY POINT ===============================
if __name__ == "__main__":
    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        main()
    except Exception as e:
        st.error(f"🚨 Application Error: {e}")
        st.exception(e)
