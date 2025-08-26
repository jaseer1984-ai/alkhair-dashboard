from __future__ import annotations

import io
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

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
    # ⚠️ Put your sheet URL here (direct /d/{id}/… recommended)
    DEFAULT_PUBLISHED_URL = (
        "https://docs.google.com/spreadsheets/d/1sYd2tCx_u3vh41ecdQNPVAOOn7qbu_Ns/edit?gid=1185316772#gid=1185316772"
    )
    BRANCHES = {
        "Al khair - 102": {"name": "Al Khair", "code": "102", "color": "#3b82f6"},
        "Noora - 104": {"name": "Noora", "code": "104", "color": "#10b981"},
        "Hamra - 109": {"name": "Hamra", "code": "109", "color": "#f59e0b"},
        "Magnus - 107": {"name": "Magnus", "code": "107", "color": "#8b5cf6"},
    }
    TARGETS = {"sales_achievement": 95.0, "nob_achievement": 90.0, "abv_achievement": 90.0}


config = Config()
st.set_page_config(page_title=config.PAGE_TITLE, layout=config.LAYOUT, initial_sidebar_state="expanded")


# =========================================
# CSS (desktop + mobile responsive)
# =========================================
def apply_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&display=swap');
        html, body, [data-testid="stAppViewContainer"] { font-family: 'Inter', sans-serif; }
        .main .block-container { padding: 1.2rem 2rem; max-width: 1500px; }
        .title { font-weight:900; font-size:1.6rem; margin-bottom:.25rem; }
        .subtitle { color:#6b7280; font-size:.95rem; margin-bottom:1rem; }

        [data-testid="metric-container"] {
            background:#fff !important; border-radius:16px !important; border:1px solid rgba(0,0,0,.06)!important; padding:18px!important;
        }
        .card { background:#fcfcfc; border:1px solid #f1f5f9; border-radius:16px; padding:16px; height:100%; }

        .pill { display:inline-block; padding:4px 10px; border-radius:999px; font-weight:700; font-size:.80rem; }
        .pill.excellent { background:#ecfdf5; color:#059669; }
        .pill.good      { background:#eff6ff; color:#2563eb; }
        .pill.warn      { background:#fffbeb; color:#d97706; }
        .pill.danger    { background:#fef2f2; color:#dc2626; }

        .stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: none; }
        .stTabs [data-baseweb="tab"] {
          border-radius: 10px 10px 0 0 !important;
          padding: 10px 18px !important; font-weight: 700 !important;
          background: #f3f4f6 !important; color: #374151 !important;
          border: 1px solid #e5e7eb !important; border-bottom: none !important;
        }
        .stTabs [data-baseweb="tab"]:hover { background:#e5e7eb !important; }
        .stTabs [aria-selected="true"] { color: #fff !important; box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
        .stTabs [data-baseweb="tab"]:nth-child(1)[aria-selected="true"] { background: linear-gradient(135deg,#ef4444 0%,#f87171 100%) !important; }
        .stTabs [data-baseweb="tab"]:nth-child(2)[aria-selected="true"] { background: linear-gradient(135deg,#3b82f6 0%,#60a5fa 100%) !important; }
        .stTabs [data-baseweb="tab"]:nth-child(3)[aria-selected="true"] { background: linear-gradient(135deg,#8b5cf6 0%,#a78bfa 100%) !important; }
        .stTabs [data-baseweb="tab"]:nth-child(4)[aria-selected="true"] { background: linear-gradient(135deg,#10b981 0%,#34d399 100%) !important; }

        [data-testid="stSidebar"] {
          background: #f7f9fc; border-right: 1px solid #e5e7eb; min-width: 280px; max-width: 320px;
        }
        [data-testid="stSidebar"] .block-container { padding: 18px 16px 20px; }
        .sb-title { display:flex; gap:10px; align-items:center; font-weight:900; font-size:1.05rem; }
        .sb-subtle { color:#6b7280; font-size:.85rem; margin:6px 0 12px; }
        .sb-section { font-size:.80rem; font-weight:800; letter-spacing:.02em; text-transform:uppercase; color:#64748b; margin:12px 4px 6px; }
        .sb-card { background:#ffffff; border:1px solid #eef2f7; border-radius:14px; padding:12px; box-shadow: 0 1px 2px rgba(16,24,40,.04); }
        .sb-hr { height:1px; background:#e5e7eb; margin:12px 0; border-radius:999px; }
        .sb-foot { margin-top:10px; font-size:.75rem; color:#9ca3af; text-align:center; }

        @media (max-width: 680px) {
          .main .block-container { padding: 0.6rem 0.8rem; }
          .title { font-size: 1.3rem; }
          .subtitle { font-size: .85rem; }
          [data-testid="stSidebar"] { display: none; }
          .mobile-only { display: block !important; }
          .desktop-only { display: none !important; }
        }
        .mobile-only { display: none; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================
# LOADERS (no html5lib required)
# =========================================
def _extract_spreadsheet_id(url: str) -> Optional[str]:
    m = re.search(r"/spreadsheets/d/([^/]+)/", url)
    return m.group(1) if m else None


def _xlsx_export_url(spreadsheet_id: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=xlsx"


def _sheet_list_from_html(url: str) -> List[Tuple[str, int]]:
    """
    Read the sheet edit page HTML and extract (title, gid) via regex.
    Works without BeautifulSoup/html5lib.
    """
    headers = {"User-Agent": "Mozilla/5.0 (Streamlit Loader)"}
    r = requests.get(url, timeout=60, headers=headers)
    r.raise_for_status()
    html = r.text

    # Matches: "sheetId":123456789,"title":"Sheet Name"
    pairs: List[Tuple[str, int]] = []
    pattern = re.compile(r'"sheetId"\s*:\s*(\d+)\s*,\s*"title"\s*:\s*"([^"]+)"')
    for m in pattern.finditer(html):
        gid = int(m.group(1))
        title = m.group(2)
        pairs.append((title, gid))

    # Deduplicate while preserving order
    seen = set()
    unique_pairs = []
    for title, gid in pairs:
        if gid not in seen:
            seen.add(gid)
            unique_pairs.append((title, gid))
    return unique_pairs


def _csv_export_url(spreadsheet_id: str, gid: int) -> str:
    return f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv&gid={gid}"


@st.cache_data(show_spinner=False)
def load_workbook_from_gsheet(sheet_url: str) -> Dict[str, pd.DataFrame]:
    """
    1) Try whole-workbook XLSX export.
    2) If that fails, discover tabs (title,gid) from edit HTML and fetch each tab via CSV export.
    """
    url_in = (sheet_url or config.DEFAULT_PUBLISHED_URL).strip()
    ssid = _extract_spreadsheet_id(url_in)
    if not ssid:
        st.error("Invalid Google Sheet URL (missing /d/{id}/).")
        return {}

    # Try XLSX
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Streamlit Loader)"}
        r = requests.get(_xlsx_export_url(ssid), timeout=60, headers=headers)
        r.raise_for_status()
        xls = pd.ExcelFile(io.BytesIO(r.content))
        sheets: Dict[str, pd.DataFrame] = {}
        for sn in xls.sheet_names:
            df = xls.parse(sn)
            df = df.dropna(how="all").dropna(axis=1, how="all")
            if not df.empty:
                sheets[sn] = df
        if sheets:
            return sheets
    except Exception:
        st.info("XLSX export failed, using CSV-per-sheet fallback…")

    # CSV-per-sheet fallback
    try:
        name_gid_list = _sheet_list_from_html(url_in)
        if not name_gid_list:
            st.error("Could not discover sheet tabs (names/gids). Ensure the sheet is public or published.")
            return {}

        sheets: Dict[str, pd.DataFrame] = {}
        headers = {"User-Agent": "Mozilla/5.0 (Streamlit Loader)"}
        for title, gid in name_gid_list:
            try:
                csv_url = _csv_export_url(ssid, gid)
                resp = requests.get(csv_url, timeout=60, headers=headers)
                resp.raise_for_status()
                df = pd.read_csv(io.StringIO(resp.content.decode("utf-8", errors="ignore")))
                df = df.dropna(how="all").dropna(axis=1, how="all")
                if not df.empty:
                    sheets[title] = df
            except Exception:
                continue

        if sheets:
            return sheets
        st.warning("CSV fallback parsed no non-empty sheets.")
        return {}
    except Exception as e:
        st.exception(e)
        return {}


# =========================================
# COMMON HELPERS
# =========================================
def _parse_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False).str.strip(),
        errors="coerce",
    )


# =========================================
# SALES/NOB/ABV BRANCH DATA (kept)
# =========================================
def process_branch_data(excel_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    combined: List[pd.DataFrame] = []
    mapping = {
        "Date": "Date",
        "Day": "Day",
        "Sales Target": "SalesTarget",
        "SalesTarget": "SalesTarget",
        "Sales Achivement": "SalesActual",
        "Sales Achievement": "SalesActual",
        "SalesActual": "SalesActual",
        "Sales %": "SalesPercent",
        " Sales %": "SalesPercent",
        "SalesPercent": "SalesPercent",
        "NOB Target": "NOBTarget",
        "NOBTarget": "NOBTarget",
        "NOB Achievemnet": "NOBActual",
        "NOB Achievement": "NOBActual",
        "NOBActual": "NOBActual",
        "NOB %": "NOBPercent",
        " NOB %": "NOBPercent",
        "NOBPercent": "NOBPercent",
        "ABV Target": "ABVTarget",
        "ABVTarget": "ABVTarget",
        "ABV Achievement": "ABVActual",
        " ABV Achievement": "ABVActual",
        "ABVActual": "ABVActual",
        "ABV %": "ABVPercent",
        "ABVPercent": "ABVPercent",
    }
    for sheet_name, df in excel_data.items():
        if sheet_name.strip().lower() == "liquidity":
            continue
        if sheet_name not in config.BRANCHES:
            continue
        if df.empty:
            continue

        d = df.copy()
        d.columns = d.columns.astype(str).str.strip()
        for old, new in mapping.items():
            if old in d.columns and new not in d.columns:
                d.rename(columns={old: new}, inplace=True)
        if "Date" not in d.columns:
            maybe = [c for c in d.columns if "date" in c.lower()]
            if maybe:
                d.rename(columns={maybe[0]: "Date"}, inplace=True)

        d["Branch"] = sheet_name
        meta = config.BRANCHES.get(sheet_name, {})
        d["BranchName"] = meta.get("name", sheet_name)
        d["BranchCode"] = meta.get("code", "000")

        if "Date" in d.columns:
            d["Date"] = pd.to_datetime(d["Date"], errors="coerce")

        for col in [
            "SalesTarget",
            "SalesActual",
            "SalesPercent",
            "NOBTarget",
            "NOBActual",
            "NOBPercent",
            "ABVTarget",
            "ABVActual",
            "ABVPercent",
        ]:
            if col in d.columns:
                d[col] = _parse_numeric(d[col])

        combined.append(d)

    return pd.concat(combined, ignore_index=True) if combined else pd.DataFrame()


# =========================================
# LIQUIDITY (wide → long for your layout)
# =========================================
def _first_two_rows_look_like_headers(df: pd.DataFrame) -> bool:
    if df.shape[0] < 2:
        return False
    r0 = df.iloc[0].tolist()
    r1 = df.iloc[1].tolist()
    # heuristic: many non-empty strings in first two rows
    def count_str(xs): return sum(isinstance(x, str) and x.strip() != "" for x in xs)
    return count_str(r0) >= max(1, int(0.4 * len(r0))) and count_str(r1) >= max(1, int(0.4 * len(r1)))


def _clean_branch_from_header(h: str) -> str:
    s = str(h).strip()
    s = re.sub(r"\b(total\s*liquidity|change\s*in\s*liquid|%?\s*of\s*change)\b", "", s, flags=re.I)
    s = s.replace("|", " ").replace("  ", " ").strip(" -|")
    return s.strip()


def process_liquidity_wide(excel_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Parse sheet that looks like:
      DATE | [CASH, BANK, Total Liquidity, Change in Liquid, % of Change] x N branches
    with branch name appearing in a merged header row above the 5 columns.
    """
    # find liquidity-like tab
    liq_key = None
    for k in excel_data.keys():
        if "liquid" in k.lower() or "cash" in k.lower():
            liq_key = k
            break
    if not liq_key:
        return pd.DataFrame()

    raw = excel_data[liq_key].copy().dropna(how="all")
    raw.columns = raw.columns.astype(str)

    # If first two rows look like headers, combine them to form single-row headers
    if _first_two_rows_look_like_headers(raw):
        top = raw.iloc[0].astype(str).replace("nan", "").tolist()
        sub = raw.iloc[1].astype(str).replace("nan", "").tolist()
        new_cols = []
        for a, b in zip(top, sub):
            a = a.strip()
            b = b.strip()
            if a and b:
                new_cols.append(f"{a} | {b}")
            elif a:
                new_cols.append(a)
            elif b:
                new_cols.append(b)
            else:
                new_cols.append("")
        raw = raw.iloc[2:].reset_index(drop=True)
        raw.columns = [c if c else f"col_{i}" for i, c in enumerate(new_cols)]
    else:
        # otherwise keep existing single header row
        pass

    # Find DATE column
    date_candidates = [c for c in raw.columns if re.search(r"\bdate\b", str(c), re.I)]
    if not date_candidates:
        return pd.DataFrame()
    date_col = date_candidates[0]

    # Everything after DATE should be 5-col blocks per branch
    cols = raw.columns.tolist()
    try:
        start_idx = cols.index(date_col) + 1
    except ValueError:
        return pd.DataFrame()

    # Build blocks of size 5
    block_size = 5
    blocks: List[List[str]] = []
    for i in range(start_idx, len(cols), block_size):
        block = cols[i:i + block_size]
        if len(block) == block_size:
            blocks.append(block)

    if not blocks:
        return pd.DataFrame()

    out_rows: List[pd.DataFrame] = []
    for block in blocks:
        # Attempt to infer branch name from any of the headers (prefer the 'Total Liquidity' col)
        total_col = None
        for c in block:
            if re.search(r"total\s*liq", c, re.I):
                total_col = c
                break
        branch_name_source = total_col or block[0]
        branch_name = _clean_branch_from_header(branch_name_source)
        if not branch_name:
            # try to salvage by removing subheader tokens from all five and pick the longest
            branch_name = _clean_branch_from_header(max(block, key=lambda s: len(str(s))))

        sub = raw[[date_col] + block].copy()
        # Normalize block column names to Cash/Bank/Total/Change/Pct
        sub.columns = ["Date", "Cash", "Bank", "Total", "Change", "PctChange"]
        sub["Branch"] = branch_name
        out_rows.append(sub)

    df = pd.concat(out_rows, ignore_index=True)

    # Coerce types
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ["Cash", "Bank", "Total", "Change"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
    if "PctChange" in df.columns:
        df["PctChange"] = pd.to_numeric(df["PctChange"].astype(str).str.replace("%", "", regex=False), errors="coerce")

    df = df.dropna(subset=["Date"])
    df = df.sort_values(["Date", "Branch"]).reset_index(drop=True)
    return df


def liquidity_kpis(df: pd.DataFrame, start_date: datetime.date, end_date: datetime.date) -> Dict[str, Any]:
    if df.empty:
        return {}
    f = df[(df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)].copy()
    if f.empty:
        return {}

    k: Dict[str, Any] = {}
    # Sum across branches by day (Total = Cash + Bank OR use provided Total if present)
    if "Total" in f:
        totals = f.groupby("Date", as_index=False)["Total"].sum()
    else:
        temp = f.copy()
        temp["Total"] = temp.get("Cash", 0).fillna(0) + temp.get("Bank", 0).fillna(0)
        totals = temp.groupby("Date", as_index=False)["Total"].sum()

    k["latest_date"] = totals["Date"].max().date()
    k["latest_total"] = float(totals.loc[totals["Date"] == totals["Date"].max(), "Total"].sum())

    # Flows (if present)
    k["total_change"] = float(f.get("Change", pd.Series(dtype=float)).sum()) if "Change" in f else 0.0

    # Per-branch snapshot (latest)
    last = f["Date"].max()
    snap = f[f["Date"] == last].copy()
    if "Total" not in snap.columns:
        snap["Total"] = snap.get("Cash", 0).fillna(0) + snap.get("Bank", 0).fillna(0)
    k["snapshot"] = snap.groupby("Branch", as_index=True)["Total"].sum().sort_values(ascending=False)

    return k


def liquidity_charts(df: pd.DataFrame, start_date: datetime.date, end_date: datetime.date) -> Dict[str, go.Figure]:
    figs: Dict[str, go.Figure] = {"total_trend": go.Figure(), "branch_trend": go.Figure()}
    if df.empty:
        return figs

    f = df[(df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)].copy()
    if f.empty:
        return figs

    # Total liquidity trend (sum all branches per day)
    g = f.copy()
    if "Total" not in g.columns:
        g["Total"] = g.get("Cash", 0).fillna(0) + g.get("Bank", 0).fillna(0)
    daily = g.groupby("Date", as_index=False)["Total"].sum()

    fig_total = go.Figure()
    fig_total.add_trace(
        go.Scatter(
            x=daily["Date"], y=daily["Total"], mode="lines+markers", name="Total Liquidity",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Total: %{y:,.0f}<extra></extra>", line=dict(width=3)
        )
    )
    fig_total.update_layout(
        title="Total Liquidity (All Branches)",
        height=420, showlegend=True, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        xaxis_title="Date", yaxis_title="SAR"
    )

    # Per-branch trend (Total)
    fig_branch = go.Figure()
    for br in sorted(f["Branch"].dropna().unique()):
        d = f[f["Branch"] == br].copy()
        if "Total" not in d.columns:
            d["Total"] = d.get("Cash", 0).fillna(0) + d.get("Bank", 0).fillna(0)
        fig_branch.add_trace(
            go.Scatter(
                x=d["Date"], y=d["Total"], mode="lines+markers", name=br,
                hovertemplate=f"{br}<br>%{{x|%Y-%m-%d}} • %{y:,.0f}<extra></extra>", line=dict(width=2)
            )
        )
    fig_branch.update_layout(
        title="Liquidity by Branch",
        height=420, showlegend=True, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        xaxis_title="Date", yaxis_title="SAR"
    )

    figs["total_trend"] = fig_total
    figs["branch_trend"] = fig_branch
    return figs


def liquidity_tables(df: pd.DataFrame, start_date: datetime.date, end_date: datetime.date) -> Dict[str, pd.DataFrame]:
    tbls: Dict[str, pd.DataFrame] = {}
    if df.empty:
        return tbls
    f = df[(df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)].copy()
    if f.empty:
        return tbls

    dd = f.copy()
    if "Total" not in dd.columns:
        dd["Total"] = dd.get("Cash", 0).fillna(0) + dd.get("Bank", 0).fillna(0)

    tbls["daily_total"] = dd.groupby("Date", as_index=False)["Total"].sum().sort_values("Date")
    tbls["latest_by_branch"] = (
        dd[dd["Date"] == dd["Date"].max()].groupby("Branch", as_index=True)["Total"].sum().sort_values(ascending=False)
    ).to_frame("Total")
    return tbls


# =========================================
# CHARTS (sales/NOB/ABV) — unchanged
# =========================================
def _metric_area(df: pd.DataFrame, y_col: str, title: str, *, show_target: bool = True) -> go.Figure:
    if df.empty or "Date" not in df.columns or df["Date"].isna().all():
        return go.Figure()

    agg_func = "sum" if y_col != "ABVActual" else "mean"
    daily_actual = df.groupby(["Date", "BranchName"]).agg({y_col: agg_func}).reset_index()
    target_col_map = {"SalesActual": "SalesTarget", "NOBActual": "NOBTarget", "ABVActual": "ABVTarget"}
    target_col = target_col_map.get(y_col)

    daily_target: Optional[pd.DataFrame] = None
    if show_target and target_col and target_col in df.columns:
        agg_func_target = "sum" if y_col != "ABVActual" else "mean"
        daily_target = df.groupby(["Date", "BranchName"]).agg({target_col: agg_func_target}).reset_index()

    fig = go.Figure()
    palette = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444", "#14b8a6"]
    for i, br in enumerate(sorted(daily_actual["BranchName"].unique())):
        d_actual = daily_actual[daily_actual["BranchName"] == br]
        color = palette[i % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=d_actual["Date"],
                y=d_actual[y_col],
                name=f"{br} - Actual",
                mode="lines+markers",
                line=dict(width=3, color=color),
                fill="tozeroy",
                hovertemplate=f"<b>{br}</b><br>Date: %{{x|%Y-%m-%d}}<br>Actual: %{{y:,.0f}}<extra></extra>",
            )
        )
        if daily_target is not None:
            d_target = daily_target[daily_target["BranchName"] == br]
            fig.add_trace(
                go.Scatter(
                    x=d_target["Date"],
                    y=d_target[target_col],
                    name=f"{br} - Target",
                    mode="lines",
                    line=dict(width=2, color=color, dash="dash"),
                    fill=None,
                    hovertemplate=f"<b>{br}</b><br>Date: %{{x|%Y-%m-%d}}<br>Target: %{{y:,.0f}}<extra></extra>",
                    showlegend=False if len(sorted(daily_actual["BranchName"].unique())) > 1 else True,
                )
            )
    fig.update_layout(
        title=title,
        height=420,
        showlegend=True,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        xaxis_title="Date",
        yaxis_title="Value",
    )
    return fig


def _branch_comparison_chart(bp: pd.DataFrame) -> go.Figure:
    if bp.empty:
        return go.Figure()
    metrics = {"SalesPercent": "Sales %", "NOBPercent": "NOB %", "ABVPercent": "ABV %"}
    fig = go.Figure()
    x = bp.index.tolist()
    palette = ["#3b82f6", "#10b981", "#f59e0b"]
    for i, (col, label) in enumerate(metrics.items()):
        y = bp[col].tolist()
        fig.add_trace(go.Bar(x=x, y=y, name=label, marker=dict(color=palette[i % len(palette)])))
    fig.update_layout(
        barmode="group",
        title="Branch Performance Comparison (%)",
        xaxis_title="Branch",
        yaxis_title="Percent",
        height=400,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    )
    return fig


# =========================================
# RENDER HELPERS — unchanged
# =========================================
def branch_status_class(pct: float) -> str:
    if pct >= 95:
        return "excellent"
    if pct >= 85:
        return "good"
    if pct >= 75:
        return "warn"
    return "danger"


def _branch_color_by_name(name: str) -> str:
    for _, meta in config.BRANCHES.items():
        if meta.get("name") == name:
            return meta.get("color", "#6b7280")
    return "#6b7280"


def render_branch_cards(bp: pd.DataFrame):
    st.markdown("### 🏪 Branch Overview")
    if bp.empty:
        st.info("No branch summary available.")
        return
    cols_per_row = 3
    items = list(bp.index)
    for i in range(0, len(items), cols_per_row):
        row = st.columns(cols_per_row, gap="medium")
        for j, br in enumerate(items[i:i+cols_per_row]):
            with row[j]:
                r = bp.loc[br]
                avg_pct = float(np.nanmean([r.get("SalesPercent", 0), r.get("NOBPercent", 0), r.get("ABVPercent", 0)]) or 0)
                cls = branch_status_class(avg_pct)
                color = _branch_color_by_name(br)
                st.markdown(
                    f"""
                    <div class="card" style="background: linear-gradient(180deg, {color}1a 0%, #ffffff 40%);">
                      <div style="display:flex;justify-content:space-between;align-items:center">
                        <div style="font-weight:800">{br}</div>
                        <span class="pill {cls}">{avg_pct:.1f}%</span>
                      </div>
                      <div style="margin-top:8px;font-size:.92rem;color:#6b7280">Overall score</div>
                      <div style="display:flex;gap:12px;margin-top:10px;flex-wrap:wrap">
                        <div><div style="font-size:.8rem;color:#64748b">Sales %</div><div style="font-weight:700">{r.get('SalesPercent',0):.1f}%</div></div>
                        <div><div style="font-size:.8rem;color:#64748b">NOB %</div><div style="font-weight:700">{r.get('NOBPercent',0):.1f}%</div></div>
                        <div><div style="font-size:.8rem;color:#64748b">ABV %</div><div style="font-weight:700">{r.get('ABVPercent',0):.1f}%</div></div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_overview(df: pd.DataFrame, k: Dict[str, Any]):
    st.markdown("### 🏆 Overall Performance")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💰 Total Sales", f"SAR {k.get('total_sales_actual',0):,.0f}", delta=f"vs Target: {k.get('overall_sales_percent',0):.1f}%")
    variance_color = "normal" if k.get("total_sales_variance", 0) >= 0 else "inverse"
    c2.metric("📊 Sales Variance", f"SAR {k.get('total_sales_variance',0):,.0f}", delta=f"{k.get('overall_sales_percent',0)-100:+.1f}%", delta_color=variance_color)
    nob_color = "normal" if k.get("overall_nob_percent", 0) >= config.TARGETS['nob_achievement'] else "inverse"
    c3.metric("🛍️ Total Baskets", f"{k.get('total_nob_actual',0):,.0f}", delta=f"Achievement: {k.get('overall_nob_percent',0):.1f}%", delta_color=nob_color)
    c4.metric("💎 Avg Basket Value", f"SAR {k.get('avg_abv_actual',0):,.2f}", delta=f"vs Target: {k.get('overall_abv_percent',0):.1f}%")
    score_color = "normal" if k.get("performance_score", 0) >= 80 else "off"
    c5.metric("⭐ Performance Score", f"{k.get('performance_score',0):.0f}/100", delta="Weighted Score", delta_color=score_color)

    if "branch_performance" in k and not k["branch_performance"].empty:
        render_branch_cards(k["branch_performance"])

    if "branch_performance" in k and not k["branch_performance"].empty:
        st.markdown("### 📊 Comparison Table")
        bp = k["branch_performance"].copy()
        df_table = (
            bp[["SalesPercent", "NOBPercent", "ABVPercent", "SalesActual", "SalesTarget"]]
            .rename(
                columns={
                    "SalesPercent": "Sales %",
                    "NOBPercent": "NOB %",
                    "ABVPercent": "ABV %",
                    "SalesActual": "Sales (Actual)",
                    "SalesTarget": "Sales (Target)",
                }
            )
            .round(1)
        )
        st.dataframe(
            df_table.style.format(
                {"Sales %": "{:,.1f}", "NOB %": "{:,.1f}", "ABV %": "{:,.1f}", "Sales (Actual)": "{:,.0f}", "Sales (Target)": "{:,.0f}"}
            ),
            use_container_width=True,
        )

        st.markdown("### 📉 Branch Performance Comparison")
        fig_cmp = _branch_comparison_chart(bp)
        st.plotly_chart(fig_cmp, use_container_width=True, config={"displayModeBar": False})


# -------- Branch filter via toggle buttons (kept)
def render_branch_filter_buttons(df: pd.DataFrame, key_prefix: str = "br_") -> List[str]:
    if "BranchName" not in df.columns:
        return []

    all_branches = sorted([b for b in df["BranchName"].dropna().unique()])
    if "selected_branches" not in st.session_state:
        st.session_state.selected_branches = all_branches.copy()

    st.markdown("### 🏪 Branches")
    col_a, col_b = st.columns([1, 6])
    with col_a:
        c1, c2 = st.columns(2)
        if c1.button("Select all"):
            st.session_state.selected_branches = all_branches.copy()
            st.rerun()
        if c2.button("Clear"):
            st.session_state.selected_branches = []
            st.rerun()

    with col_b:
        btn_cols = st.columns(len(all_branches)) if all_branches else []
        for i, b in enumerate(all_branches):
            is_on = b in st.session_state.selected_branches
            label = ("✅ " if is_on else "➕ ") + b
            if btn_cols[i].button(label, key=f"{key_prefix}{i}"):
                if is_on:
                    st.session_state.selected_branches.remove(b)
                else:
                    st.session_state.selected_branches.append(b)
                st.rerun()

    return st.session_state.selected_branches


# =========================================
# MAIN
# =========================================
def main():
    apply_css()

    # Load data
    sheets_map = load_workbook_from_gsheet(config.DEFAULT_PUBLISHED_URL)
    if not sheets_map:
        st.warning("No non-empty sheets found.")
        st.stop()

    # Sales/NOB/ABV (existing)
    df_all = process_branch_data(sheets_map)
    if df_all.empty:
        st.error("Could not process branch data. Check branch sheet names/columns.")
        st.stop()

    # Liquidity (wide layout → long)
    liq_df = process_liquidity_wide(sheets_map)

    # Sidebar bounds
    all_branches = sorted(df_all["BranchName"].dropna().unique()) if "BranchName" in df_all else []
    if "selected_branches" not in st.session_state:
        st.session_state.selected_branches = list(all_branches)

    df_for_bounds = df_all[df_all["BranchName"].isin(st.session_state.selected_branches)].copy() if all_branches else df_all.copy()
    if "Date" in df_for_bounds.columns and df_for_bounds["Date"].notna().any():
        dmin_sb = df_for_bounds["Date"].min().date()
        dmax_sb = df_for_bounds["Date"].max().date()
    else:
        dmin_sb = dmax_sb = datetime.today().date()

    if "start_date" not in st.session_state:
        st.session_state.start_date = dmin_sb
    if "end_date" not in st.session_state:
        st.session_state.end_date = dmax_sb

    # -------- Sidebar --------
    with st.sidebar:
        st.markdown('<div class="sb-title">📊 <span>AL KHAIR DASHBOARD</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-subtle">Filters affect all tabs.</div>', unsafe_allow_html=True)

        st.markdown('<div class="sb-section">Actions</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        if cols[0].button("🔄 Refresh Data", use_container_width=True):
            load_workbook_from_gsheet.clear()
            st.session_state.pop("start_date", None)
            st.session_state.pop("end_date", None)
            st.rerun()
        if cols[1].button("🧹 Reset Filters", use_container_width=True):
            st.session_state.selected_branches = list(all_branches)
            st.session_state.start_date = dmin_sb
            st.session_state.end_date = dmax_sb
            st.rerun()

        st.markdown('<div class="sb-hr"></div>', unsafe_allow_html=True)

        st.markdown('<div class="sb-section">Quick Range</div>', unsafe_allow_html=True)
        preset = st.radio("", ["Last 7d", "Last 30d", "This Month", "YTD", "All Time"], index=1, key="sb_quick_range")
        today = dmax_sb
        if preset == "Last 7d":
            st.session_state.start_date, st.session_state.end_date = max(dmin_sb, today - timedelta(days=7)), today
        elif preset == "Last 30d":
            st.session_state.start_date, st.session_state.end_date = max(dmin_sb, today - timedelta(days=30)), today
        elif preset == "This Month":
            first = today.replace(day=1)
            st.session_state.start_date, st.session_state.end_date = max(dmin_sb, first), today
        elif preset == "YTD":
            first = today.replace(month=1, day=1)
            st.session_state.start_date, st.session_state.end_date = max(dmin_sb, first), today
        elif preset == "All Time":
            st.session_state.start_date, st.session_state.end_date = dmin_sb, dmax_sb

        st.markdown('<div class="sb-section">Custom Date</div>', unsafe_allow_html=True)
        _sd = st.date_input("Start:", value=st.session_state.start_date, min_value=dmin_sb, max_value=dmax_sb, key="sb_sd")
        _ed = st.date_input("End:",   value=st.session_state.end_date,   min_value=dmin_sb, max_value=dmax_sb, key="sb_ed")
        st.session_state.start_date = max(dmin_sb, min(_sd, dmax_sb))
        st.session_state.end_date   = max(dmin_sb, min(_ed, dmax_sb))
        if st.session_state.start_date > st.session_state.end_date:
            st.session_state.start_date, st.session_state.end_date = dmin_sb, dmax_sb

        st.markdown('<div class="sb-hr"></div>', unsafe_allow_html=True)

        # Snapshot
        df_snap = df_for_bounds.copy()
        if "Date" in df_snap.columns and df_snap["Date"].notna().any():
            mask_snap = (df_snap["Date"].dt.date >= st.session_state.start_date) & (df_snap["Date"].dt.date <= st.session_state.end_date)
            df_snap = df_snap.loc[mask_snap]
        total_sales_actual = float(df_snap.get("SalesActual", pd.Series(dtype=float)).sum()) if not df_snap.empty else 0.0
        total_sales_target = float(df_snap.get("SalesTarget", pd.Series(dtype=float)).sum()) if not df_snap.empty else 0.0
        overall_sales_percent = (total_sales_actual / total_sales_target * 100) if total_sales_target > 0 else 0.0
        total_nob_actual = float(df_snap.get("NOBActual", pd.Series(dtype=float)).sum()) if not df_snap.empty else 0.0

        st.markdown('<div class="sb-section">Today\'s Snapshot</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="sb-card">
              <div>💰 <b>Sales:</b> SAR {total_sales_actual:,.0f}</div>
              <div>📊 <b>Variance:</b> {overall_sales_percent-100:+.1f}%</div>
              <div>🛍️ <b>Baskets:</b> {total_nob_actual:,.0f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sb-foot">UI preview • v2.0</div>', unsafe_allow_html=True)

    # ------- Main header -------
    st.markdown(f"<div class='title'>📊 {config.PAGE_TITLE}</div>", unsafe_allow_html=True)
    date_span = ""
    if "Date" in df_all.columns and df_all["Date"].notna().any():
        date_span = f"{df_all['Date'].min().date()} → {df_all['Date'].max().date()}"
    st.markdown(
        f"<div class='subtitle'>Branches: {df_all['BranchName'].nunique() if 'BranchName' in df_all else 0} • Rows: {len(df_all):,} {('• ' + date_span) if date_span else ''}</div>",
        unsafe_allow_html=True,
    )

    # ===== Mobile filter drawer =====
    st.markdown('<div class="mobile-only">', unsafe_allow_html=True)
    with st.expander("📱 Filters", expanded=False):
        preset_m = st.radio(
            "Quick range", ["Last 7d", "Last 30d", "This Month", "YTD", "All Time"],
            index=1, horizontal=True, key="m_quick_range",
        )
        dmin_all = df_all["Date"].min().date() if "Date" in df_all else datetime.today().date()
        dmax_all = df_all["Date"].max().date() if "Date" in df_all else datetime.today().date()
        today_all = dmax_all
        if preset_m == "Last 7d":
            st.session_state.start_date, st.session_state.end_date = max(dmin_all, today_all - timedelta(days=7)), today_all
        elif preset_m == "Last 30d":
            st.session_state.start_date, st.session_state.end_date = max(dmin_all, today_all - timedelta(days=30)), today_all
        elif preset_m == "This Month":
            first = today_all.replace(day=1)
            st.session_state.start_date, st.session_state.end_date = max(dmin_all, first), today_all
        elif preset_m == "YTD":
            first = today_all.replace(month=1, day=1)
            st.session_state.start_date, st.session_state.end_date = max(dmin_all, first), today_all
        elif preset_m == "All Time":
            st.session_state.start_date, st.session_state.end_date = dmin_all, dmax_all

        _ms = st.date_input("Start date", value=st.session_state.start_date, min_value=dmin_all, max_value=dmax_all, key="m_start")
        _me = st.date_input("End date",   value=st.session_state.end_date,   min_value=dmin_all, max_value=dmax_all, key="m_end")
        st.session_state.start_date = max(dmin_all, min(_ms, dmax_all))
        st.session_state.end_date   = max(dmin_all, min(_me, dmax_all))
        if st.session_state.start_date > st.session_state.end_date:
            st.session_state.start_date, st.session_state.end_date = dmin_all, dmax_all
    st.markdown('</div>', unsafe_allow_html=True)

    # Apply branch filter (existing)
    df = df_all.copy()
    if "BranchName" in df.columns:
        prev_sel = tuple(st.session_state.get("selected_branches", []))
        selected = render_branch_filter_buttons(df)
        if selected:
            df = df[df["BranchName"].isin(selected)].copy()
        if tuple(selected) != prev_sel:
            st.session_state.pop("start_date", None)
            st.session_state.pop("end_date", None)
        if df.empty:
            st.warning("No rows after branch filter.")
            st.stop()

    # --- Date filter (existing) ---
    if "Date" in df.columns and df["Date"].notna().any():
        dmin = df["Date"].min().date()
        dmax = df["Date"].max().date()

        if "start_date" not in st.session_state:
            st.session_state.start_date = dmin
        if "end_date" not in st.session_state:
            st.session_state.end_date = dmax

        s = st.session_state.start_date
        e = st.session_state.end_date
        if s < dmin or s > dmax: s = dmin
        if e > dmax or e < dmin: e = dmax
        if s > e: s, e = dmin, dmax
        st.session_state.start_date, st.session_state.end_date = s, e

        c1, c2, _ = st.columns([2, 2, 6])
        with c1:
            start_d = st.date_input("Start Date", value=st.session_state.start_date, min_value=dmin, max_value=dmax, key="date_start")
        with c2:
            end_d = st.date_input("End Date", value=st.session_state.end_date, min_value=dmin, max_value=dmax, key="date_end")

        st.session_state.start_date = max(dmin, min(start_d, dmax))
        st.session_state.end_date = max(dmin, min(end_d, dmax))
        if st.session_state.start_date > st.session_state.end_date:
            st.session_state.start_date, st.session_state.end_date = dmin, dmax

        mask = (df["Date"].dt.date >= st.session_state.start_date) & (df["Date"].dt.date <= st.session_state.end_date)
        df = df.loc[mask].copy()
        if df.empty:
            st.warning("No rows in selected date range.")
            st.stop()

    # KPIs + Quick insights (existing)
    def calc_kpis(df_in: pd.DataFrame) -> Dict[str, Any]:
        if df_in.empty:
            return {}
        k: Dict[str, Any] = {}
        k["total_sales_target"] = float(df_in.get("SalesTarget", pd.Series(dtype=float)).sum())
        k["total_sales_actual"] = float(df_in.get("SalesActual", pd.Series(dtype=float)).sum())
        k["total_sales_variance"] = k["total_sales_actual"] - k["total_sales_target"]
        k["overall_sales_percent"] = (k["total_sales_actual"] / k["total_sales_target"] * 100) if k["total_sales_target"] > 0 else 0.0
        k["total_nob_target"] = float(df_in.get("NOBTarget", pd.Series(dtype=float)).sum())
        k["total_nob_actual"] = float(df_in.get("NOBActual", pd.Series(dtype=float)).sum())
        k["overall_nob_percent"] = (k["total_nob_actual"] / k["total_nob_target"] * 100) if k["total_nob_target"] > 0 else 0.0
        k["avg_abv_target"] = float(df_in.get("ABVTarget", pd.Series(dtype=float)).mean()) if "ABVTarget" in df_in else 0.0
        k["avg_abv_actual"] = float(df_in.get("ABVActual", pd.Series(dtype=float)).mean()) if "ABVActual" in df_in else 0.0
        k["overall_abv_percent"] = (k["avg_abv_actual"] / k["avg_abv_target"] * 100) if "ABVTarget" in df_in and k["avg_abv_target"] > 0 else 0.0
        if "BranchName" in df_in.columns:
            k["branch_performance"] = (
                df_in.groupby("BranchName")
                .agg(
                    {
                        "SalesTarget": "sum",
                        "SalesActual": "sum",
                        "SalesPercent": "mean",
                        "NOBTarget": "sum",
                        "NOBActual": "sum",
                        "NOBPercent": "mean",
                        "ABVTarget": "mean",
                        "ABVActual": "mean",
                        "ABVPercent": "mean",
                    }
                ).round(2)
            )
        if "Date" in df_in.columns and df_in["Date"].notna().any():
            k["date_range"] = {"start": df_in["Date"].min(), "end": df_in["Date"].max(), "days": int(df_in["Date"].dt.date.nunique())}
        score = w = 0.0
        if k.get("overall_sales_percent", 0) > 0:
            score += min(k["overall_sales_percent"] / 100, 1.2) * 40; w += 40
        if k.get("overall_nob_percent", 0) > 0:
            score += min(k["overall_nob_percent"] / 100, 1.2) * 35; w += 35
        if k.get("overall_abv_percent", 0) > 0:
            score += min(k["overall_abv_percent"] / 100, 1.2) * 25; w += 25
        k["performance_score"] = (score / w * 100) if w > 0 else 0.0
        return k

    k = calc_kpis(df)
    with st.expander("⚡ Quick Insights", expanded=True):
        insights: List[str] = []
        if "branch_performance" in k and not k["branch_performance"].empty:
            bp = k["branch_performance"]
            insights.append(f"🥇 Best Sales %: {bp['SalesPercent'].idxmax()} — {bp['SalesPercent'].max():.1f}%")
            insights.append(f"🔻 Lowest Sales %: {bp['SalesPercent'].idxmin()} — {bp['SalesPercent'].min():.1f}%")
            below = bp[bp["SalesPercent"] < config.TARGETS["sales_achievement"]].index.tolist()
            if below:
                insights.append("⚠️ Below 95% target: " + ", ".join(below))
        if k.get("total_sales_variance", 0) < 0:
            insights.append(f"🟥 Overall variance negative by SAR {abs(k['total_sales_variance']):,.0f}")
        if insights:
            for it in insights:
                st.markdown("- " + it)
        else:
            st.write("All metrics look healthy for the current selection.")

    # ===== Tabs (Liquidity always visible) =====
    t1, t2, t3, t4 = st.tabs(["🏠 Branch Overview", "📈 Daily Trends", "📥 Export", "💧 Liquidity"])

    with t1:
        render_overview(df, k)

    with t2:
        st.markdown("#### Choose Metric")
        mtab1, mtab2, mtab3 = st.tabs(["💰 Sales", "🛍️ NOB", "💎 ABV"])
        if "Date" in df.columns and df["Date"].notna().any():
            opts = ["Last 7 Days", "Last 30 Days", "Last 3 Months", "All Time"]
            choice = st.selectbox("Time Period", opts, index=1, key="trend_window")
            today = (df["Date"].max() if df["Date"].notna().any() else pd.Timestamp.today()).date()
            start = {"Last 7 Days": today - timedelta(days=7),
                     "Last 30 Days": today - timedelta(days=30),
                     "Last 3 Months": today - timedelta(days=90)}.get(choice, df["Date"].min().date())
            f = df[(df["Date"].dt.date >= start) & (df["Date"].dt.date <= today)].copy()
        else:
            f = df.copy()

        with mtab1:
            st.plotly_chart(_metric_area(f, "SalesActual", "Daily Sales (Actual vs Target)"),
                            use_container_width=True, config={"displayModeBar": False})
        with mtab2:
            st.plotly_chart(_metric_area(f, "NOBActual", "Number of Baskets (Actual vs Target)"),
                            use_container_width=True, config={"displayModeBar": False})
        with mtab3:
            st.plotly_chart(_metric_area(f, "ABVActual", "Average Basket Value (Actual vs Target)"),
                            use_container_width=True, config={"displayModeBar": False})

    with t3:
        if df.empty:
            st.info("No data to export.")
        else:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                df.to_excel(writer, sheet_name="All Branches", index=False)
                if "branch_performance" in k:
                    k["branch_performance"].to_excel(writer, sheet_name="Branch Summary")
                if not liq_df.empty:
                    liq_df.to_excel(writer, sheet_name="Liquidity", index=False)
            st.download_button(
                "📊 Download Excel Report",
                buf.getvalue(),
                f"Alkhair_Branch_Analytics_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
            st.download_button(
                "📄 Download CSV",
                df.to_csv(index=False).encode("utf-8"),
                f"Alkhair_Branch_Data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with t4:
        st.markdown("### 💧 Liquidity Overview")
        if liq_df.empty:
            st.info("No Liquidity data parsed. Ensure the 'Liquidity' tab uses the wide layout and the sheet is public, then click 🔄 Refresh Data.")
        else:
            lk = liquidity_kpis(liq_df, st.session_state.start_date, st.session_state.end_date)
            c1, c2, c3 = st.columns(3)
            c1.metric("💼 Latest Total", f"SAR {lk.get('latest_total',0):,.0f}", f"as of {lk.get('latest_date')}")
            c2.metric("🔄 Sum of Change", f"SAR {lk.get('total_change',0):,.0f}")
            if isinstance(lk.get("snapshot"), pd.Series) and not lk["snapshot"].empty:
                top_branch = lk["snapshot"].idxmax()
                c3.metric("🏦 Top Branch (latest)", top_branch, f"SAR {lk['snapshot'].max():,.0f}")

            st.markdown("#### Charts")
            figs = liquidity_charts(liq_df, st.session_state.start_date, st.session_state.end_date)
            st.plotly_chart(figs["total_trend"], use_container_width=True, config={"displayModeBar": False})
            st.plotly_chart(figs["branch_trend"], use_container_width=True, config={"displayModeBar": False})

            st.markdown("#### Tables")
            tbls = liquidity_tables(liq_df, st.session_state.start_date, st.session_state.end_date)
            if "daily_total" in tbls:
                st.dataframe(tbls["daily_total"].style.format({"Total": "{:,.0f}"}), use_container_width=True)
            if "latest_by_branch" in tbls:
                st.markdown("**Latest Snapshot by Branch**")
                st.dataframe(tbls["latest_by_branch"].style.format({"Total": "{:,.0f}"}), use_container_width=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("❌ Application Error. Please adjust filters or refresh.")
        st.exception(e)
