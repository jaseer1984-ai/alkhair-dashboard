# app.py — Alkhair Family Market Daily Dashboard (Power BI–style cards)

import io, re
from datetime import datetime
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Alkhair Family Market — Daily Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Your published Google Sheets CSV (keep as published link with output=csv)
GSHEET_URL = (
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vSW_ui1m_393ipZv8NAliu1rly6zeifFxMfOWpQF17hjVIDa9Ll8PiGCaz8gTRkMQ/pub?output=csv"
)

# Auto-refresh interval (minutes)
REFRESH_MINUTES = 1
REFRESH_SECONDS = REFRESH_MINUTES * 60

# Hide sidebar & auto refresh (very lightweight)
st.markdown("<style>[data-testid='stSidebar']{display:none!important}</style>", unsafe_allow_html=True)
components.html(f"<meta http-equiv='refresh' content='{REFRESH_SECONDS}'>", height=0)

# =========================
# THEME / CSS (Power BI–ish)
# =========================
st.markdown("""
<style>
:root{
  --bg:#0b1220;           /* app background */
  --card:#111a2b;         /* card background */
  --ink:#eaf0ff;          /* text */
  --ink-weak:#b9c2d6;
  --accent:#4f8cff;       /* brand blue */
  --accent-2:#19c37d;     /* green */
  --shadow:0 10px 24px rgba(0,0,0,.35);
  --radius:18px;
}
html, body, [data-testid="stAppViewContainer"]{
  background: var(--bg);
  color: var(--ink);
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial, "Noto Sans";
}
h1,h2,h3{ color: var(--ink); }
hr{ border-color:#1f2a44; }
.card{
  background: var(--card);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 18px 18px 14px 18px;
  border: 1px solid rgba(255,255,255,.06);
}
.card-header{
  display:flex; align-items:center; gap:.6rem; margin-bottom:.5rem;
  font-weight:700; color: var(--ink);
}
.kpi-grid{ display:grid; grid-template-columns: repeat(4, 1fr); gap:14px; }
.kpi{
  background: linear-gradient(180deg, rgba(79,140,255,.18), rgba(79,140,255,.05));
  border:1px solid rgba(255,255,255,.08);
  border-radius: 16px; padding: 16px; box-shadow: var(--shadow);
}
.kpi .label{ color: var(--ink-weak); font-size:.9rem; }
.kpi .value{ font-size:1.6rem; font-weight:800; margin-top:.2rem; color: var(--ink); }
.kpi .icon{ font-size:1.2rem; margin-right:.4rem; }
.brandbar{
  position:sticky; top:0; z-index:999; padding: 10px 0 8px 0; margin-bottom:12px;
  background: linear-gradient(90deg, rgba(79,140,255,.18), rgba(25,195,125,.18));
  border-bottom:1px solid rgba(255,255,255,.07);
  backdrop-filter: blur(6px);
}
.btn-refresh button{
  width:100%; border-radius:12px; border:1px solid rgba(255,255,255,.15);
  background: linear-gradient(180deg,rgba(25,195,125,.25), rgba(25,195,125,.08));
}
.small{ font-size:.85rem; color:var(--ink-weak); }
.dataframe tbody tr{ background: transparent !important; }
[data-testid="stMetricValue"]{ color: var(--ink); }
[data-testid="stMetricDelta"]{ color: var(--accent-2); }
</style>
""", unsafe_allow_html=True)

# =========================
# ALIASES & HELPERS
# =========================
DATE_ALIASES    = ["date", "bill_date", "txn_date", "invoice_date", "posting_date"]
STORE_ALIASES   = ["store", "branch", "location", "shop"]
SALES_ALIASES   = ["sales", "amount", "net_sales", "revenue", "netamount", "net_amount"]
COGS_ALIASES    = ["cogs", "cost", "purchase_cost", "cost_of_goods"]
QTY_ALIASES     = ["qty", "quantity", "qty_sold", "quantity_sold", "units", "pcs", "pieces", "units_sold", "qnty", "qnt", "qty."]
TICKETS_ALIASES = ["tickets", "bills", "invoice_count", "transactions", "bills_count"]
CAT_ALIASES     = ["category", "cat"]
ITEM_ALIASES    = ["item", "product", "sku", "item_name"]
INVQ_ALIASES    = ["inventoryqty", "inventory_qty", "stock_qty", "stockqty", "stock_quantity"]
INVV_ALIASES    = ["inventoryvalue", "inventory_value", "stock_value", "stockvalue"]

def _first_exact_or_fuzzy(df: pd.DataFrame, aliases: List[str], fuzzy_tokens: Optional[List[str]]=None):
    low = {c.lower().strip(): c for c in df.columns}
    for a in aliases:
        if a in low: return low[a]
    tokens = (fuzzy_tokens or []) + aliases
    for k, orig in low.items():
        for t in tokens:
            if t in k: return orig
    return None

def _gsheet_to_csv_url(url: str) -> str:
    if "docs.google.com/spreadsheets" not in url: return url
    if "/spreadsheets/d/e/" in url and "output=csv" in url: return url
    if "/export" in url and "format=csv" in url: return url
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    if not m: return url
    sheet_id = m.group(1)
    gid = re.search(r"gid=(\d+)", url).group(1) if "gid=" in url else "0"
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

_num_cleanup_re = re.compile(r"[^\d\-\.\,()]")
def _to_num(x):
    if isinstance(x, (int, float, np.number)): return float(x)
    if x is None: return np.nan
    s = str(x).strip()
    if s == "": return np.nan
    s = _num_cleanup_re.sub("", s)
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg=True; s=s[1:-1]
    s = s.replace(",", "")
    try:
        v=float(s); return -v if neg else v
    except: return np.nan

@st.cache_data(ttl=0, show_spinner=False)
def load_data(url: str):
    return pd.read_csv(_gsheet_to_csv_url(url.strip()))

def standardize(df: pd.DataFrame):
    if df.empty: 
        return df.copy(), {}
    mapping = {}
    df = df.copy()

    def map_col(aliases, new, to_num=False, fill0=False, fuzzy_extra=None):
        col = _first_exact_or_fuzzy(df, aliases, fuzzy_extra)
        if col:
            mapping[new] = col
            df.rename(columns={col:new}, inplace=True)
            if to_num:
                df[new] = df[new].map(_to_num)
                if fill0: df[new] = df[new].fillna(0)
        else:
            mapping[new] = "(missing)"
            df[new] = 0 if fill0 else np.nan

    # Date (dayfirst)
    cd = _first_exact_or_fuzzy(df, DATE_ALIASES, ["date"])
    if cd:
        mapping["Date"]=cd; df.rename(columns={cd:"Date"}, inplace=True)
    else:
        mapping["Date"]="(missing)"; df["Date"]=np.nan
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)

    # Others
    map_col(STORE_ALIASES, "Store")
    map_col(SALES_ALIASES, "Sales", to_num=True, fill0=True)
    map_col(COGS_ALIASES, "COGS", to_num=True)
    map_col(QTY_ALIASES, "Qty", to_num=True, fill0=True, fuzzy_extra=["unit","piece","pcs","qty","qnty","quantity"])
    map_col(TICKETS_ALIASES, "Tickets", to_num=True)
    map_col(CAT_ALIASES, "Category")
    map_col(ITEM_ALIASES, "Item")
    map_col(INVQ_ALIASES, "InventoryQty", to_num=True)
    map_col(INVV_ALIASES, "InventoryValue", to_num=True)

    # Derived
    df["GrossProfit"] = (df["Sales"] - df["COGS"]) if df["COGS"].notna().any() else np.nan
    df["Store"] = df["Store"].fillna("").astype(str).str.strip()

    # Fallback: if Qty all zeros but Tickets present, use Tickets
    if df["Qty"].fillna(0).sum()==0 and df["Tickets"].fillna(0).sum()>0:
        df["Qty"] = df["Tickets"].fillna(0)
        mapping["Qty"] += " (fallback from Tickets)"

    return df, mapping

def summarize(df: pd.DataFrame):
    if df.empty:
        return {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    total_sales = float(np.nansum(df["Sales"]))
    total_gp    = float(np.nansum(df["GrossProfit"])) if df["GrossProfit"].notna().any() else np.nan
    tickets     = float(np.nansum(df["Tickets"])) if df["Tickets"].notna().any() else np.nan
    avg_basket  = (total_sales / tickets) if tickets and tickets > 0 else np.nan
    inv_value   = float(np.nansum(df["InventoryValue"])) if df["InventoryValue"].notna().any() else np.nan
    kpis = {"Sales": round(total_sales,2),
            "Gross Profit": None if np.isnan(total_gp) else round(total_gp,2),
            "Tickets": None if np.isnan(tickets) else round(tickets,0),
            "Avg Basket": None if np.isnan(avg_basket) else round(avg_basket,2),
            "Inventory Value": None if np.isnan(inv_value) else round(inv_value,2)}
    ts = df.dropna(subset=["Date"]).groupby("Date", as_index=False).agg({"Sales":"sum","GrossProfit":"sum","Tickets":"sum"})
    by_store = df.groupby("Store", as_index=False).agg({"Sales":"sum","GrossProfit":"sum","Tickets":"sum"}).sort_values("Sales",ascending=False)
    by_cat = (df.groupby("Category", as_index=False).agg({"Sales":"sum","GrossProfit":"sum","Qty":"sum"})
              .sort_values("Sales",ascending=False)) if df["Category"].notna().any() else pd.DataFrame()
    top_items = (df.groupby("Item", as_index=False).agg({"Sales":"sum","GrossProfit":"sum","Qty":"sum"})
                 .sort_values("Sales",ascending=False).head(20)) if df["Item"].notna().any() else pd.DataFrame()
    return kpis, ts, by_store, by_cat, top_items

def _pct(a,b):
    if b in (0,None) or pd.isna(b): return None
    return 100*(a-b)/b

def quick_insights(df, ts, by_store, by_cat, kpis):
    out=[]
    if not by_store.empty:
        out.append(f"🏆 Top store: **{by_store.iloc[0]['Store']}** — SAR {by_store.iloc[0]['Sales']:,.0f}")
        if len(by_store)>1:
            out.append(f"📉 Lowest store: **{by_store.iloc[-1]['Store']}** — SAR {by_store.iloc[-1]['Sales']:,.0f}")
    if not ts.empty and len(ts)>=2:
        ts2=ts.sort_values("Date")
        chg=_pct(float(ts2["Sales"].iloc[-1]), float(ts2["Sales"].iloc[-2]))
        if chg is not None:
            out.append(("▲" if chg>=0 else "▼")+f" DoD change: **{chg:+.1f}%**")
    if not by_cat.empty:
        out.append(f"🧺 Top category: **{by_cat.iloc[0]['Category']}** — SAR {by_cat.iloc[0]['Sales']:,.0f}")
    gp,k=kpis.get("Gross Profit"), kpis.get("Sales",0)
    if gp is not None and k:
        out.append(f"💰 Gross margin: **{(gp/k)*100:,.1f}%** (GP SAR {gp:,.0f})")
    if df["Tickets"].notna().any():
        t=float(np.nansum(df["Tickets"]))
        if t: out.append(f"🎟️ Avg basket: **SAR {float(np.nansum(df['Sales']))/t:,.2f}**")
    return out

# =========================
# HEADER BAR + REFRESH
# =========================
st.markdown(f"""
<div class="brandbar card">
  <div style="display:flex;align-items:center;justify-content:space-between;">
    <div style="font-weight:800;font-size:1.2rem;">🏪 Alkhair Family Market — Daily Dashboard</div>
    <div class="small">Auto refresh: every {REFRESH_MINUTES} min • UTC {datetime.utcnow().strftime('%H:%M:%S')}</div>
  </div>
</div>
""", unsafe_allow_html=True)

_, refresh_col = st.columns([8,1.2])
with refresh_col:
    st.markdown('<div class="btn-refresh">', unsafe_allow_html=True)
    if st.button("🔄 Refresh", use_container_width=True, help="Reload data now"):
        st.cache_data.clear(); st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# LOAD & STANDARDIZE
# =========================
try:
    raw = load_data(GSHEET_URL)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

std, mapping = standardize(raw)

# Date range (show as DD/MM/YYYY)
if std["Date"].notna().any():
    dmin = pd.to_datetime(std["Date"].min()).date()
    dmax = pd.to_datetime(std["Date"].max()).date()
    start_date, end_date = st.date_input("Date range", value=(dmin, dmax))
    st.caption(f"Selected: {start_date.strftime('%d/%m/%Y')} – {end_date.strftime('%d/%m/%Y')}")
else:
    start_date, end_date = None, None
    st.caption("No valid dates detected.")

# Mapping (debug)
with st.expander("Data mapping (detected columns)"):
    st.dataframe(pd.DataFrame({"Target": list(mapping.keys()), "Source column": list(mapping.values())}),
                 use_container_width=True)

# Filters
f = std.copy()
if start_date and end_date and f["Date"].notna().any():
    d1 = pd.to_datetime(start_date)
    d2 = pd.to_datetime(end_date) + pd.Timedelta(days=1)
    f = f[(f["Date"]>=d1) & (f["Date"]<d2)]

stores = sorted([s for s in f["Store"].dropna().unique().tolist() if s])
default_stores = [s for s in ["Magnus","Alkhair","Zelayi"] if s in stores] or stores
store_sel = st.multiselect("Store(s)", stores, default=default_stores)
if store_sel: f = f[f["Store"].isin(store_sel)]

# =========================
# SUMMARIES
# =========================
kpis, ts, by_store, by_cat, top_items = summarize(f)

# KPI CARDS
st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
for label, icon, val in [
    ("Sales (SAR)", "💵", f"{kpis.get('Sales', 0):,.2f}"),
    ("Gross Profit (SAR)", "📈", "-" if kpis.get("Gross Profit") is None else f"{kpis.get('Gross Profit'):,.2f}"),
    ("Tickets", "🎟️", "-" if kpis.get("Tickets") is None else f"{kpis.get('Tickets'):,.0f}"),
    ("Avg Basket (SAR)", "🧺", "-" if kpis.get("Avg Basket") is None else f"{kpis.get('Avg Basket'):,.2f}"),
]:
    with st.container():
        st.markdown('<div class="kpi">', unsafe_allow_html=True)
        st.markdown(f'<div class="label"><span class="icon">{icon}</span>{label}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="value">{val}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# =========================
# CHART CARDS
# =========================
col1, col2 = st.columns([1.2, 1])
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">📈 Sales Trend</div>', unsafe_allow_html=True)
    if not ts.empty:
        st.line_chart(ts.set_index("Date")[["Sales"]])
    else:
        st.info("No data for selected period.")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">🏬 Sales by Store</div>', unsafe_allow_html=True)
    if not by_store.empty:
        st.bar_chart(by_store.set_index("Store")[["Sales"]])
    else:
        st.info("No store data.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)
col3, col4 = st.columns(2)

with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">🛍️ Sales by Category</div>', unsafe_allow_html=True)
    if not by_cat.empty:
        # Horizontal bar for readability
        dfc = by_cat.copy().sort_values("Sales", ascending=True)
        fig, ax = plt.subplots(figsize=(6,4))
        ax.barh(dfc["Category"].astype(str), dfc["Sales"])
        ax.set_xlabel("Sales (SAR)")
        ax.set_ylabel("")
        ax.ticklabel_format(style="plain", axis="x")
        for i, v in enumerate(dfc["Sales"]):
            ax.text(v, i, f" {v:,.0f}", va="center")
        st.pyplot(fig)
    else:
        st.info("No category data.")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">⭐ Top Items</div>', unsafe_allow_html=True)
    if not top_items.empty:
        show = top_items.head(10).copy()
        show["Sales"] = show["Sales"].map(lambda x: f"{x:,.0f}")
        if show["GrossProfit"].notna().any():
            show["GrossProfit"] = show["GrossProfit"].map(lambda x: f"{x:,.0f}")
        show["Qty"] = show["Qty"].fillna(0).astype(float).astype(int)
        st.dataframe(show, use_container_width=True)
    else:
        st.info("No item data.")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# QUICK INSIGHTS + DOWNLOAD
# =========================
st.markdown("<br/>", unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-header">💡 Quick Insights</div>', unsafe_allow_html=True)
ins = quick_insights(f, ts, by_store, by_cat, kpis)
if ins:
    for i in ins:
        st.write(i)
else:
    st.write("No insights for current filters.")
st.markdown('</div>', unsafe_allow_html=True)

# Download filtered view as Excel
buff = io.BytesIO()
with pd.ExcelWriter(buff, engine="xlsxwriter") as wr:
    f.to_excel(wr, sheet_name="Data", index=False)
    if not ts.empty:        ts.to_excel(wr, sheet_name="TimeSeries", index=False)
    if not by_store.empty:  by_store.to_excel(wr, sheet_name="ByStore", index=False)
    if not by_cat.empty:    by_cat.to_excel(wr, sheet_name="ByCategory", index=False)
    if not top_items.empty: top_items.to_excel(wr, sheet_name="TopItems", index=False)

st.download_button(
    "⬇️ Download Excel (current view)",
    data=buff.getvalue(),
    file_name=f"Alkhair_Dashboard_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
