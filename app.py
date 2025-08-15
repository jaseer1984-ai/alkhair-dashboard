# app.py — Alkhair Family Market Daily Dashboard (no Cash/Bank KPIs)
# Paste a Google Sheets link (share or published CSV) OR upload CSV/Excel.
# The app auto-converts normal sheet links, and accepts published CSV links directly.

import io, re
from urllib.parse import urlparse, parse_qs
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---------- Page ----------
st.set_page_config(page_title="Alkhair Family Market — Daily Dashboard", layout="wide")

# ---------- Column aliases ----------
DATE_ALIASES    = ["date", "bill_date", "txn_date"]
STORE_ALIASES   = ["store", "branch", "location"]
SALES_ALIASES   = ["sales", "amount", "net_sales", "revenue"]
COGS_ALIASES    = ["cogs", "cost", "purchase_cost"]
QTY_ALIASES     = ["qty", "quantity"]
TICKETS_ALIASES = ["tickets", "bills", "invoice_count"]
CAT_ALIASES     = ["category", "cat"]
ITEM_ALIASES    = ["item", "product", "sku"]
INVQ_ALIASES    = ["inventoryqty", "inventory_qty", "stock_qty"]
INVV_ALIASES    = ["inventoryvalue", "inventory_value", "stock_value"]

ROUND_DP = 2

# ---------- Helpers ----------
def _first_col(df: pd.DataFrame, aliases: List[str]):
    low = {c.lower().strip(): c for c in df.columns}
    for a in aliases:
        if a in low:
            return low[a]
    return None

def _gsheet_to_csv_url(url: str) -> str:
    """
    Accepts:
      - Normal sheet links: .../spreadsheets/d/<ID>/edit#gid=...
      - Export links:       .../spreadsheets/d/<ID>/export?format=csv&gid=...
      - Published links:    .../spreadsheets/d/e/<PUB_ID>/pub?output=csv
    Returns a direct CSV URL. For published links (/d/e/...), returns the URL unchanged.
    """
    if "docs.google.com/spreadsheets" not in url:
        return url

    # If it's already a published CSV link, just return it.
    if "/spreadsheets/d/e/" in url and "output=csv" in url:
        return url

    # If it's already an export csv link, return it
    if "/export" in url and "format=csv" in url:
        return url

    # Otherwise, convert /d/<ID> style links to export csv with gid
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    if not m:
        return url
    sheet_id = m.group(1)

    # Try to get gid from query/hash; default 0
    gid = "0"
    m2 = re.search(r"gid=(\d+)", url)
    if m2:
        gid = m2.group(1)

    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

@st.cache_data(show_spinner=False)
def load_data(files, url):
    frames = []

    # URL (optional)
    if url:
        try:
            url2 = _gsheet_to_csv_url(url.strip())
            frames.append(pd.read_csv(url2))
        except Exception as e:
            st.warning(f"Couldn't read URL: {e}")

    # Uploaded files (optional)
    for f in files or []:
        try:
            if f.name.lower().endswith((".xlsx", ".xls")):
                frames.append(pd.read_excel(f))
            else:
                frames.append(pd.read_csv(f))
        except Exception as e:
            st.warning(f"Couldn't read file {f.name}: {e}")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def standardize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})

    def map_col(aliases, new, to_num=False):
        c = _first_col(df, aliases)
        if c is not None:
            df.rename(columns={c: new}, inplace=True)
            if to_num:
                df[new] = pd.to_numeric(df[new], errors="coerce")
        else:
            df[new] = np.nan

    map_col(DATE_ALIASES, "Date")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)

    map_col(STORE_ALIASES, "Store")
    map_col(SALES_ALIASES, "Sales", True)
    map_col(COGS_ALIASES, "COGS", True)
    map_col(QTY_ALIASES, "Qty", True)
    map_col(TICKETS_ALIASES, "Tickets", True)
    map_col(CAT_ALIASES, "Category")
    map_col(ITEM_ALIASES, "Item")
    map_col(INVQ_ALIASES, "InventoryQty", True)
    map_col(INVV_ALIASES, "InventoryValue", True)

    df["GrossProfit"] = df["Sales"] - df["COGS"] if df["COGS"].notna().any() else np.nan
    df["Store"] = df["Store"].fillna("").astype(str).str.strip()
    return df

def summarize(df: pd.DataFrame):
    if df.empty:
        return {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    total_sales = float(np.nansum(df["Sales"]))
    total_gp    = float(np.nansum(df["GrossProfit"])) if df["GrossProfit"].notna().any() else np.nan
    tickets     = float(np.nansum(df["Tickets"])) if df["Tickets"].notna().any() else np.nan
    avg_basket  = (total_sales / tickets) if tickets and tickets > 0 else np.nan
    inv_value   = float(np.nansum(df["InventoryValue"])) if df["InventoryValue"].notna().any() else np.nan

    kpis = {
        "Sales":            round(total_sales, ROUND_DP),
        "Gross Profit":     None if np.isnan(total_gp) else round(total_gp, ROUND_DP),
        "Tickets":          None if np.isnan(tickets) else round(tickets, 0),
        "Avg Basket":       None if np.isnan(avg_basket) else round(avg_basket, ROUND_DP),
        "Inventory Value":  None if np.isnan(inv_value) else round(inv_value, ROUND_DP),
    }

    ts = df.dropna(subset=["Date"]).groupby("Date", as_index=False).agg({
        "Sales": "sum", "GrossProfit": "sum", "Tickets": "sum"
    })

    by_store = df.groupby("Store", as_index=False).agg({
        "Sales": "sum", "GrossProfit": "sum", "Tickets": "sum"
    }).sort_values("Sales", ascending=False)

    by_cat = (df.groupby("Category", as_index=False)
                .agg({"Sales":"sum","GrossProfit":"sum","Qty":"sum"})
                .sort_values("Sales", ascending=False)
             ) if df["Category"].notna().any() else pd.DataFrame()

    top_items = (df.groupby("Item", as_index=False)
                   .agg({"Sales":"sum","GrossProfit":"sum","Qty":"sum"})
                   .sort_values("Sales", ascending=False).head(20)
                ) if df["Item"].notna().any() else pd.DataFrame()

    return kpis, ts, by_store, by_cat, top_items

def _pct(a, b):
    if b in (0, None) or pd.isna(b): return None
    return 100.0 * (a - b) / b

def quick_insights(df, ts, by_store, by_cat, kpis):
    out = []
    if not by_store.empty:
        out.append(f"🏆 Top store: **{by_store.iloc[0]['Store']}** — SAR {by_store.iloc[0]['Sales']:,.0f}")
        if len(by_store) > 1:
            out.append(f"📉 Lowest store: **{by_store.iloc[-1]['Store']}** — SAR {by_store.iloc[-1]['Sales']:,.0f}")
    if not ts.empty and len(ts) >= 2:
        ts2 = ts.sort_values("Date")
        chg = _pct(float(ts2["Sales"].iloc[-1]), float(ts2["Sales"].iloc[-2]))
        if chg is not None:
            arrow = "▲" if chg >= 0 else "▼"
            out.append(f"{arrow} Day-over-day change: **{chg:+.1f}%**")
    if not by_cat.empty:
        out.append(f"🧺 Top category: **{by_cat.iloc[0]['Category']}** — SAR {by_cat.iloc[0]['Sales']:,.0f}")
    gp, sales = kpis.get("Gross Profit"), kpis.get("Sales", 0)
    if gp is not None and sales:
        out.append(f"💰 Gross margin: **{(gp/sales)*100:,.1f}%** (GP SAR {gp:,.0f})")
    if df["Tickets"].notna().any():
        t = float(np.nansum(df["Tickets"]))
        if t:
            out.append(f"🎟️ Avg basket: **SAR {float(np.nansum(df['Sales']))/t:,.2f}**")
    return out

# ---------- UI ----------
st.markdown("<h1 style='text-align:center;'>Alkhair Family Market — Daily Dashboard</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Data Source")
    url = st.text_input(
        "Google Sheets (share or published CSV) / CSV / Excel URL",
        value="https://docs.google.com/spreadsheets/d/e/2PACX-1vSW_ui1m_393ipZv8NAliu1rly6zeifFxMfOWpQF17hjVIDa9Ll8PiGCaz8gTRkMQ/pub?output=csv",
    )
    files = st.file_uploader("Or upload CSV/Excel", accept_multiple_files=True, type=["csv","xlsx","xls"])
    st.caption("Tip: if you paste a normal Sheets link, the app converts it to CSV automatically. Published CSV links work as-is.")

raw = load_data(files, url)
if raw.empty:
    st.info("No rows loaded. Check the link/file permissions or try uploading a CSV/Excel to test.")
    st.stop()

std = standardize(raw)

# Filters
if std["Date"].notna().any():
    dmin, dmax = pd.to_datetime(std["Date"].min()), pd.to_datetime(std["Date"].max())
    date_rng = st.date_input("Date range", (dmin.date(), dmax.date()))
else:
    date_rng = None

stores = sorted([s for s in std["Store"].dropna().unique().tolist() if s])
default_stores = [s for s in ["Magnus","Alkhair","Zelayi"] if s in stores] or stores
store_sel = st.multiselect("Store(s)", stores, default=default_stores)

f = std.copy()
if store_sel:
    f = f[f["Store"].isin(store_sel)]
if date_rng and isinstance(date_rng, tuple) and len(date_rng) == 2 and f["Date"].notna().any():
    d1 = pd.to_datetime(date_rng[0]); d2 = pd.to_datetime(date_rng[1]) + pd.Timedelta(days=1)
    f = f[(f["Date"] >= d1) & (f["Date"] < d2)]

# Summaries
kpis, ts, by_store, by_cat, top_items = summarize(f)

# KPI cards
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Sales (SAR)", f"{kpis.get('Sales', 0):,.2f}")
with c2:
    gp = kpis.get("Gross Profit")
    st.metric("Gross Profit (SAR)", "-" if gp is None else f"{gp:,.2f}")
with c3:
    t = kpis.get("Tickets")
    st.metric("Tickets", "-" if t is None else f"{t:,.0f}")
with c4:
    ab = kpis.get("Avg Basket")
    st.metric("Avg Basket (SAR)", "-" if ab is None else f"{ab:,.2f}")

st.divider()

# Charts
left, right = st.columns([1.2, 1])
with left:
    st.subheader("📈 Sales Trend")
    if not ts.empty:
        st.line_chart(ts.set_index("Date")[["Sales"]])
    else:
        st.info("No dates available for trend.")
with right:
    st.subheader("🏬 Sales by Store")
    if not by_store.empty:
        st.bar_chart(by_store.set_index("Store")[["Sales"]])
    else:
        st.info("No store data.")

col1, col2 = st.columns([1,1])
with col1:
    st.subheader("🧺 Sales by Category")
    if not by_cat.empty:
        fig, ax = plt.subplots(figsize=(5,4))
        ax.pie(by_cat["Sales"].values, labels=by_cat["Category"].astype(str).values,
               autopct="%1.0f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)
    else:
        st.info("No category column found.")
with col2:
    st.subheader("⭐ Top Items")
    if not top_items.empty:
        st.dataframe(top_items.head(10), use_container_width=True)
    else:
        st.info("No item column found.")

# Quick Insights
st.subheader("✨ Quick Insights")
ins = quick_insights(f, ts, by_store, by_cat, kpis)
if ins:
    st.markdown("\n".join([f"- {x}" for x in ins]))
else:
    st.info("No insights for current filters.")

# Export current view
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
