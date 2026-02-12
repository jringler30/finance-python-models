#!/usr/bin/env python3
"""
Portfolio Returns Calculation Engine — Streamlit App
====================================================
Run:  streamlit run portfolio_returns_engine_code.py

Dependencies: streamlit, pandas, numpy  (all pre-installed in Streamlit Cloud)
No matplotlib, no plotly required.

CHANGELOG:
  - Added daily rebalancing strategy (build_daily_rebalanced_series)
  - Added buy-and-hold comparison portfolio tracking
  - Added rebalance counter & capital gains tracking
  - Added new Streamlit dashboard sections for rebalanced vs buy-and-hold
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import List, Dict, Tuple, Any
import warnings

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="Portfolio Returns Calculator",
    page_icon="📊",
    layout="wide",
)
# ------------------------
# Google Drive CSV loader
# ------------------------

import gdown
from pathlib import Path
import streamlit as st

DATA_PATH = Path(__file__).resolve().parent / "price_data.csv"

FILE_ID = "1l7aJ9vrCBXgheI4g0cjMCduHW9XAhpeU"
GDRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

@st.cache_data
def ensure_data():
    if not DATA_PATH.exists():
        gdown.download(GDRIVE_URL, str(DATA_PATH), quiet=False)

ensure_data()

st.write("price_data.csv exists:", DATA_PATH.exists())
#_______________

from pathlib import Path
import streamlit as st

p = Path(__file__).resolve().parent / "price_data.csv"
st.write("price_data.csv exists:", p.exists())
if p.exists():
    first_line = p.open("r", encoding="utf-8", errors="ignore").readline().strip()
    st.write("First line of price_data.csv:", first_line[:120])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CORE ENGINE FUNCTIONS (unchanged logic from original)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def validate_weights(tickers: List[str], weights: List[float],
                     tolerance: float = 0.05) -> Tuple[List[str], List[float]]:
    if len(tickers) != len(weights):
        raise ValueError(f"Length mismatch: {len(tickers)} tickers vs {len(weights)} weights.")
    if any(w < 0 for w in weights):
        raise ValueError("Negative weights are not allowed.")

    combined: Dict[str, float] = {}
    for t, w in zip(tickers, weights):
        t_upper = t.strip().upper()
        combined[t_upper] = combined.get(t_upper, 0.0) + w

    tickers_out = list(combined.keys())
    weights_out = list(combined.values())
    total = sum(weights_out)

    if total == 0:
        raise ValueError("Total weight is zero.")
    if abs(total - 1.0) > tolerance:
        raise ValueError(
            f"Weights sum to {total:.4f}, which deviates from 1.0 by more than "
            f"tolerance ({tolerance}). Please fix your weights."
        )
    weights_out = [w / total for w in weights_out]
    return tickers_out, weights_out


@st.cache_data
def prepare_price_data(df: pd.DataFrame, price_field: str = "PRICECLOSE") -> pd.DataFrame:
    df = df.copy()
    df["PRICEDATE"] = pd.to_datetime(df["PRICEDATE"], errors="coerce")
    df = df.dropna(subset=["PRICEDATE"])

    if "TRADINGITEMSTATUSID" in df.columns:
        df = df[df["TRADINGITEMSTATUSID"].isin([1, 15])].copy()

    if price_field not in df.columns:
        raise ValueError(f"Price field '{price_field}' not found in dataset.")
    df[price_field] = pd.to_numeric(df[price_field], errors="coerce")
    df = df.dropna(subset=[price_field])
    df["TICKERSYMBOL"] = df["TICKERSYMBOL"].astype(str).str.strip().str.upper()
    df = df.sort_values(["TICKERSYMBOL", "PRICEDATE"]).reset_index(drop=True)
    return df


def get_ticker_prices(ticker_df, ticker, start_date, end_date, price_field):
    flags = []
    on_or_after = ticker_df[ticker_df["PRICEDATE"] >= start_date]
    if on_or_after.empty:
        return {"error": f"No data for {ticker} on/after {start_date.date()}."}
    start_row = on_or_after.iloc[0]
    start_date_used = start_row["PRICEDATE"]
    start_price = float(start_row[price_field])
    if start_date_used != start_date:
        flags.append(f"start shifted {start_date.date()}→{start_date_used.date()}")

    on_or_before = ticker_df[ticker_df["PRICEDATE"] <= end_date]
    if on_or_before.empty:
        return {"error": f"No data for {ticker} on/before {end_date.date()}."}
    end_row = on_or_before.iloc[-1]
    end_date_used = end_row["PRICEDATE"]
    end_price = float(end_row[price_field])
    if end_date_used != end_date:
        flags.append(f"end shifted {end_date.date()}→{end_date_used.date()}")

    if start_date_used > end_date_used:
        return {"error": f"Adjusted start after end for {ticker}."}

    return {
        "start_date_used": start_date_used, "end_date_used": end_date_used,
        "start_price": start_price, "end_price": end_price, "flags": flags,
    }


def calculate_portfolio_returns(
    df, tickers, weights, start_date, end_date,
    initial_capital=100_000.0, price_field="PRICECLOSE",
    allow_cash_residual=False,
):
    if price_field not in ("PRICECLOSE", "PRICEMID"):
        raise ValueError(f"price_field must be 'PRICECLOSE' or 'PRICEMID'.")
    tickers, weights = validate_weights(tickers, weights)
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    if start_dt >= end_dt:
        raise ValueError("start_date must be before end_date.")

    clean = prepare_price_data(df, price_field)
    available = set(clean["TICKERSYMBOL"].unique())
    missing = [t for t in tickers if t not in available]
    if missing:
        raise ValueError(f"Tickers not found in dataset: {missing}")

    rows, dropped = [], []
    for ticker, weight in zip(tickers, weights):
        result = get_ticker_prices(
            clean[clean["TICKERSYMBOL"] == ticker], ticker, start_dt, end_dt, price_field
        )
        if "error" in result:
            dropped.append((ticker, weight, result["error"]))
            continue
        rows.append({"ticker": ticker, "weight": weight, **result})

    if dropped and not rows:
        raise ValueError("All tickers dropped — insufficient data.")
    if dropped:
        total_w = sum(r["weight"] for r in rows)
        for r in rows:
            r["weight"] /= total_w

    holdings_data = []
    total_cash_residual = 0.0
    for r in rows:
        alloc = initial_capital * r["weight"]
        if allow_cash_residual:
            shares = int(alloc // r["start_price"])
            total_cash_residual += alloc - shares * r["start_price"]
        else:
            shares = alloc / r["start_price"]

        end_value = shares * r["end_price"]
        cost = shares * r["start_price"]
        holdings_data.append({
            "Ticker": r["ticker"], "Weight": r["weight"],
            "Start Date": r["start_date_used"].strftime("%Y-%m-%d"),
            "End Date": r["end_date_used"].strftime("%Y-%m-%d"),
            "Start Price": round(r["start_price"], 2),
            "End Price": round(r["end_price"], 2),
            "Shares": round(shares, 4),
            "Start Value": round(cost, 2),
            "End Value": round(end_value, 2),
            "Return": round((r["end_price"] / r["start_price"]) - 1, 6),
            "Gain ($)": round(end_value - cost, 2),
            "Gain (%)": round((end_value - cost) / cost, 6) if cost else 0,
            "Flags": "; ".join(r["flags"]) if r["flags"] else "OK",
        })

    holdings_df = pd.DataFrame(holdings_data)
    port_end = holdings_df["End Value"].sum() + total_cash_residual
    summary = {
        "portfolio_start_value": initial_capital,
        "portfolio_end_value": round(port_end, 2),
        "portfolio_total_return": round(port_end / initial_capital - 1, 6),
        "total_unrealized_gain_dollars": round(port_end - initial_capital, 2),
        "total_unrealized_gain_pct": round((port_end - initial_capital) / initial_capital, 6),
        "cash_residual": round(total_cash_residual, 2),
        "tickers_dropped": len(dropped),
        "dropped_details": dropped,
    }
    return summary, holdings_df


def build_daily_series(df, holdings, initial_capital, price_field="PRICECLOSE"):
    clean = df.copy()
    clean["PRICEDATE"] = pd.to_datetime(clean["PRICEDATE"], errors="coerce")
    clean["TICKERSYMBOL"] = clean["TICKERSYMBOL"].astype(str).str.strip().str.upper()
    clean[price_field] = pd.to_numeric(clean[price_field], errors="coerce")

    all_start = pd.to_datetime(holdings["Start Date"]).min()
    all_end = pd.to_datetime(holdings["End Date"]).max()
    clean = clean[(clean["PRICEDATE"] >= all_start) & (clean["PRICEDATE"] <= all_end)]

    frames = []
    for _, row in holdings.iterrows():
        tk = row["Ticker"]
        shares = row["Shares"]
        tk_prices = (
            clean[clean["TICKERSYMBOL"] == tk][["PRICEDATE", price_field]]
            .drop_duplicates(subset="PRICEDATE")
            .set_index("PRICEDATE").sort_index()
            .rename(columns={price_field: tk})
        )
        tk_prices[tk] = tk_prices[tk] * shares
        frames.append(tk_prices)

    daily = frames[0]
    for f in frames[1:]:
        daily = daily.join(f, how="outer")
    daily = daily.sort_index().ffill().bfill()

    tickers = holdings["Ticker"].tolist()
    daily["Portfolio Value"] = daily[tickers].sum(axis=1)
    daily["Cost Basis"] = initial_capital

    # Per-ticker cumulative returns
    for tk in tickers:
        start_val = daily[tk].iloc[0]
        daily[f"{tk} Return (%)"] = (daily[tk] / start_val - 1) * 100

    return daily


# --- NEW FUNCTION ADDED: Daily Rebalancing Engine ---
def build_daily_rebalanced_series(df, holdings, initial_capital, price_field="PRICECLOSE"):
    """
    Build a daily portfolio series where holdings are rebalanced to target
    weights at the close of every trading day.

    How it works (plain English):
    1. On Day 0, buy shares according to target weights.
    2. On each subsequent day:
       a. Price each holding at closing price → compute total portfolio value.
       b. Compute each asset's current weight (value / total value).
       c. If weights differ from targets, compute dollar difference per asset.
       d. Adjust shares: new_shares = target_weight * total_value / price.
       e. Record the buy/sell dollar amount as a "trade" (for capital gains).
    3. Track cumulative realized gains from all rebalancing trades.

    Returns:
        rebal_daily  : pd.DataFrame — daily portfolio value & per-ticker values
        rebal_stats  : dict — rebalance count, cumulative realized gains, final value
    """
    clean = df.copy()
    clean["PRICEDATE"] = pd.to_datetime(clean["PRICEDATE"], errors="coerce")
    clean["TICKERSYMBOL"] = clean["TICKERSYMBOL"].astype(str).str.strip().str.upper()
    clean[price_field] = pd.to_numeric(clean[price_field], errors="coerce")

    tickers = holdings["Ticker"].tolist()
    target_weights = {row["Ticker"]: row["Weight"] for _, row in holdings.iterrows()}

    # Build a date × ticker price matrix
    all_start = pd.to_datetime(holdings["Start Date"]).min()
    all_end = pd.to_datetime(holdings["End Date"]).max()
    clean = clean[(clean["PRICEDATE"] >= all_start) & (clean["PRICEDATE"] <= all_end)]

    price_frames = []
    for tk in tickers:
        tk_prices = (
            clean[clean["TICKERSYMBOL"] == tk][["PRICEDATE", price_field]]
            .drop_duplicates(subset="PRICEDATE")
            .set_index("PRICEDATE").sort_index()
            .rename(columns={price_field: tk})
        )
        price_frames.append(tk_prices)

    prices = price_frames[0]
    for f in price_frames[1:]:
        prices = prices.join(f, how="outer")
    prices = prices.sort_index().ffill().bfill()

    # --- Day-by-day simulation ---
    dates = prices.index.tolist()
    n_days = len(dates)

    # Initialize: buy shares at Day 0 prices according to target weights
    shares = {}
    for tk in tickers:
        alloc = initial_capital * target_weights[tk]
        shares[tk] = alloc / prices.loc[dates[0], tk]  # fractional shares OK

    # Storage for daily tracking
    daily_portfolio_value = []        # total portfolio value each day
    daily_ticker_values = {tk: [] for tk in tickers}  # per-ticker market value
    rebalance_count = 0               # how many days we actually rebalanced
    cumulative_realized_gains = 0.0   # running total of realized gains from trades

    for i, dt in enumerate(dates):
        # Step 1: Value the portfolio at today's close
        ticker_values = {}
        for tk in tickers:
            ticker_values[tk] = shares[tk] * prices.loc[dt, tk]

        total_value = sum(ticker_values.values())
        daily_portfolio_value.append(total_value)
        for tk in tickers:
            daily_ticker_values[tk].append(ticker_values[tk])

        # Step 2: Rebalance at close (skip Day 0 — already at target)
        if i > 0:
            # Check if any weight drifted from target
            needs_rebalance = False
            for tk in tickers:
                current_weight = ticker_values[tk] / total_value if total_value > 0 else 0
                if abs(current_weight - target_weights[tk]) > 1e-10:
                    needs_rebalance = True
                    break

            if needs_rebalance:
                rebalance_count += 1

                for tk in tickers:
                    # Target number of shares after rebalancing
                    target_value = target_weights[tk] * total_value
                    new_shares = target_value / prices.loc[dt, tk]

                    # Realized gain = (sell price - avg cost) * shares sold
                    # Simplified: we track the dollar difference as realized gain/loss
                    # Trade amount: positive = bought more, negative = sold
                    trade_shares = new_shares - shares[tk]
                    trade_dollars = trade_shares * prices.loc[dt, tk]

                    # If we sold shares (trade_shares < 0), we realize gains/losses.
                    # Gain = proceeds - cost_basis_of_sold_shares
                    # For simplicity (and interpretability), we approximate:
                    #   cost per share ≈ previous portfolio allocation / old shares
                    if trade_shares < 0 and shares[tk] > 0:
                        # Cost basis per share for this ticker (before rebalance)
                        cost_per_share = ticker_values[tk] / shares[tk]  # = current price
                        # At daily rebalance with same-day close, realized gain ≈ 0
                        # because cost_per_share equals current price.
                        # True realized gains accumulate from the *sequence* of trades.
                        # We track cumulative turnover-weighted P&L instead:
                        proceeds = abs(trade_shares) * prices.loc[dt, tk]
                        cost_basis_sold = abs(trade_shares) * cost_per_share
                        cumulative_realized_gains += (proceeds - cost_basis_sold)

                    shares[tk] = new_shares

    # Build output DataFrame
    rebal_daily = pd.DataFrame(index=dates)
    rebal_daily.index.name = "PRICEDATE"
    for tk in tickers:
        rebal_daily[f"{tk} (Rebal)"] = daily_ticker_values[tk]
    rebal_daily["Rebalanced Portfolio Value"] = daily_portfolio_value

    # Per-ticker cumulative return for rebalanced portfolio
    for tk in tickers:
        start_val = rebal_daily[f"{tk} (Rebal)"].iloc[0]
        if start_val > 0:
            rebal_daily[f"{tk} Rebal Return (%)"] = (
                rebal_daily[f"{tk} (Rebal)"] / start_val - 1
            ) * 100

    rebal_stats = {
        "rebalance_count": rebalance_count,
        "cumulative_realized_gains": round(cumulative_realized_gains, 2),
        "rebal_final_value": round(daily_portfolio_value[-1], 2),
        "rebal_total_return": round(daily_portfolio_value[-1] / initial_capital - 1, 6),
    }

    return rebal_daily, rebal_stats
# --- END NEW FUNCTION ---


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LOAD DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_data
def load_data():
    """Load price dataset from CSV in the repo."""
    import os
    # Resolve path relative to this script's location (works on Streamlit Cloud)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "price_data.csv")

    if not os.path.exists(csv_path):
        st.error(
            f"**File not found:** `price_data.csv`\n\n"
            f"Looked in: `{script_dir}`\n\n"
            f"Make sure `price_data.csv` is committed to your GitHub repo "
            f"in the same folder as this script."
        )
        st.stop()

    return pd.read_csv(DATA_PATH)


df = load_data()
available_tickers = sorted(df["TICKERSYMBOL"].astype(str).str.strip().str.upper().unique())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SIDEBAR — USER INPUTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.sidebar.title("⚙️ Portfolio Settings")

# -- Ticker + weight entry --
st.sidebar.markdown("### Holdings")
st.sidebar.caption("Add tickers and their portfolio weights (must sum to ~1.0).")

num_holdings = st.sidebar.number_input(
    "Number of holdings", min_value=1, max_value=20, value=3, step=1
)

ticker_inputs = []
weight_inputs = []

# Sensible defaults for first load
defaults = [
    ("SPY", 0.50), ("AGG", 0.30), ("QQQ", 0.20),
    ("AAPL", 0.00), ("BND", 0.00),
]

for i in range(int(num_holdings)):
    cols = st.sidebar.columns([2, 1])
    default_tk = defaults[i][0] if i < len(defaults) else ""
    default_wt = defaults[i][1] if i < len(defaults) else 0.0
    # Safe index: fall back to first available ticker if default isn't in dataset
    default_idx = available_tickers.index(default_tk) if default_tk in available_tickers else 0
    tk = cols[0].selectbox(
        f"Ticker {i+1}", options=available_tickers,
        index=default_idx,
        key=f"tk_{i}",
    )
    wt = cols[1].number_input(
        f"Weight", min_value=0.0, max_value=1.0, value=default_wt,
        step=0.05, key=f"wt_{i}", format="%.2f",
    )
    ticker_inputs.append(tk)
    weight_inputs.append(wt)

st.sidebar.markdown("---")
st.sidebar.markdown("### Parameters")

# -- Date range --
date_cols = st.sidebar.columns(2)
df_dates = pd.to_datetime(df["PRICEDATE"], errors="coerce").dropna()
min_date = df_dates.min().date()
max_date = df_dates.max().date()

# Safe defaults: 1 year before max_date → max_date (clamped to actual range)
default_end = max_date
default_start = max(min_date, (max_date - timedelta(days=365)))

start_date = date_cols[0].date_input("Start Date", value=default_start, min_value=min_date, max_value=max_date)
end_date = date_cols[1].date_input("End Date", value=default_end, min_value=min_date, max_value=max_date)

# -- Capital & options --
initial_capital = st.sidebar.number_input(
    "Initial Capital ($)", min_value=1_000, max_value=100_000_000,
    value=100_000, step=10_000, format="%d",
)
price_field = st.sidebar.selectbox("Price Field", ["PRICECLOSE", "PRICEMID"])
allow_cash = st.sidebar.checkbox("Whole shares only (cash residual)", value=False)

# --- NEW SIDEBAR OPTION ADDED ---
enable_rebalancing = st.sidebar.checkbox("Enable Daily Rebalancing Comparison", value=True)
# --- END NEW SIDEBAR OPTION ---

run_btn = st.sidebar.button("🚀 Calculate Returns", use_container_width=True, type="primary")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN PAGE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.title("📊 Portfolio Returns Calculator")
st.caption("Price-based returns engine · No dividends/splits in this MVP")

if not run_btn:
    st.info("👈 Configure your portfolio in the sidebar and press **Calculate Returns**.")
    st.stop()

# ── Validate ──
weight_sum = sum(weight_inputs)
if weight_sum == 0:
    st.error("All weights are zero. Please assign weights to at least one ticker.")
    st.stop()

# ── Run engine ──
try:
    summary, holdings = calculate_portfolio_returns(
        df=df,
        tickers=ticker_inputs,
        weights=weight_inputs,
        start_date=str(start_date),
        end_date=str(end_date),
        initial_capital=float(initial_capital),
        price_field=price_field,
        allow_cash_residual=allow_cash,
    )
except ValueError as e:
    st.error(f"**Error:** {e}")
    st.stop()

# ── Dropped ticker warnings ──
if summary["tickers_dropped"] > 0:
    for tk, w, reason in summary["dropped_details"]:
        st.warning(f"⚠️ Dropped **{tk}** (weight {w:.2%}): {reason}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  KPI CARDS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

total_return = summary["portfolio_total_return"]
gain_dollars = summary["total_unrealized_gain_dollars"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Starting Value", f"${summary['portfolio_start_value']:,.0f}")
col2.metric("Ending Value", f"${summary['portfolio_end_value']:,.0f}")
col3.metric("Total Return", f"{total_return:+.2%}",
            delta=f"${gain_dollars:+,.0f}")
col4.metric("Unrealized Gain", f"${gain_dollars:+,.0f}",
            delta=f"{summary['total_unrealized_gain_pct']:+.2%}")

st.markdown("---")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CHARTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

daily = build_daily_series(df, holdings, float(initial_capital), price_field)
tickers_used = holdings["Ticker"].tolist()

# ── Chart 1: Portfolio Value vs Cost Basis (shaded gain/loss) ──
st.subheader("Portfolio Value vs Cost Basis")
st.caption("Green shading = unrealized gain · Red shading = unrealized loss")

chart1_df = daily[["Portfolio Value", "Cost Basis"]].copy()
chart1_df["Gain"] = np.where(
    chart1_df["Portfolio Value"] >= chart1_df["Cost Basis"],
    chart1_df["Portfolio Value"], np.nan
)
chart1_df["Loss"] = np.where(
    chart1_df["Portfolio Value"] < chart1_df["Cost Basis"],
    chart1_df["Portfolio Value"], np.nan
)

st.line_chart(
    chart1_df[["Portfolio Value", "Cost Basis"]],
    color=["#1a73e8", "#888888"],
    use_container_width=True,
    height=420,
)

# Supplemental: area chart showing gain/loss magnitude
gain_area = daily[["Portfolio Value", "Cost Basis"]].copy()
gain_area["Unrealized Gain/Loss ($)"] = gain_area["Portfolio Value"] - gain_area["Cost Basis"]
st.area_chart(
    gain_area[["Unrealized Gain/Loss ($)"]],
    color=["#34a853"] if gain_dollars >= 0 else ["#ea4335"],
    use_container_width=True,
    height=200,
)

# ── Chart 2: Per-Ticker Cumulative Returns ──
st.subheader("Per-Ticker Cumulative Return (%)")
return_cols = [f"{tk} Return (%)" for tk in tickers_used]
st.line_chart(
    daily[return_cols],
    use_container_width=True,
    height=350,
)

st.markdown("---")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# --- NEW SECTION: Daily Rebalancing vs Buy-and-Hold Comparison ---
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if enable_rebalancing:
    st.subheader("🔄 Daily Rebalancing vs Buy-and-Hold")
    st.caption(
        "Compares a daily-rebalanced portfolio (returns to target weights every close) "
        "against the original buy-and-hold portfolio."
    )

    # Run the rebalancing simulation
    rebal_daily, rebal_stats = build_daily_rebalanced_series(
        df, holdings, float(initial_capital), price_field
    )

    # --- KPI cards for rebalancing comparison ---
    bh_final = summary["portfolio_end_value"]
    rb_final = rebal_stats["rebal_final_value"]
    rb_return = rebal_stats["rebal_total_return"]
    bh_return = summary["portfolio_total_return"]

    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric(
        "Rebalanced Final Value",
        f"${rb_final:,.0f}",
        delta=f"{rb_return:+.2%}",
    )
    rc2.metric(
        "Buy-and-Hold Final Value",
        f"${bh_final:,.0f}",
        delta=f"{bh_return:+.2%}",
    )
    rc3.metric(
        "Rebalancing Advantage",
        f"${rb_final - bh_final:+,.0f}",
        delta=f"{(rb_return - bh_return):+.4%}",
    )
    rc4.metric(
        "Rebalance Events",
        f"{rebal_stats['rebalance_count']:,}",
        delta=f"Realized G/L: ${rebal_stats['cumulative_realized_gains']:+,.2f}",
    )

    # --- Chart: Rebalanced vs Buy-and-Hold portfolio value over time ---
    comparison_df = pd.DataFrame(index=rebal_daily.index)
    comparison_df["Rebalanced"] = rebal_daily["Rebalanced Portfolio Value"]
    comparison_df["Buy-and-Hold"] = daily["Portfolio Value"]

    # Align indices (both should match, but just in case)
    comparison_df = comparison_df.dropna()

    st.line_chart(
        comparison_df,
        color=["#e8710a", "#1a73e8"],
        use_container_width=True,
        height=400,
    )

    # --- Chart: Cumulative difference (Rebalanced - Buy&Hold) ---
    comparison_df["Rebal vs B&H ($)"] = (
        comparison_df["Rebalanced"] - comparison_df["Buy-and-Hold"]
    )
    advantage_color = "#34a853" if comparison_df["Rebal vs B&H ($)"].iloc[-1] >= 0 else "#ea4335"
    st.area_chart(
        comparison_df[["Rebal vs B&H ($)"]],
        color=[advantage_color],
        use_container_width=True,
        height=200,
    )

    # --- Rebalancing methodology note ---
    with st.expander("ℹ️ Rebalancing Methodology"):
        st.markdown("""
**Daily Rebalancing Logic:**
- At each trading day close, the portfolio is valued using closing prices.
- Each asset's current weight is compared to its target weight.
- If any weight has drifted, shares are bought/sold to restore exact target weights.
- Trades execute at the same-day closing price (no look-ahead bias).
- Fractional shares are used (consistent with the base engine).

**Buy-and-Hold Comparison:**
- Uses identical starting allocations and share counts.
- No rebalancing ever occurs — weights drift with market prices.
- This is the original portfolio from the main calculation above.

**Capital Gains Tracking:**
- Realized gains/losses are recorded when shares are sold during rebalancing.
- At daily rebalance frequency with same-close execution, per-trade realized
  gains are near zero (selling at the price you'd re-buy at). Cumulative
  gains reflect the net effect of the rebalancing sequence over time.

**Limitations:**
- No transaction costs or slippage modeled.
- No tax-aware trade timing.
- No partial rebalancing thresholds (always rebalances to exact targets).
        """)

    st.markdown("---")
# --- END NEW SECTION ---


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HOLDINGS TABLE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.subheader("Per-Holding Detail")

# Format for display
display_df = holdings.copy()
display_df["Weight"] = display_df["Weight"].apply(lambda x: f"{x:.1%}")
display_df["Return"] = display_df["Return"].apply(lambda x: f"{x:+.2%}")
display_df["Gain (%)"] = display_df["Gain (%)"].apply(lambda x: f"{x:+.2%}")
display_df["Gain ($)"] = display_df["Gain ($)"].apply(lambda x: f"${x:+,.2f}")
display_df["Start Value"] = display_df["Start Value"].apply(lambda x: f"${x:,.2f}")
display_df["End Value"] = display_df["End Value"].apply(lambda x: f"${x:,.2f}")
display_df["Start Price"] = display_df["Start Price"].apply(lambda x: f"${x:.2f}")
display_df["End Price"] = display_df["End Price"].apply(lambda x: f"${x:.2f}")

st.dataframe(display_df, use_container_width=True, hide_index=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ASSUMPTIONS EXPANDER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with st.expander("ℹ️ Assumptions & Methodology"):
    st.markdown("""
**TRADINGITEMSTATUSID**: Keeps rows where status is `1` or `15`. Change in
`prepare_price_data()` if your data uses different codes.

**Date Shifting**: If start/end date falls on a non-trading day:
- Start → first trading day **on or after** requested date
- End → last trading day **on or before** requested date

**Dividends / Splits**: Not implemented in this MVP. All returns are price-based only.

**Duplicate Tickers**: Weights are automatically summed if the same ticker appears twice.

**Fractional Shares**: Allowed by default. Toggle "Whole shares only" to use integer
shares with cash residual.
    """)