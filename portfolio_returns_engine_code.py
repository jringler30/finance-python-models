#!/usr/bin/env python3
"""
Portfolio Returns Calculation Engine — Streamlit App
====================================================
Run:  streamlit run portfolio_returns_engine_code.py

Dependencies: streamlit, pandas, numpy, gdown  (all pre-installed in Streamlit Cloud)
No matplotlib, no plotly required (uses native Streamlit charts).

CHANGELOG:
  - Original: buy-and-hold engine with daily series
  - V2: Added daily rebalancing strategy
  - V3 (CURRENT):
      • Unified rebalancing engine: Daily / Weekly / Monthly / Quarterly
      • Multi-strategy comparison dashboard (all frequencies + buy-and-hold)
      • Performance metrics table: Total Return, CAGR, Vol, Sharpe, Max DD, Turnover
      • Memory-optimized: single pivot, filtered early, vectorized math
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
# Google Drive Parquet loader
# ------------------------
import gdown
from pathlib import Path

DATA_PATH = Path("/tmp/price_data.parquet")
FILE_ID = "1pMQ817V05j4RK0vqJcVkBMmOBK5zRrug"
GDRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"


@st.cache_data(show_spinner=True)
def ensure_data():
    if not DATA_PATH.exists():
        out = gdown.download(GDRIVE_URL, str(DATA_PATH), quiet=False, fuzzy=True)
        if out is None or not DATA_PATH.exists():
            st.error("Google Drive download failed (permissions/quota/bad link).")
            st.stop()


ensure_data()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CORE ENGINE FUNCTIONS
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

    clean = df
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
    """Buy-and-hold daily series (original logic, unchanged)."""
    clean = df.copy()
    clean["PRICEDATE"] = pd.to_datetime(clean["PRICEDATE"], errors="coerce")
    clean["TICKERSYMBOL"] = clean["TICKERSYMBOL"].astype(str).str.strip().str.upper()
    clean[price_field] = pd.to_numeric(clean[price_field], errors="coerce")

    all_start = pd.to_datetime(holdings["Start Date"]).min()
    all_end = pd.to_datetime(holdings["End Date"]).max()
    clean = clean[(clean["PRICEDATE"] >= all_start) & (clean["PRICEDATE"] <= all_end)]

    tickers = holdings["Ticker"].tolist()
    clean = clean[clean["TICKERSYMBOL"].isin(tickers)]

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

    daily["Portfolio Value"] = daily[tickers].sum(axis=1)
    daily["Cost Basis"] = initial_capital

    for tk in tickers:
        start_val = daily[tk].iloc[0]
        daily[f"{tk} Return (%)"] = (daily[tk] / start_val - 1) * 100

    return daily


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# V3 CHANGE: Unified multi-frequency rebalancing engine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_prices_wide(df, tickers, start_date, end_date, price_field="PRICECLOSE"):
    """
    Build a (date × ticker) wide price matrix from the cleaned long DataFrame.
    Filters to only the needed tickers and date range FIRST for memory safety.
    Returns a DataFrame with DatetimeIndex and one column per ticker.
    """
    mask = (
        df["TICKERSYMBOL"].isin(tickers)
        & (df["PRICEDATE"] >= pd.Timestamp(start_date))
        & (df["PRICEDATE"] <= pd.Timestamp(end_date))
    )
    subset = df.loc[mask, ["TICKERSYMBOL", "PRICEDATE", price_field]].copy()
    subset = subset.drop_duplicates(subset=["TICKERSYMBOL", "PRICEDATE"])

    wide = subset.pivot(index="PRICEDATE", columns="TICKERSYMBOL", values=price_field)
    wide = wide.sort_index().ffill().bfill()

    # Ensure all requested tickers are present
    missing_cols = [t for t in tickers if t not in wide.columns]
    if missing_cols:
        raise ValueError(f"Tickers missing from price data after filtering: {missing_cols}")

    # Keep only requested tickers in requested order
    wide = wide[tickers]
    return wide


def _get_rebalance_dates(trading_dates, freq):
    """
    Given a sorted array of trading dates and a frequency string,
    return the SET of dates on which rebalancing should occur.

    freq options:
      "Daily"     → every trading day (skip day 0)
      "Weekly"    → first trading day of each ISO week
      "Monthly"   → first trading day of each calendar month
      "Quarterly" → first trading day of each quarter (Jan/Apr/Jul/Oct)
    """
    dates = pd.DatetimeIndex(trading_dates)
    if len(dates) < 2:
        return set()

    if freq == "Daily":
        # Rebalance every day except day 0
        return set(dates[1:])

    rebal_set = set()

    if freq == "Weekly":
        prev_week = dates[0].isocalendar()[1]
        prev_year = dates[0].year
        for dt in dates[1:]:
            iso = dt.isocalendar()
            if iso[1] != prev_week or dt.year != prev_year:
                rebal_set.add(dt)
                prev_week = iso[1]
                prev_year = dt.year

    elif freq == "Monthly":
        prev_month = dates[0].month
        prev_year = dates[0].year
        for dt in dates[1:]:
            if dt.month != prev_month or dt.year != prev_year:
                rebal_set.add(dt)
                prev_month = dt.month
                prev_year = dt.year

    elif freq == "Quarterly":
        quarter_months = {1, 4, 7, 10}
        prev_month = dates[0].month
        prev_year = dates[0].year
        for dt in dates[1:]:
            if dt.month in quarter_months and (dt.month != prev_month or dt.year != prev_year):
                rebal_set.add(dt)
            if dt.month != prev_month or dt.year != prev_year:
                prev_month = dt.month
                prev_year = dt.year

    else:
        raise ValueError(f"Unknown rebalance frequency: {freq}")

    return rebal_set


def build_rebalanced_series(prices_wide, target_weights, initial_capital, rebalance_freq):
    """
    Unified rebalancing engine.

    Parameters:
        prices_wide    : DataFrame (date × ticker) of prices — already filtered & pivoted
        target_weights : dict {ticker: weight} summing to 1.0
        initial_capital: float
        rebalance_freq : str in {"Daily", "Weekly", "Monthly", "Quarterly"}

    Returns:
        rebal_daily : DataFrame with "Portfolio Value" column + per-ticker value columns
        rebal_stats : dict with rebalance_count, turnover, final_value, total_return
    """
    tickers = list(target_weights.keys())
    dates = prices_wide.index.tolist()
    n_days = len(dates)

    if n_days == 0:
        raise ValueError("No trading dates in the filtered price data.")

    # Determine which dates are rebalance dates
    rebal_dates = _get_rebalance_dates(dates, rebalance_freq)

    # Initialize shares at Day 0
    shares = {}
    for tk in tickers:
        alloc = initial_capital * target_weights[tk]
        shares[tk] = alloc / prices_wide.loc[dates[0], tk]

    # Pre-allocate arrays for speed
    portfolio_values = np.empty(n_days, dtype=np.float64)
    ticker_values_arr = {tk: np.empty(n_days, dtype=np.float64) for tk in tickers}

    rebalance_count = 0
    total_turnover_dollars = 0.0  # sum of |trade_value| across all rebalances

    for i, dt in enumerate(dates):
        # Value the portfolio
        total_value = 0.0
        tv = {}
        for tk in tickers:
            val = shares[tk] * prices_wide.loc[dt, tk]
            tv[tk] = val
            total_value += val

        portfolio_values[i] = total_value
        for tk in tickers:
            ticker_values_arr[tk][i] = tv[tk]

        # Rebalance if this date is in the schedule
        if dt in rebal_dates and total_value > 0:
            day_turnover = 0.0
            needs_rebalance = False
            for tk in tickers:
                current_weight = tv[tk] / total_value
                if abs(current_weight - target_weights[tk]) > 1e-10:
                    needs_rebalance = True
                    break

            if needs_rebalance:
                rebalance_count += 1
                for tk in tickers:
                    target_value = target_weights[tk] * total_value
                    new_shares = target_value / prices_wide.loc[dt, tk]
                    trade_shares = new_shares - shares[tk]
                    trade_dollars = abs(trade_shares * prices_wide.loc[dt, tk])
                    day_turnover += trade_dollars
                    shares[tk] = new_shares

                total_turnover_dollars += day_turnover

    # Turnover proxy: total |trade$| / average portfolio value
    avg_port_value = np.mean(portfolio_values)
    turnover_proxy = (total_turnover_dollars / avg_port_value) if avg_port_value > 0 else 0.0

    # Build output DataFrame
    rebal_daily = pd.DataFrame(index=dates)
    rebal_daily.index.name = "PRICEDATE"
    for tk in tickers:
        rebal_daily[f"{tk} (Rebal)"] = ticker_values_arr[tk]
    rebal_daily["Portfolio Value"] = portfolio_values

    rebal_stats = {
        "rebalance_count": rebalance_count,
        "turnover_proxy": round(turnover_proxy, 4),
        "final_value": round(portfolio_values[-1], 2),
        "total_return": round(portfolio_values[-1] / initial_capital - 1, 6),
    }

    return rebal_daily, rebal_stats


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# V3 CHANGE: Performance metrics calculator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_strategy_metrics(daily_values: np.ndarray, initial_capital: float) -> dict:
    """
    Compute standard performance metrics from a daily portfolio value series.

    Returns dict with:
      total_return, cagr, annualized_vol, sharpe, max_drawdown
    """
    n = len(daily_values)
    if n < 2:
        return {
            "total_return": 0.0, "cagr": 0.0, "annualized_vol": 0.0,
            "sharpe": 0.0, "max_drawdown": 0.0,
        }

    final = daily_values[-1]
    total_return = final / initial_capital - 1

    # CAGR using 252 trading days/year
    years = n / 252.0
    if years > 0 and final > 0 and initial_capital > 0:
        cagr = (final / initial_capital) ** (1 / years) - 1
    else:
        cagr = 0.0

    # Daily returns
    daily_rets = np.diff(daily_values) / daily_values[:-1]
    ann_vol = np.std(daily_rets, ddof=1) * np.sqrt(252) if len(daily_rets) > 1 else 0.0

    # Sharpe (Rf = 0)
    sharpe = (cagr / ann_vol) if ann_vol > 0 else 0.0

    # Max drawdown
    running_max = np.maximum.accumulate(daily_values)
    drawdowns = (daily_values - running_max) / running_max
    max_dd = float(np.min(drawdowns))

    return {
        "total_return": round(total_return, 6),
        "cagr": round(cagr, 6),
        "annualized_vol": round(ann_vol, 6),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_dd, 6),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LOAD DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_REQUIRED_COLS = ["TICKERSYMBOL", "PRICEDATE", "PRICECLOSE", "PRICEMID", "TRADINGITEMSTATUSID"]


@st.cache_data(show_spinner=True)
def load_data():
    ensure_data()

    if not DATA_PATH.exists():
        st.error(f"Dataset not found: {DATA_PATH}")
        st.stop()
    if DATA_PATH.stat().st_size < 10 * 1024 * 1024:
        st.error("Downloaded file looks too small (likely corrupt). Reboot app to retry.")
        st.stop()

    try:
        df = pd.read_parquet(DATA_PATH, columns=_REQUIRED_COLS)
    except Exception as e:
        st.error(f"Failed to read parquet: {e}")
        st.stop()

    df = prepare_price_data(df, price_field="PRICECLOSE")
    return df


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

defaults = [
    ("SPY", 0.50), ("AGG", 0.30), ("QQQ", 0.20),
    ("AAPL", 0.00), ("BND", 0.00),
]

for i in range(int(num_holdings)):
    cols = st.sidebar.columns([2, 1])
    default_tk = defaults[i][0] if i < len(defaults) else ""
    default_wt = defaults[i][1] if i < len(defaults) else 0.0
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

# --- V3 CHANGE: Rebalancing controls ---
st.sidebar.markdown("---")
st.sidebar.markdown("### Rebalancing")

enable_rebalancing = st.sidebar.checkbox("Enable Rebalancing Comparison", value=True)

REBAL_FREQS = ["Daily", "Weekly", "Monthly", "Quarterly"]
selected_freq = st.sidebar.selectbox(
    "Rebalance Frequency",
    options=REBAL_FREQS,
    index=2,  # default Monthly
    disabled=not enable_rebalancing,
)

show_all_strategies = st.sidebar.checkbox(
    "Show all strategies (slower)",
    value=False,
    disabled=not enable_rebalancing,
    help="Compute & compare Buy-and-Hold + all 4 rebalance frequencies at once.",
)
# --- END V3 SIDEBAR CHANGES ---

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
#  KPI CARDS (Buy-and-Hold)
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
#  CHARTS (Buy-and-Hold)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

daily = build_daily_series(df, holdings, float(initial_capital), price_field)
tickers_used = holdings["Ticker"].tolist()

# ── Chart 1: Portfolio Value vs Cost Basis ──
st.subheader("Portfolio Value vs Cost Basis")
st.caption("Green shading = unrealized gain · Red shading = unrealized loss")

chart1_df = daily[["Portfolio Value", "Cost Basis"]].copy()
st.line_chart(
    chart1_df[["Portfolio Value", "Cost Basis"]],
    color=["#1a73e8", "#888888"],
    use_container_width=True,
    height=420,
)

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
# V3 CHANGE: Multi-Frequency Rebalancing Comparison Section
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if enable_rebalancing:
    st.subheader("🔄 Rebalancing Strategy Comparison")

    # Build target weights dict from holdings
    target_weights = {row["Ticker"]: row["Weight"] for _, row in holdings.iterrows()}

    # Build wide price matrix ONCE (memory-efficient: filtered by tickers + dates)
    try:
        all_start = pd.to_datetime(holdings["Start Date"]).min()
        all_end = pd.to_datetime(holdings["End Date"]).max()
        prices_wide = build_prices_wide(df, tickers_used, all_start, all_end, price_field)
    except ValueError as e:
        st.error(f"**Error building price matrix:** {e}")
        st.stop()

    # Determine which strategies to compute
    if show_all_strategies:
        freqs_to_run = REBAL_FREQS  # Daily, Weekly, Monthly, Quarterly
        st.caption("Computing all strategies: Buy & Hold + Daily / Weekly / Monthly / Quarterly rebalancing…")
    else:
        freqs_to_run = [selected_freq]
        st.caption(f"Comparing Buy & Hold vs **{selected_freq}** rebalancing.")

    # Run rebalancing for each frequency
    strategy_results = {}  # freq -> (rebal_daily, rebal_stats)
    for freq in freqs_to_run:
        try:
            rd, rs = build_rebalanced_series(prices_wide, target_weights, float(initial_capital), freq)
            strategy_results[freq] = (rd, rs)
        except ValueError as e:
            st.warning(f"⚠️ Could not compute {freq} rebalancing: {e}")

    if not strategy_results:
        st.error("No rebalancing strategies could be computed.")
    else:
        # ── Build comparison DataFrame for chart ──
        comparison_df = pd.DataFrame(index=prices_wide.index)
        comparison_df.index.name = "PRICEDATE"

        # Buy-and-hold series (from daily)
        bh_values = daily["Portfolio Value"].reindex(comparison_df.index)
        comparison_df["Buy & Hold"] = bh_values

        for freq, (rd, rs) in strategy_results.items():
            comparison_df[f"Rebal: {freq}"] = rd["Portfolio Value"].reindex(comparison_df.index)

        comparison_df = comparison_df.dropna()

        # ── Metrics table ──
        bh_vals_arr = comparison_df["Buy & Hold"].values
        bh_metrics = compute_strategy_metrics(bh_vals_arr, float(initial_capital))
        bh_metrics["rebalance_count"] = 0
        bh_metrics["turnover_proxy"] = 0.0

        metrics_rows = [{
            "Strategy": "Buy & Hold",
            "Final Value ($)": f"${bh_vals_arr[-1]:,.0f}",
            "Total Return (%)": f"{bh_metrics['total_return']:+.2%}",
            "CAGR (%)": f"{bh_metrics['cagr']:+.2%}",
            "Ann. Volatility (%)": f"{bh_metrics['annualized_vol']:.2%}",
            "Sharpe Ratio": f"{bh_metrics['sharpe']:.3f}",
            "Max Drawdown (%)": f"{bh_metrics['max_drawdown']:.2%}",
            "Turnover": f"{bh_metrics['turnover_proxy']:.2f}",
            "Rebal Events": 0,
        }]

        for freq in freqs_to_run:
            if freq not in strategy_results:
                continue
            rd, rs = strategy_results[freq]
            vals = comparison_df[f"Rebal: {freq}"].values
            m = compute_strategy_metrics(vals, float(initial_capital))
            metrics_rows.append({
                "Strategy": f"Rebal: {freq}",
                "Final Value ($)": f"${rs['final_value']:,.0f}",
                "Total Return (%)": f"{m['total_return']:+.2%}",
                "CAGR (%)": f"{m['cagr']:+.2%}",
                "Ann. Volatility (%)": f"{m['annualized_vol']:.2%}",
                "Sharpe Ratio": f"{m['sharpe']:.3f}",
                "Max Drawdown (%)": f"{m['max_drawdown']:.2%}",
                "Turnover": f"{rs['turnover_proxy']:.2f}",
                "Rebal Events": rs["rebalance_count"],
            })

        st.markdown("#### Performance Metrics")
        metrics_df = pd.DataFrame(metrics_rows)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        # ── KPI cards: selected strategy vs buy-and-hold ──
        primary_freq = freqs_to_run[0]
        if primary_freq in strategy_results:
            _, primary_stats = strategy_results[primary_freq]
            bh_final = bh_vals_arr[-1]
            rb_final = primary_stats["final_value"]
            rb_return = primary_stats["total_return"]
            bh_ret = bh_metrics["total_return"]

            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric(
                f"Rebal ({primary_freq}) Final",
                f"${rb_final:,.0f}",
                delta=f"{rb_return:+.2%}",
            )
            rc2.metric(
                "Buy-and-Hold Final",
                f"${bh_final:,.0f}",
                delta=f"{bh_ret:+.2%}",
            )
            rc3.metric(
                "Rebalancing Advantage",
                f"${rb_final - bh_final:+,.0f}",
                delta=f"{(rb_return - bh_ret):+.4%}",
            )
            rc4.metric(
                "Rebalance Events",
                f"{primary_stats['rebalance_count']:,}",
                delta=f"Turnover: {primary_stats['turnover_proxy']:.2f}",
            )

        # ── Chart: Portfolio value over time ──
        st.markdown("#### Portfolio Value Over Time")
        # Assign colors: blue for B&H, then orange/green/purple/red for rebal frequencies
        strategy_colors = ["#1a73e8"]  # B&H
        freq_color_map = {
            "Daily": "#e8710a",
            "Weekly": "#34a853",
            "Monthly": "#9c27b0",
            "Quarterly": "#ea4335",
        }
        for freq in freqs_to_run:
            if freq in strategy_results:
                strategy_colors.append(freq_color_map.get(freq, "#666666"))

        st.line_chart(
            comparison_df,
            color=strategy_colors,
            use_container_width=True,
            height=420,
        )

        # ── Toggle: Drawdown chart ──
        show_drawdown = st.checkbox("Show Drawdown Chart", value=False)
        if show_drawdown:
            st.markdown("#### Drawdown Over Time")
            dd_df = pd.DataFrame(index=comparison_df.index)
            for col in comparison_df.columns:
                vals = comparison_df[col].values
                running_max = np.maximum.accumulate(vals)
                dd_df[col] = ((vals - running_max) / running_max) * 100  # as percentage

            st.area_chart(
                dd_df,
                color=strategy_colors,
                use_container_width=True,
                height=300,
            )

        # ── Difference chart: selected rebal vs B&H ──
        if primary_freq in strategy_results:
            diff_col = f"Rebal: {primary_freq}"
            if diff_col in comparison_df.columns:
                diff_series = comparison_df[diff_col] - comparison_df["Buy & Hold"]
                diff_chart = pd.DataFrame({"Rebal vs B&H ($)": diff_series})
                advantage_color = "#34a853" if diff_series.iloc[-1] >= 0 else "#ea4335"
                st.area_chart(
                    diff_chart,
                    color=[advantage_color],
                    use_container_width=True,
                    height=200,
                )

        # ── Methodology expander ──
        with st.expander("ℹ️ Rebalancing Methodology"):
            st.markdown("""
**Rebalancing Logic:**
- At each scheduled rebalance date, the portfolio is valued using closing prices.
- Each asset's current weight is compared to its target weight.
- If any weight has drifted, shares are bought/sold to restore exact target weights.
- Trades execute at the same-day closing price (no look-ahead bias).
- Fractional shares are used (consistent with the base engine).

**Rebalance Schedules:**
- **Daily:** Every trading day.
- **Weekly:** First trading day of each ISO week (typically Monday).
- **Monthly:** First trading day of each calendar month.
- **Quarterly:** First trading day of each quarter (Jan / Apr / Jul / Oct).

**Metrics Definitions:**
- **CAGR:** Compound Annual Growth Rate, using 252 trading days = 1 year.
- **Annualized Volatility:** Std dev of daily returns × √252.
- **Sharpe Ratio:** CAGR / Annualized Volatility (risk-free rate = 0%).
- **Max Drawdown:** Largest peak-to-trough decline as a percentage.
- **Turnover:** Sum of |trade dollars| / average portfolio value. Higher = more trading.

**Limitations:**
- No transaction costs or slippage modeled.
- No tax-aware trade timing.
- No partial rebalancing thresholds (always rebalances to exact targets).
            """)

    st.markdown("---")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HOLDINGS TABLE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.subheader("Per-Holding Detail")

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