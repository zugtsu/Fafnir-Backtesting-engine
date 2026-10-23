"""Data page: markets already loaded, and adding your own price history."""

import re
import shutil

import numpy as np
import pandas as pd
import streamlit as st

from gui.common import (UPLOAD_DIR, error_box, load_cfg, market_rows, read_user_assets, write_user_assets)
from saty_bt.config import load_config, resolve_path, timeframe_minutes
from saty_bt.data.build import build_asset

st.title("Data")
cfg = load_cfg()

# --- what's loaded --------------------------------------------------------------
st.subheader("Markets")
rows = market_rows(cfg)
if rows:
    st.dataframe(pd.DataFrame([{**r, "timeframes": ", ".join(r["timeframes"])} for r in rows]),
                 width="stretch", hide_index=True)
else:
    st.info("No markets yet. Add one below.")

user = read_user_assets()
if user:
    with st.expander("Update or remove a market you added"):
        pick = st.selectbox("Market", sorted(user))
        downloaded = user[pick]["source"]["kind"] in ("yfinance", "ccxt")
        c = st.columns(2)
        if downloaded and c[0].button(f"Update {pick} to today"):
            try:
                with st.spinner(f"Downloading {pick}…"):
                    build_asset(load_cfg(), pick, network=True, verbose=False)
                st.cache_data.clear()
                st.success(f"{pick} is up to date.")
            except Exception as e:  # noqa: BLE001
                error_box(e)
        if c[1].button(f"Remove {pick}", type="secondary"):
            blocks = read_user_assets()
            blocks.pop(pick, None)
            write_user_assets(blocks)
            shutil.rmtree(resolve_path(cfg, "store") / pick, ignore_errors=True)
            (UPLOAD_DIR / f"{pick}.csv").unlink(missing_ok=True)
            st.cache_data.clear()
            st.success(f"Removed {pick}.")
            st.rerun()

# --- free downloads -------------------------------------------------------------
st.divider()
st.subheader("Download free data")

STOCKS = {
    "Index ETFs": ["SPY", "QQQ", "DIA", "IWM", "VTI"],
    "Tech": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "NFLX", "AMD", "AVGO", "ORCL", "CRM",
             "ADBE", "INTC", "PLTR", "UBER"],
    "Finance": ["JPM", "BAC", "GS", "V", "MA", "BRK-B", "COIN"],
    "Consumer and health": ["WMT", "COST", "KO", "PEP", "MCD", "NKE", "DIS", "UNH", "JNJ", "PFE", "LLY"],
    "Energy and industrial": ["XOM", "CVX", "BA", "CAT", "GE"],
    "Sector and commodity ETFs": ["XLK", "XLF", "XLE", "XLV", "GLD", "SLV", "TLT", "USO", "ARKK"],
}
CRYPTO = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "DOGE/USD", "ADA/USD", "AVAX/USD", "LINK/USD", "LTC/USD"]


base_cfg = load_config()
BUILTIN = base_cfg["assets"]
RESERVED = set(BUILTIN) | set(base_cfg.get("auxiliary", {}))
have = {r["market"] for r in rows}          # markets whose data is actually built


def market_name(symbol: str) -> str:
    """BRK-B -> BRK_B, ETH/USD -> ETHUSD. A built-in market with this exact feed keeps its
    name (its config already knows how to download it); any other clash gets _D."""
    name = re.sub(r"[^A-Z0-9_]", "_", symbol.upper().replace("/", "")).strip("_")[:15]
    src = BUILTIN.get(name, {}).get("source", {})
    if src.get("kind") in ("yfinance", "ccxt") and src.get("symbol", "").upper() == symbol.upper():
        return name
    return f"{name[:13]}_D" if name in RESERVED else name


def download(blocks: dict[str, dict]) -> None:
    """Add each market's settings, build it from the internet, and report per market."""
    done, failed = [], []
    status = st.status(f"Downloading {len(blocks)} market(s)…", expanded=True)
    for name, block in blocks.items():
        status.write(f"{name}…")
        own = name not in BUILTIN                  # built-ins already have settings in config.yaml
        saved = read_user_assets()
        if own:
            saved[name] = block
            write_user_assets(saved)
        try:
            meta = build_asset(load_cfg(), name, network=True, verbose=False)
            frames = {tf: v.get("bars", 0) for tf, v in meta["frames"].items() if tf in cfg["timeframes"] and v.get("bars")}
            done.append(f"{name} ({', '.join(frames)})")
        except Exception as e:  # noqa: BLE001
            if own:
                saved.pop(name, None)
                write_user_assets(saved)
                shutil.rmtree(resolve_path(cfg, "store") / name, ignore_errors=True)
            failed.append(f"{name}: {e}")
    st.cache_data.clear()
    status.update(label=f"Added {len(done)}, failed {len(failed)}", state="complete" if not failed else "error")
    if done:
        st.success("Added " + "; ".join(done) + ". Pick them on the Backtest page.")
    for f in failed:
        st.error(f)


tab_stocks, tab_crypto = st.tabs(["Stocks and ETFs", "Crypto"])
with tab_stocks:
    st.caption("From Yahoo Finance. Daily bars go back to 2015; Yahoo only gives the last 2 years of 1-hour "
               "bars and 60 days of 30-minute bars.")
    picked = []
    cols = st.columns(3)
    for i, (group, tickers) in enumerate(STOCKS.items()):
        options = [t for t in tickers if market_name(t) not in have]
        picked += cols[i % 3].multiselect(group, options, key=f"dl_{group}",
                                          placeholder="Already added" if not options else "Choose")
    extra = st.text_input("Any other Yahoo Finance tickers, comma separated", placeholder="SHOP, ^VIX, EURUSD=X")
    picked += [t.strip().upper() for t in extra.split(",") if t.strip()]
    if st.button("Download stocks", type="primary", disabled=not picked):
        download({market_name(t): {
            "class": "equity", "session": "rth_us", "timeframes": ["30m", "1h", "4h", "1d"],
            "source": {"kind": "yfinance", "symbol": t},
            "volume": {"mode": "native", "kind": "real"}, "volume_gate": "off",
            "costs": {"spread": 0.01, "slippage_bps": 1.0, "commission_bps": 0.5},
        } for t in dict.fromkeys(picked)})

with tab_crypto:
    st.caption("1-hour bars from a public exchange, no account needed. Longer history takes longer "
               "(about a minute for a few years).")
    c = st.columns([3, 1, 1])
    pairs = c[0].multiselect("Pairs", [p for p in CRYPTO if market_name(p) not in have])
    exchange = c[1].selectbox("Exchange", ["coinbase", "kraken"])
    since = c[2].selectbox("From", ["2023-01-01", "2022-01-01", "2021-01-01", "2020-01-01"])
    if st.button("Download crypto", type="primary", disabled=not pairs):
        download({market_name(p): {
            "class": "crypto", "session": "crypto_utc", "timeframes": ["1h", "4h", "1d"],
            "source": {"kind": "ccxt", "exchange": exchange, "symbol": p, "timeframe": "1h", "since": since},
            "volume": {"mode": "native", "kind": "real"}, "volume_gate": "off",
            "costs": {"spread": 0.0, "slippage_bps": 2.0, "commission_bps": 10.0},
        } for p in pairs})

# --- add your own ---------------------------------------------------------------
st.divider()
st.subheader("Add your own price history")
st.markdown(
    "Upload a CSV (or Parquet) with one row per bar: a time column plus **open, high, low, close**, and "
    "volume if you have it. Any bar size works (1-minute, 1-hour, daily); the app builds every larger "
    "timeframe from it. Exports from TradingView, MetaTrader, Dukascopy, Binance and most brokers work."
)

up = st.file_uploader("Price history file", type=["csv", "txt", "parquet"])
if up is None:
    st.stop()

try:
    raw = pd.read_parquet(up) if up.name.lower().endswith(".parquet") else pd.read_csv(up, sep=None, engine="python")
    if isinstance(raw.index, pd.DatetimeIndex):
        raw = raw.reset_index()
except Exception as e:  # noqa: BLE001
    error_box(e)
    st.stop()

st.caption(f"{len(raw):,} rows. First rows:")
st.dataframe(raw.head(8), width="stretch", hide_index=True)


def guess(names: list[str], *keys: str) -> int:
    low = [str(n).lower().strip() for n in names]
    for k in keys:
        for i, n in enumerate(low):
            if n == k:
                return i
    for k in keys:
        for i, n in enumerate(low):
            if k in n:
                return i
    return 0


cols = list(raw.columns)
st.markdown("**Which column is which?**")
c = st.columns(6)
t_col = c[0].selectbox("Time", cols, index=guess(cols, "timestamp", "time", "date", "datetime", "gmt time"))
o_col = c[1].selectbox("Open", cols, index=guess(cols, "open", "o"))
h_col = c[2].selectbox("High", cols, index=guess(cols, "high", "h"))
l_col = c[3].selectbox("Low", cols, index=guess(cols, "low", "l"))
cl_col = c[4].selectbox("Close", cols, index=guess(cols, "close", "c", "last", "price"))
vol_opts = ["(no volume)"] + cols
v_guess = guess(cols, "volume", "vol")
has_vol = any("vol" in str(x).lower() for x in cols)
v_col = c[5].selectbox("Volume", vol_opts, index=(v_guess + 1) if has_vol else 0)

c = st.columns(3)
tz = c[0].selectbox("Times are in", ["UTC", "America/New_York", "America/Chicago", "Europe/London",
                                     "Europe/Berlin", "Asia/Tokyo", "Asia/Hong_Kong", "Australia/Sydney"],
                    help="Ignored if the times already include a timezone (like 2024-01-02T09:30:00-05:00).")
day_first = c[1].checkbox("Dates are day-first (31.12.2024)", value=False)

st.markdown("**About this market**")
c = st.columns(3)
name = c[0].text_input("Market name", placeholder="ETHUSD").strip().upper()
kind = c[1].radio("Type", ["crypto", "fx", "us_stocks", "other"],
                  format_func={"crypto": "Crypto (24/7)", "fx": "Forex (day starts 5pm New York)",
                               "us_stocks": "US stocks / index (regular hours)", "other": "Other 24h market"}.get)
c[2].caption("Trading costs (price units; commission in basis points per side):")
spread = c[2].number_input("Spread", 0.0, 1e6, 0.0, format="%.5f")
slippage = c[2].number_input("Slippage per fill", 0.0, 1e6, 0.0, format="%.5f")
commission = c[2].number_input("Commission (bps)", 0.0, 100.0, 0.0)

built_in = set(load_config()["assets"])
problems = []
if not re.fullmatch(r"[A-Z0-9_]{1,15}", name or ""):
    problems.append("Give it a name: letters, numbers or _, up to 15 characters.")
if name in built_in:
    problems.append(f"{name} is a built-in market. Pick another name.")
replacing = name in user
if replacing:
    st.warning(f"{name} already exists and will be replaced.")
for p in problems:
    st.caption(f"⚠️ {p}")

if not st.button("Add market", type="primary", disabled=bool(problems)):
    st.stop()


def parse_time(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):                # unix time: seconds, ms or µs
        mx = float(s.max())
        unit = "s" if mx < 1e11 else "ms" if mx < 1e14 else "us"
        return pd.to_datetime(s, unit=unit, utc=True)
    t = pd.to_datetime(s, dayfirst=day_first, format="mixed")
    return t.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC") if t.dt.tz is None \
        else t.dt.tz_convert("UTC")


try:
    with st.spinner("Building every timeframe…"):
        df = pd.DataFrame({"timestamp": parse_time(raw[t_col]),
                           "open": pd.to_numeric(raw[o_col], errors="coerce"),
                           "high": pd.to_numeric(raw[h_col], errors="coerce"),
                           "low": pd.to_numeric(raw[l_col], errors="coerce"),
                           "close": pd.to_numeric(raw[cl_col], errors="coerce"),
                           "volume": pd.to_numeric(raw[v_col], errors="coerce") if v_col != "(no volume)" else np.nan})
        df = df.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp")
        df = df.drop_duplicates("timestamp", keep="last")
        if len(df) < 200:
            raise ValueError(f"only {len(df)} usable rows; need at least 200")

        step = int(df["timestamp"].diff().dt.total_seconds().median() // 60)
        tfs = [tf for tf in cfg["timeframes"] if timeframe_minutes(tf) >= step and timeframe_minutes(tf) % max(step, 1) == 0]
        if step >= 1440 or not tfs:
            tfs = ["1d"]

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        path = UPLOAD_DIR / f"{name}.csv"
        out = df.assign(timestamp=df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00"))
        out.to_csv(path, index=False)

        session, klass = {"crypto": ("crypto_utc", "crypto"), "fx": ("fx_17ny", "fx"),
                          "us_stocks": ("rth_us", "equity"), "other": ("crypto_utc", "other")}[kind]
        block = {"class": klass, "session": session, "timeframes": tfs,
                 "source": {"kind": "csv_import", "path": str(path), "format": "generic"},
                 "volume": {"mode": "native", "kind": "real" if v_col != "(no volume)" else "none"},
                 "volume_gate": "off",
                 "costs": {"spread": spread, "slippage": slippage, "commission_bps": commission}}
        blocks = read_user_assets()
        previous = blocks.get(name)
        blocks[name] = block
        write_user_assets(blocks)
        try:
            shutil.rmtree(resolve_path(cfg, "store") / name, ignore_errors=True)
            meta = build_asset(load_cfg(), name, network=False, verbose=False)
        except Exception:
            # leave things as they were if the build fails
            if previous is None:
                blocks.pop(name, None)
            else:
                blocks[name] = previous
            write_user_assets(blocks)
            raise
    st.cache_data.clear()
    built = {tf: v.get("bars", 0) for tf, v in meta["frames"].items() if tf in cfg["timeframes"]}
    st.success(f"Added {name}: {len(df):,} bars of {step}-minute data, "
               + ", ".join(f"{tf} ({n:,})" for tf, n in built.items() if n) + ". Pick it on the Backtest page.")
except Exception as e:  # noqa: BLE001
    error_box(e)
