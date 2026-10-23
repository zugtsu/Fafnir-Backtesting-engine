"""Backtest page: every setting in the sidebar, the latest result in the main area."""

from datetime import date

import pandas as pd
import streamlit as st

from gui.common import (FILTERS, SYSTEMS, custom_indicators, equity_chart, error_box, features,
                        indicator_label, load_cfg, market_rows, metric_cards, price_chart, run, runs,
                        trades_table)

cfg = load_cfg()
markets = market_rows(cfg)
st.title("Backtest")
if not markets:
    st.warning("No market data yet. Add some on the **Data** page.")
    st.stop()


def number(spec: dict, key: str):
    """A number box from a catalog spec; ints stay ints so lengths aren't 14.0."""
    is_int = isinstance(spec["default"], int) and isinstance(spec["step"], int)
    cast = int if is_int else float
    return st.number_input(spec["label"], min_value=cast(spec["min"]), max_value=cast(spec["max"]),
                           value=cast(spec["default"]), step=cast(spec["step"]), key=key, help=spec.get("help") or None)


sb = st.sidebar

# --- market and dates -------------------------------------------------------
sb.header("Market")
names = [m["market"] for m in markets]
market = sb.selectbox("Market", names, index=names.index("SPX") if "SPX" in names else 0)
row = next(m for m in markets if m["market"] == market)
tf = sb.selectbox("Timeframe", row["timeframes"], index=row["timeframes"].index("1h") if "1h" in row["timeframes"] else 0,
                  help="Bar size. Each bar's close is when signals are checked.")
first, last = date.fromisoformat(row["from"]), date.fromisoformat(row["to"])
dates = sb.date_input("Dates", value=(first, last), min_value=first, max_value=last)
start, end = (dates if isinstance(dates, tuple) and len(dates) == 2 else (first, last))

# --- indicators -------------------------------------------------------------------
sb.header("Indicators")
custom = custom_indicators()
choices = list(SYSTEMS) + [f"custom:{k}" for k, v in custom.items() if not v["error"]]
# A different starting pick for each slot. Not MACD zero next to the MA cross:
# with the same lengths they are the same signal and would always agree.
DEFAULT_PICKS = ["ma_cross", "rsi_trend", "supertrend", "adx_di", "aroon"]


def indicator_inputs(i: int) -> dict:
    """One indicator slot: its picker and only the settings it uses. Widget keys carry the slot number."""
    default = DEFAULT_PICKS[i] if DEFAULT_PICKS[i] in choices else choices[0]
    ind = st.selectbox("Indicator", choices, index=choices.index(default), key=f"ind_{i}",
                       format_func=lambda k: ("⭐ " if k.startswith("custom:") else "") + indicator_label(k))
    params = {}
    if ind.startswith("custom:"):
        info = custom[ind.split(":", 1)[1]]
        if info["description"]:
            st.caption(info["description"])
        for k, v in info["params"].items():
            key = f"c{i}_{ind}_{k}"
            if isinstance(v, bool):
                params[k] = st.checkbox(k, value=v, key=key)
            elif isinstance(v, int):
                params[k] = int(st.number_input(k, value=v, step=1, key=key))
            elif isinstance(v, float):
                params[k] = float(st.number_input(k, value=v, key=key))
            else:
                params[k] = st.text_input(k, value=str(v), key=key)
    else:
        _, style, what, specs = SYSTEMS[ind]
        st.caption(f"**{style}.** {what}")
        for k, spec in specs.items():
            params[k] = number(spec, f"p{i}_{ind}_{k}")
        if ind == "ma_cross":
            params["ma_kind"] = st.radio("Average type", ["ema", "sma"], horizontal=True, format_func=str.upper,
                                         key=f"mak_{i}")
    return {"indicator": ind, "params": params}


count = int(sb.selectbox("How many indicators", [1, 2, 3, 4, 5],
                         help="Combine up to five. Each one gets its own settings below."))
legs = []
for i in range(count):
    with sb.expander(f"Indicator {i + 1}" if count > 1 else "Entry signal", expanded=True):
        legs.append(indicator_inputs(i))

combine, need = "all", count
if count > 1:
    combine = sb.radio("Combine them", ["all", "at_least", "trigger"], key="combine", format_func={
        "all": "All must agree", "at_least": "At least some agree",
        "trigger": "#1 triggers, the rest confirm"}.get)
    if combine == "at_least":
        need = sb.slider("How many must agree", 1, count, value=max(2, (count + 1) // 2) if count > 1 else 1)
    sb.caption({
        "all": "Long only while every indicator says long (short likewise). The trade closes when they stop agreeing.",
        "at_least": f"Long while at least {need} of {count} say long, and more say long than short.",
        "trigger": "Indicator 1 decides when to enter and exit; the others only have to agree at the moment it fires.",
    }[combine])

direction = sb.radio("Trade", ["both", "long", "short"], horizontal=True,
                     format_func={"both": "Both ways", "long": "Long only", "short": "Short only"}.get)

with sb.expander("Filter (optional)"):
    filt = st.selectbox("Filter", list(FILTERS), format_func=lambda k: FILTERS[k][0], label_visibility="collapsed")
    filter_params = {k: number(spec, f"f_{filt}_{k}") for k, spec in FILTERS[filt][1].items()}

with sb.expander("Exits"):
    stop_atr_k = st.number_input("Stop loss (ATRs from entry)", 0.5, 10.0, 2.0, 0.25,
                                 help="ATR = the average bar size. 2 means the stop sits two average bars away.")
    exit_on_flip = st.checkbox("Close when the signal turns against the trade", value=True)
    target_r = st.number_input("Take profit at (× the risk, 0 = off)", 0.0, 20.0, 0.0, 0.5)

# --- money --------------------------------------------------------------------
with sb.expander("Account and costs"):
    equity = st.number_input("Starting account ($)", 100.0, 10_000_000.0, 10_000.0, 100.0)
    sizing = st.radio("Position size", ["fixed_fractional", "fixed_margin"],
                      format_func={"fixed_fractional": "Risk % per trade", "fixed_margin": "Fixed margin × leverage"}.get)
    risk_pct = st.number_input("Risk per trade (%)", 0.1, 10.0, 1.0, 0.1, disabled=sizing != "fixed_fractional")
    margin_usd = st.number_input("Margin per trade ($)", 10.0, 1_000_000.0, 200.0, 10.0, disabled=sizing != "fixed_margin")
    leverage = st.number_input("Leverage", 1.0, 100.0, 5.0, 1.0, disabled=sizing != "fixed_margin")
    acosts = cfg["assets"][market].get("costs", {})
    st.caption("Costs for this market (price units, commission in basis points):")
    spread = st.number_input("Spread", 0.0, 1000.0, float(acosts.get("spread", 0.0)), format="%.5f")
    slippage = st.number_input("Slippage per fill", 0.0, 1000.0, float(acosts.get("slippage", 0.0)), format="%.5f")
    commission = st.number_input("Commission (bps per side)", 0.0, 100.0, float(acosts.get("commission_bps", 0.0)))
    safety = st.checkbox("Use the engine's safety stops (daily loss limit, 15% drawdown halt)", value=False,
                         help="Off = the test runs the whole period even after a bad streak, which is usually what you want for research.")

label = sb.text_input("Name this run (optional)")
go_run = sb.button("Run backtest", type="primary", width="stretch")

# --- run ------------------------------------------------------------------------
if go_run:
    settings = {"market": market, "tf": tf, "start": str(start), "end": str(end),
                "legs": legs, "combine": combine, "need": need,
                "indicator": legs[0]["indicator"], "params": legs[0]["params"],
                "direction": direction, "filter": filt, "filter_params": filter_params,
                "stop_atr_k": stop_atr_k, "exit_on_flip": exit_on_flip, "target_r": target_r,
                "equity": equity, "sizing": sizing, "risk_pct": risk_pct, "margin_usd": margin_usd,
                "leverage": leverage, "safety_limits": safety, "label": label.strip(),
                "costs": {"spread": spread, "slippage": slippage, "commission_bps": commission}}
    try:
        with st.spinner("Running…"):
            r = run(settings)
        runs().append(r)
    except Exception as e:  # noqa: BLE001
        error_box(e)

if not runs():
    st.info("Pick your settings on the left and press **Run backtest**. Every run is kept on the **Results** page.")
    st.stop()

r = runs()[-1]
m = r["metrics"]
st.subheader(r["label"])
s = r["settings"]
st.caption(f"{s['market']} · {s['tf']} · {s['start']} → {s['end']} · {m.get('bars', 0):,} bars · "
           f"buy and hold: {m.get('buy_hold_pct', 0):+.1f}%")
metric_cards(m)
st.plotly_chart(equity_chart([(r["label"], r["equity"])]), width="stretch")

close = features(cfg, s["market"], s["tf"])["close"]
close = close[(close.index >= pd.Timestamp(s["start"], tz="UTC")) &
              (close.index < pd.Timestamp(s["end"], tz="UTC") + pd.Timedelta(days=1))]
st.plotly_chart(price_chart(close, r["trades"]), width="stretch")
with st.expander(f"Trades ({len(r['trades'])})"):
    st.dataframe(trades_table(r["trades"]), width="stretch", hide_index=True)
st.caption("Compare runs, save them and download trades on the **Results** page.")
