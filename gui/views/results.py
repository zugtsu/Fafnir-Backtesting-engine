"""Results page: every run in a table, overlaid equity curves, full detail on one run."""

import json

import pandas as pd
import streamlit as st

from gui.common import (drawdown_chart, equity_chart, error_box, features, fmt, load_cfg, load_saved_run,
                        metric_cards, price_chart, runs, save_run, saved_runs, trades_table)

st.title("Results")

with st.expander("Load saved runs"):
    paths = saved_runs()
    if not paths:
        st.caption("Nothing saved yet. Use **Save this run** below.")
    else:
        loaded = {r["id"] for r in runs()}
        pick = st.multiselect("Saved runs", paths, format_func=lambda p: p.name.removeprefix("gui_"))
        if st.button("Load", disabled=not pick):
            for p in pick:
                try:
                    r = load_saved_run(p)
                    if r["id"] not in loaded:
                        runs().append(r)
                except Exception as e:  # noqa: BLE001
                    error_box(e)
            st.rerun()

if not runs():
    st.info("No runs yet. Go to **Backtest** and press Run.")
    st.stop()

# --- comparison table ---------------------------------------------------------
st.subheader("All runs")
table = pd.DataFrame([{
    "run": r["label"], "market": r["settings"]["market"], "tf": r["settings"]["tf"],
    "dates": f"{r['settings']['start']} → {r['settings']['end']}",
    "trades": r["metrics"].get("trades", 0),
    "return": fmt(r["metrics"], "return_pct", "pct"), "buy & hold": fmt(r["metrics"], "buy_hold_pct", "pct"),
    "win rate": fmt(r["metrics"], "win_rate", "pct"), "profit factor": fmt(r["metrics"], "profit_factor"),
    "max drawdown": fmt(r["metrics"], "max_drawdown_pct", "pct"), "avg trade": fmt(r["metrics"], "expectancy_r", "r"),
    "sharpe": fmt(r["metrics"], "sharpe"),
} for r in runs()])
st.dataframe(table, width="stretch", hide_index=True)

labels = [f"{i + 1}. {r['label']}" for i, r in enumerate(runs())]
overlay = st.multiselect("Compare equity curves", labels, default=labels[-min(3, len(labels)):])
if overlay:
    items = [(lab, runs()[labels.index(lab)]["equity"]) for lab in overlay]
    st.plotly_chart(equity_chart(items, normalize=True), width="stretch")
    st.caption("Shown as % return so runs with different account sizes line up.")

c1, c2 = st.columns([3, 1])
if c2.button("Clear all runs"):
    st.session_state["runs"] = []
    st.rerun()

# --- one run in detail ------------------------------------------------------------
st.divider()
pick = st.selectbox("Run details", labels, index=len(labels) - 1)
r = runs()[labels.index(pick)]
s, m = r["settings"], r["metrics"]
st.subheader(r["label"])
st.caption(f"{s['market']} · {s['tf']} · {s['start']} → {s['end']} · {m.get('bars', 0):,} bars")
metric_cards(m)

more = st.columns(4)
more[0].metric("Longs / shorts", f"{m.get('longs', 0)} / {m.get('shorts', 0)}")
more[1].metric("Best / worst trade", f"{fmt(m, 'largest_win', 'usd')} / {fmt(m, 'largest_loss', 'usd')}")
more[2].metric("Longest win / loss streak", f"{m.get('max_win_streak', 0)} / {m.get('max_loss_streak', 0)}")
more[3].metric("Commissions", fmt(m, "commissions_usd", "usd"))

tab_eq, tab_px, tab_tr, tab_set = st.tabs(["Equity", "Price and trades", "Trades", "Settings"])
with tab_eq:
    st.plotly_chart(equity_chart([(r["label"], r["equity"])]), width="stretch")
    st.plotly_chart(drawdown_chart(r["equity"]), width="stretch")
with tab_px:
    try:
        close = features(load_cfg(), s["market"], s["tf"])["close"]
        close = close[(close.index >= pd.Timestamp(s["start"], tz="UTC")) &
                      (close.index < pd.Timestamp(s["end"], tz="UTC") + pd.Timedelta(days=1))]
        st.plotly_chart(price_chart(close, r["trades"]), width="stretch")
    except Exception as e:  # noqa: BLE001
        error_box(e)
with tab_tr:
    st.dataframe(trades_table(r["trades"]), width="stretch", hide_index=True)
    st.download_button("Download all trade columns (CSV)", r["trades"].to_csv(index=False).encode(),
                       file_name=f"{r['id']}_trades.csv", mime="text/csv", disabled=not len(r["trades"]))
with tab_set:
    st.code(json.dumps(s, indent=2, default=str), language="json")

if st.button("Save this run"):
    try:
        st.success(f"Saved to {save_run(r).name}")
    except Exception as e:  # noqa: BLE001
        error_box(e)
