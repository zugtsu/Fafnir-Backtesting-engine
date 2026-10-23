"""Indicators page: the built-in catalog, plus writing, checking and saving your own."""

import types

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from gui.common import (CUSTOM_DIR, FILTERS, SYSTEMS, TEMPLATE, check_module, custom_indicators, error_box,
                        features, glass_layout, load_cfg, market_rows, slugify, test_signal)

st.title("Indicators")
tab_mine, tab_write, tab_builtin = st.tabs(["My indicators", "Write or edit", "Built-in"])
mine = custom_indicators()

# --- yours ------------------------------------------------------------------------
with tab_mine:
    if not mine:
        st.info("You haven't added any yet. Use **Write or edit**.")
    for key, info in mine.items():
        with st.container(border=True, key=f"glass_{key}"):
            top = st.columns([4, 1, 1])
            top[0].markdown(f"**{info['name']}**  \n`custom_indicators/{info['path'].name}`")
            if top[1].button("Edit", key=f"edit_{key}"):
                st.session_state["ind_code"] = info["path"].read_text(encoding="utf-8")
                st.session_state["ind_file"] = info["path"].name
                st.session_state["ind_editor"] = st.session_state["ind_code"]
                st.toast("Loaded into the editor. Open the **Write or edit** tab.")
            if top[2].button("Delete", key=f"del_{key}"):
                info["path"].unlink(missing_ok=True)
                st.rerun()
            if info["error"]:
                st.error(f"Doesn't load: {info['error']}")
            else:
                if info["description"]:
                    st.caption(info["description"])
                if info["params"]:
                    st.caption("Settings: " + ", ".join(f"{k} = {v}" for k, v in info["params"].items()))

# --- editor -------------------------------------------------------------------------
with tab_write:
    st.markdown(
        "An indicator is a short Python function. It gets the price bars and returns, for every bar, "
        "**1** (want to be long), **-1** (want to be short) or **0** (stay out). The Backtest page handles "
        "entries, stops, exits, costs and sizing, the same way it does for the built-ins."
    )
    st.session_state.setdefault("ind_code", TEMPLATE)
    st.session_state.setdefault("ind_editor", st.session_state["ind_code"])
    st.session_state.setdefault("ind_file", None)
    c = st.columns([1, 1, 4])
    if c[0].button("New from template"):
        st.session_state.update(ind_code=TEMPLATE, ind_editor=TEMPLATE, ind_file=None)
        st.rerun()
    if st.session_state["ind_file"]:
        c[2].caption(f"Editing `custom_indicators/{st.session_state['ind_file']}`")

    code = st.text_area("Code", key="ind_editor", height=420, label_visibility="collapsed")

    markets = market_rows(load_cfg())
    c = st.columns([2, 1, 1, 2])
    names = [m["market"] for m in markets]
    t_market = c[0].selectbox("Check on", names, index=names.index("SPX") if "SPX" in names else 0) if names else None
    tfs = next((m["timeframes"] for m in markets if m["market"] == t_market), ["1h"])
    t_tf = c[1].selectbox("Timeframe", tfs, index=tfs.index("1h") if "1h" in tfs else 0)
    check = c[2].button("Check it", type="primary")
    save = c[3].button("Save")

    def compile_code(src: str):
        mod = types.ModuleType("custom_check")
        exec(compile(src, "<your indicator>", "exec"), mod.__dict__)  # noqa: S102 - local app, your own code
        check_module(mod)
        return mod

    if check:
        try:
            mod = compile_code(code)
            bars = features(load_cfg(), t_market, t_tf).iloc[-3000:]
            with st.spinner("Running it on real bars and checking for look-ahead…"):
                rep = test_signal(mod.signal, getattr(mod, "PARAMS", {}), bars)
            ok = st.columns(4)
            ok[0].metric("Bars long", f"{rep['long_bars']:,}")
            ok[1].metric("Bars short", f"{rep['short_bars']:,}")
            ok[2].metric("Bars out", f"{rep['flat_bars']:,}")
            ok[3].metric("Trades it would start", f"{rep['trades_started']:,}")
            if rep["causal"]:
                st.success("Passed the look-ahead check: it only uses bars up to the current one.")
            else:
                st.error("Failed the look-ahead check: its past values change when future bars are removed, "
                         "so it is peeking ahead (a .shift(-1), a centered window, or similar). Backtest results "
                         "would look better than reality.")
                st.caption(rep.get("causal_detail", ""))
            if rep["trades_started"] == 0:
                st.warning("It never signals on this data. Check the settings or the conditions.")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
            fig.add_trace(go.Scattergl(x=bars.index, y=bars["close"], name="Close", line=dict(width=1)), 1, 1)
            fig.add_trace(go.Scattergl(x=rep["series"].index, y=rep["series"], name="Signal", line=dict(shape="hv")), 2, 1)
            glass_layout(fig, 420, showlegend=False)
            st.plotly_chart(fig, width="stretch")
            st.caption(f"Last {len(bars):,} bars of {t_market} {t_tf}.")
        except Exception as e:  # noqa: BLE001
            error_box(e)

    if save:
        try:
            mod = compile_code(code)
            fname = st.session_state["ind_file"] or f"{slugify(getattr(mod, 'NAME', 'indicator'))}.py"
            CUSTOM_DIR.mkdir(exist_ok=True)
            (CUSTOM_DIR / fname).write_text(code, encoding="utf-8")
            st.session_state.update(ind_code=code, ind_file=fname)
            st.success(f"Saved as custom_indicators/{fname}. It's now in the Backtest page's indicator list (marked ⭐).")
        except Exception as e:  # noqa: BLE001
            error_box(e)

    with st.expander("What's in df?"):
        st.markdown(
            "- `open`, `high`, `low`, `close`: prices. `volume`: NaN when the data has none (most forex).\n"
            "- `chart_atr`: average bar size (14-bar ATR), handy for bands and distances.\n"
            "- The index is the bar's open time in UTC, so `df.index.hour` gives the hour.\n"
            "- `p` is the `PARAMS` dict, with whatever values you set on the Backtest page.\n\n"
            "pandas rolling windows, `.ewm()`, `.shift(1)` and `numpy` are all fine. Never `.shift(-1)` or "
            "`center=True`: those read the future. **Check it** catches that."
        )

# --- built in ---------------------------------------------------------------------------
with tab_builtin:
    st.caption("All of these share the same exits: a stop a set number of ATRs away, and an exit when the "
               "signal turns (both adjustable on the Backtest page).")
    for key, (name, style, what, specs) in SYSTEMS.items():
        with st.expander(f"{name} · {style}"):
            st.write(what)
            if specs:
                st.caption("Settings: " + ", ".join(f"{s['label']} ({s['default']})" for s in specs.values()))
    st.subheader("Filters")
    st.caption("A second indicator that has to agree before a trade is allowed.")
    for key, (name, specs) in FILTERS.items():
        if key != "none":
            st.markdown(f"- {name}" + (f" ({', '.join(s['label'] for s in specs.values())})" if specs else ""))
