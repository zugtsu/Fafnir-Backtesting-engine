"""
Shared pieces for the Streamlit GUI: config with uploaded markets merged in,
the indicator catalog, custom-indicator loading, the backtest runner, charts.

Everything here calls the real engine (saty_bt); the GUI adds no trading logic
of its own.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saty_bt.config import asset_config, load_config, resolve_path, seed_everything, timeframe_minutes  # noqa: E402
from saty_bt.data.store import list_assets, load_meta  # noqa: E402
from saty_bt.engine import run_backtest  # noqa: E402
from saty_bt.indicators import build_features  # noqa: E402
from saty_bt.reports.metrics import summarize  # noqa: E402
from saty_bt.strategies import STRATEGIES  # noqa: E402

# Uploaded price history and the markets made from it live here (gitignored).
USER_DIR = ROOT / "user_data"
UPLOAD_DIR = USER_DIR / "uploads"
USER_ASSETS = USER_DIR / "user_assets.yaml"
# Custom indicators: one .py file each.
CUSTOM_DIR = ROOT / "custom_indicators"


# ---------------------------------------------------------------------------
# config and markets
# ---------------------------------------------------------------------------

def read_user_assets() -> dict:
    if not USER_ASSETS.exists():
        return {}
    return yaml.safe_load(USER_ASSETS.read_text(encoding="utf-8")) or {}


def write_user_assets(blocks: dict) -> None:
    USER_DIR.mkdir(parents=True, exist_ok=True)
    USER_ASSETS.write_text(yaml.safe_dump(blocks, sort_keys=True), encoding="utf-8")


def load_cfg() -> dict:
    """config.yaml with the markets added from the Data page merged into `assets`."""
    cfg = load_config()
    cfg["assets"].update(read_user_assets())
    return cfg


def market_rows(cfg: dict) -> list[dict]:
    """Every market that has data built, with what the Data page shows."""
    user = read_user_assets()
    rows = []
    for name in list_assets(cfg):
        if name not in cfg["assets"]:
            continue                                  # proxy / auxiliary series, not tradable
        meta = load_meta(cfg, name)
        frames = {tf: v for tf, v in meta.get("frames", {}).items()
                  if tf in cfg["timeframes"] and v.get("bars")}
        if not frames:
            continue
        span = meta["frames"].get("1d") or next(iter(frames.values()))
        kind = user[name]["source"]["kind"] if name in user else ""
        source = "built-in" if name not in user else "uploaded" if kind == "csv_import" else "downloaded"
        rows.append({"market": name, "source": source,
                     "type": cfg["assets"][name].get("class", ""),
                     "timeframes": [tf for tf in cfg["timeframes"] if tf in frames],
                     "from": str(span.get("first", ""))[:10], "to": str(span.get("last", ""))[:10]})
    return rows


@st.cache_data(show_spinner=False, max_entries=24)
def _features_cached(asset: str, tf: str, stamp: float) -> pd.DataFrame:
    return build_features(load_cfg(), asset, tf)


def features(cfg: dict, asset: str, tf: str) -> pd.DataFrame:
    # Re-reads when the market's data is rebuilt (its meta.json changes).
    meta = resolve_path(cfg, "store") / asset / "meta.json"
    return _features_cached(asset, tf, meta.stat().st_mtime if meta.exists() else 0.0)


# ---------------------------------------------------------------------------
# indicator catalog
# ---------------------------------------------------------------------------

def P(label, default, lo, hi, step, help=""):
    return {"label": label, "default": default, "min": lo, "max": hi, "step": step, "help": help}


# key -> (name, style, what it does, {setting: spec}). Settings map 1:1 onto
# strategies.indicator_lab in config.yaml.
SYSTEMS: dict[str, tuple[str, str, str, dict]] = {
    "ma_cross": ("Moving average cross", "Trend",
                 "Long when the fast average crosses above the slow one, short on the cross below.",
                 {"n_fast": P("Fast length", 12, 2, 200, 1), "n_slow": P("Slow length", 26, 3, 400, 1)}),
    "macd_cross": ("MACD signal cross", "Trend",
                   "Long when the MACD line crosses above its signal line.",
                   {"n_fast": P("Fast EMA", 12, 2, 100, 1), "n_slow": P("Slow EMA", 26, 3, 200, 1),
                    "n_signal": P("Signal EMA", 9, 2, 50, 1)}),
    "macd_zero": ("MACD zero cross", "Trend", "Long when the MACD line crosses above zero.",
                  {"n_fast": P("Fast EMA", 12, 2, 100, 1), "n_slow": P("Slow EMA", 26, 3, 200, 1)}),
    "rsi_revert": ("RSI oversold / overbought", "Mean reversion",
                   "Buy when RSI climbs back above oversold, sell when it falls back below overbought. Exits at 50.",
                   {"n": P("RSI length", 14, 2, 100, 1), "lo": P("Oversold", 30.0, 5.0, 50.0, 1.0),
                    "hi": P("Overbought", 70.0, 50.0, 95.0, 1.0)}),
    "rsi_trend": ("RSI crosses 50", "Momentum", "Long above 50, short below.",
                  {"n": P("RSI length", 14, 2, 100, 1)}),
    "stoch": ("Stochastic cross", "Mean reversion",
              "%K crosses %D inside the oversold or overbought zone.",
              {"n": P("%K length", 14, 3, 100, 1), "n_signal": P("%D length", 3, 1, 20, 1),
               "lo": P("Oversold", 20.0, 5.0, 50.0, 1.0), "hi": P("Overbought", 80.0, 50.0, 95.0, 1.0)}),
    "bb_revert": ("Bollinger band fade", "Mean reversion",
                  "Buy a close back inside the lower band, sell one back inside the upper band.",
                  {"n": P("Length", 20, 5, 200, 1), "k_band": P("Band width (std devs)", 2.0, 0.5, 4.0, 0.1)}),
    "bb_break": ("Bollinger band breakout", "Breakout", "Follow a close outside the band.",
                 {"n": P("Length", 20, 5, 200, 1), "k_band": P("Band width (std devs)", 2.0, 0.5, 4.0, 0.1)}),
    "keltner_break": ("Keltner channel breakout", "Breakout", "Follow a close outside an ATR channel.",
                      {"n": P("EMA length", 20, 5, 200, 1), "k_band": P("Channel width (ATRs)", 2.0, 0.5, 5.0, 0.1)}),
    "donchian": ("Donchian breakout (Turtle)", "Breakout",
                 "Buy a new N-bar high, exit on the opposite exit-length low.",
                 {"n": P("Entry length", 20, 5, 200, 1), "exit_n": P("Exit length", 10, 2, 100, 1)}),
    "cci": ("CCI ±100", "Momentum", "Long when CCI crosses above +100, short below -100.",
            {"n": P("CCI length", 20, 5, 100, 1)}),
    "williams_r": ("Williams %R", "Mean reversion", "Trade %R leaving the -80 / -20 zones.",
                   {"n": P("Length", 14, 3, 100, 1)}),
    "psar": ("Parabolic SAR", "Trend", "Follow the SAR flips.",
             {"psar_step": P("Step", 0.02, 0.005, 0.1, 0.005), "psar_cap": P("Max step", 0.2, 0.05, 0.5, 0.01)}),
    "supertrend": ("Supertrend", "Trend", "Follow the ATR band flips.",
                   {"n": P("ATR length", 10, 3, 100, 1), "k_band": P("Multiplier", 3.0, 0.5, 6.0, 0.1)}),
    "ichimoku": ("Ichimoku", "Trend", "Tenkan/Kijun cross, only on the right side of the cloud.",
                 {"n_fast": P("Tenkan", 9, 2, 60, 1), "n_slow": P("Kijun", 26, 5, 120, 1),
                  "n_spanb": P("Senkou B", 52, 10, 240, 1)}),
    "adx_di": ("ADX / DI cross", "Trend", "+DI crosses -DI while ADX shows a trend.",
               {"adx_min": P("Min ADX", 20.0, 5.0, 50.0, 1.0)}),
    "aroon": ("Aroon cross", "Trend", "Aroon up crosses Aroon down.", {"n": P("Length", 25, 5, 100, 1)}),
    "roc": ("Rate of change", "Momentum", "Long when price is above where it was N bars ago.",
            {"n": P("Lookback", 12, 2, 200, 1)}),
    "smi_cross": ("SMI signal cross", "Momentum", "Stochastic Momentum Index crosses its signal line.",
                  {"n": P("Length", 10, 3, 60, 1), "n_fast": P("Smoothing 1", 3, 1, 20, 1),
                   "n_slow": P("Smoothing 2", 3, 1, 20, 1), "n_signal": P("Signal", 3, 1, 20, 1)}),
    "ribbon_stack": ("EMA ribbon stack", "Trend", "Long when fast > mid > slow EMA, short when reversed.",
                     {"n_fast": P("Fast EMA", 8, 2, 100, 1), "n_mid": P("Mid EMA", 21, 3, 200, 1),
                      "n_slow": P("Slow EMA", 34, 5, 400, 1)}),
}

# key -> (name, settings). Settings use the lab's filt_* keys.
FILTERS: dict[str, tuple[str, dict]] = {
    "none": ("No filter", {}),
    "ema_trend": ("Only trade with the EMA trend", {"filt_n": P("EMA length", 200, 5, 500, 1)}),
    "sma_trend": ("Only trade with the SMA trend", {"filt_n": P("SMA length", 200, 5, 500, 1)}),
    "macd": ("Only trade with MACD's side of zero",
             {"filt_n_fast": P("Fast EMA", 12, 2, 100, 1), "filt_n_slow": P("Slow EMA", 26, 3, 200, 1)}),
    "supertrend": ("Only trade with Supertrend",
                   {"filt_n": P("ATR length", 10, 3, 100, 1), "filt_k": P("Multiplier", 3.0, 0.5, 6.0, 0.1)}),
    "rsi50": ("Only trade with RSI's side of 50", {"filt_n": P("RSI length", 14, 2, 100, 1)}),
    "ema_stack": ("Only trade when 20 > 50 > 100 > 200 EMAs line up", {}),
    "adx_min": ("Only trade when ADX shows a trend", {"filt_v": P("Min ADX", 20.0, 5.0, 50.0, 1.0)}),
    "adx_max": ("Only trade when ADX shows a range", {"filt_v": P("Max ADX", 20.0, 5.0, 50.0, 1.0)}),
}


# ---------------------------------------------------------------------------
# custom indicators
# ---------------------------------------------------------------------------

TEMPLATE = '''"""Describe your idea here."""
import pandas as pd

NAME = "My indicator"
DESCRIPTION = "Long when price is above its EMA, short when below."
# Every setting here gets a box on the Backtest page. Use whole numbers for lengths.
PARAMS = {"length": 50}


def signal(df: pd.DataFrame, p: dict) -> pd.Series:
    """
    df has one row per bar: open, high, low, close, volume (NaN if the data has none),
    plus chart_atr. Return a Series with, for every bar:
         1 = want to be long,  -1 = want to be short,  0 = stay out.
    A trade opens at the next bar's open after the value changes and closes when it
    changes back. Only look at the current and earlier bars: never .shift(-1).
    """
    ema = df["close"].ewm(span=p["length"], adjust=False).mean()
    return (df["close"] > ema).astype(int) - (df["close"] < ema).astype(int)
'''


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return s[:48] or "indicator"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"custom_ind_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_module(mod) -> None:
    if not callable(getattr(mod, "signal", None)):
        raise ValueError("needs a function: def signal(df, p)")
    if not isinstance(getattr(mod, "PARAMS", {}), dict):
        raise ValueError("PARAMS must be a dict like {\"length\": 14}")


def custom_indicators() -> dict[str, dict]:
    """file stem -> {name, description, params, fn, path, error}."""
    out = {}
    if not CUSTOM_DIR.exists():
        return out
    for path in sorted(CUSTOM_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue
        info = {"path": path, "name": path.stem, "description": "", "params": {}, "fn": None, "error": None}
        try:
            mod = load_module(path)
            check_module(mod)
            info.update(name=getattr(mod, "NAME", path.stem), description=getattr(mod, "DESCRIPTION", ""),
                        params=dict(getattr(mod, "PARAMS", {})), fn=mod.signal)
        except Exception as e:  # noqa: BLE001
            info["error"] = f"{type(e).__name__}: {e}"
        out[path.stem] = info
    return out


def test_signal(fn, params: dict, bars: pd.DataFrame) -> dict:
    """Run a custom signal on real bars: counts, and the engine's look-ahead check."""
    from saty_bt.validate import assert_causal

    out = fn(bars.copy(), dict(params))
    s = out.reindex(bars.index) if isinstance(out, pd.Series) else pd.Series(np.asarray(out, dtype=float), index=bars.index)
    if len(s) != len(bars):
        raise ValueError(f"returned {len(s)} values for {len(bars)} bars")
    s = np.sign(s.astype(float)).fillna(0.0)
    report = {"long_bars": int((s > 0).sum()), "short_bars": int((s < 0).sum()), "flat_bars": int((s == 0).sum()),
              "trades_started": int(((s != 0) & (s != s.shift(1))).sum()), "series": s}
    try:
        assert_causal(lambda d: d.assign(_x=pd.Series(np.asarray(fn(d.copy(), dict(params)), dtype=float), index=d.index)),
                      bars, ["_x"], name="custom indicator")
        report["causal"] = True
    except AssertionError as e:
        report["causal"] = False
        report["causal_detail"] = str(e)
    return report


# ---------------------------------------------------------------------------
# running a backtest
# ---------------------------------------------------------------------------

def run(settings: dict) -> dict:
    """
    One indicator-lab backtest from GUI settings. Signals are built on the full
    history (so indicators are warmed up) and the engine trades only the chosen
    date window.
    """
    cfg = load_cfg()
    seed_everything(cfg)
    asset, tf = settings["market"], settings["tf"]
    acfg = asset_config(cfg, asset)
    acfg["costs"] = {**acfg.get("costs", {}), **settings.get("costs", {})}

    scfg = copy.deepcopy(cfg["strategies"]["indicator_lab"])
    legs = [leg_config(leg) for leg in run_legs(settings)]
    if len(legs) == 1:
        scfg.update(legs[0])
    else:
        scfg.update(system="combo", combo=legs, combo_mode=settings.get("combine", "all"),
                    combo_need=int(settings.get("need", len(legs))))
    scfg.update(filt_kind=settings["filter"], **settings["filter_params"])
    scfg.update(stop_atr_k=settings["stop_atr_k"], exit_on_flip=settings["exit_on_flip"],
                target_r=settings["target_r"])

    rcfg = {"sizing": settings["sizing"], "risk_pct": settings["risk_pct"],
            "margin_usd": settings["margin_usd"], "leverage": settings["leverage"]}
    limits = {} if settings["safety_limits"] else {
        "halt_drawdown_pct": 100, "daily_loss_limit_pct": 100, "consecutive_loss_limit": 10**9}

    f = features(cfg, asset, tf)
    sig = STRATEGIES["indicator_lab"](f, scfg, acfg, cfg)
    if settings.get("direction") == "long":
        sig["short_entry"] = False
    elif settings.get("direction") == "short":
        sig["long_entry"] = False

    start = pd.Timestamp(settings["start"], tz="UTC")
    end = pd.Timestamp(settings["end"], tz="UTC") + pd.Timedelta(days=1)
    win = (f.index >= start) & (f.index < end)
    if win.sum() < 50:
        raise ValueError("fewer than 50 bars in that date range")
    fw, sw = f.loc[win], sig.loc[win]
    trades, eq = run_backtest(fw, sw, acfg, cfg, scfg, rcfg, limits, settings["equity"])
    m = summarize(trades, eq)
    m["buy_hold_pct"] = float((fw["close"].iloc[-1] / fw["close"].iloc[0] - 1) * 100)
    m["bars"] = int(len(fw))

    label = settings.get("label") or f"{run_title(settings)} · {asset} {tf}"
    return {"id": datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"), "label": label,
            "settings": settings, "metrics": m, "trades": trades, "equity": eq}


COMBINE_NAMES = {"all": "all agree", "at_least": "{need} of {n} agree", "trigger": "#1 triggers"}


def run_legs(settings: dict) -> list[dict]:
    """The run's indicators; runs saved before combining existed have a single `indicator`."""
    return settings.get("legs") or [{"indicator": settings["indicator"], "params": settings["params"]}]


def leg_config(leg: dict) -> dict:
    """One GUI indicator slot as indicator_lab settings."""
    if leg["indicator"].startswith("custom:"):
        info = custom_indicators()[leg["indicator"].split(":", 1)[1]]
        if info["error"]:
            raise RuntimeError(f"custom indicator {info['name']} failed to load: {info['error']}")
        return {"system": "custom", "custom_fn": info["fn"], "custom_params": dict(leg["params"])}
    return {"system": leg["indicator"], **leg["params"]}


def run_title(settings: dict) -> str:
    legs = run_legs(settings)
    names = " + ".join(indicator_label(leg["indicator"]) for leg in legs)
    if len(legs) == 1:
        return names
    how = COMBINE_NAMES.get(settings.get("combine", "all"), "").format(need=settings.get("need"), n=len(legs))
    return f"{names} ({how})"


def indicator_label(key: str) -> str:
    if key.startswith("custom:"):
        info = custom_indicators().get(key.split(":", 1)[1])
        return info["name"] if info else key
    return SYSTEMS[key][0] if key in SYSTEMS else key


def runs() -> list[dict]:
    return st.session_state.setdefault("runs", [])


# ---------------------------------------------------------------------------
# saving runs
# ---------------------------------------------------------------------------

def save_run(r: dict) -> Path:
    cfg = load_cfg()
    out = resolve_path(cfg, "results") / f"gui_{r['id']}_{slugify(r['label'])}"
    out.mkdir(parents=True, exist_ok=True)
    r["trades"].to_csv(out / "trades.csv", index=False)
    r["equity"].to_frame().to_parquet(out / "equity.parquet")
    meta = {"id": r["id"], "label": r["label"], "settings": r["settings"], "metrics": r["metrics"]}
    (out / "gui_run.json").write_text(json.dumps(meta, indent=2, default=_json_default), encoding="utf-8")
    return out


def saved_runs() -> list[Path]:
    root = resolve_path(load_cfg(), "results")
    return sorted((p for p in root.glob("gui_*") if (p / "gui_run.json").exists()), reverse=True)


def load_saved_run(path: Path) -> dict:
    meta = json.loads((path / "gui_run.json").read_text(encoding="utf-8"))
    trades = pd.read_csv(path / "trades.csv") if (path / "trades.csv").stat().st_size > 2 else pd.DataFrame()
    eq = pd.read_parquet(path / "equity.parquet").iloc[:, 0]
    return {**meta, "trades": trades, "equity": eq}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    return str(o)


# ---------------------------------------------------------------------------
# display
# ---------------------------------------------------------------------------

def fmt(m: dict, key: str, kind: str = "num") -> str:
    v = m.get(key)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if kind == "pct":
        return f"{v:+.1f}%" if key in ("return_pct", "buy_hold_pct") else f"{v:.1f}%"
    if kind == "usd":
        # "\$" because Streamlit reads a pair of $ in one label as a math formula
        return f"{'-' if v < 0 else ''}\\${abs(v):,.0f}"
    if kind == "r":
        return f"{v:+.2f}R"
    return f"{v:,.2f}" if isinstance(v, float) else f"{v:,}"


def metric_cards(m: dict) -> None:
    cols = st.columns(4)
    cols[0].metric("Return", fmt(m, "return_pct", "pct"), help=f"Buy and hold over the same dates: {fmt(m, 'buy_hold_pct', 'pct')}")
    cols[1].metric("Win rate", fmt(m, "win_rate", "pct"), help=f"{m.get('wins', 0)} wins / {m.get('losses', 0)} losses")
    cols[2].metric("Profit factor", fmt(m, "profit_factor"), help="Gross profit ÷ gross loss. Above 1 = made money.")
    cols[3].metric("Max drawdown", fmt(m, "max_drawdown_pct", "pct"), help="Worst peak-to-trough drop in the account")
    cols = st.columns(4)
    cols[0].metric("Trades", fmt(m, "trades"))
    cols[1].metric("Net P&L", fmt(m, "net_pnl_usd", "usd"))
    cols[2].metric("Avg trade", fmt(m, "expectancy_r", "r"), help="Average result per trade, in units of the risk taken")
    cols[3].metric("Sharpe", fmt(m, "sharpe"))


def glass_layout(fig: go.Figure, height: int, **kw) -> go.Figure:
    """Transparent chart so it sits on the glass card behind it, with faint grid lines."""
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"), template="plotly_dark",
                      colorway=["#A78BFA", "#22D3EE", "#F472B6", "#34D399", "#FBBF24"],
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      hoverlabel=dict(bgcolor="rgba(20,22,40,0.85)", bordercolor="rgba(255,255,255,0.2)"), **kw)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.07)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.07)", zerolinecolor="rgba(255,255,255,0.15)")
    return fig


_layout = glass_layout


def equity_chart(items: list[tuple[str, pd.Series]], normalize: bool = False, height: int = 340) -> go.Figure:
    fig = go.Figure()
    for name, eq in items:
        y = (eq / eq.iloc[0] - 1) * 100 if normalize else eq
        fig.add_trace(go.Scattergl(x=eq.index, y=y, mode="lines", name=name))
    fig.update_yaxes(title="Return %" if normalize else "Account $")
    return _layout(fig, height)


def drawdown_chart(eq: pd.Series, height: int = 180) -> go.Figure:
    dd = (eq / eq.cummax() - 1) * 100
    fig = go.Figure(go.Scattergl(x=dd.index, y=dd, mode="lines", fill="tozeroy", name="Drawdown",
                                 line=dict(color="#E24B4A")))
    fig.update_yaxes(title="Drawdown %")
    return _layout(fig, height)


def price_chart(close: pd.Series, trades: pd.DataFrame, height: int = 460) -> go.Figure:
    fig = go.Figure(go.Scattergl(x=close.index, y=close, mode="lines", name="Close",
                                 line=dict(color="rgba(226,232,240,0.75)", width=1)))
    if len(trades):
        t = trades.copy()
        for c in ("entry_time", "exit_time"):
            t[c] = pd.to_datetime(t[c], utc=True)
        for side, color, sym in (("long", "#22c55e", "triangle-up"), ("short", "#ef4444", "triangle-down")):
            s = t[t["direction"] == side]
            if len(s):
                fig.add_trace(go.Scattergl(x=s["entry_time"], y=s["entry_price"], mode="markers", name=f"{side} entry",
                                           marker=dict(symbol=sym, size=9, color=color)))
        win = t["pnl_usd"] > 0
        fig.add_trace(go.Scattergl(x=t["exit_time"], y=t["exit_price"], mode="markers", name="exit",
                                   marker=dict(symbol="x", size=7, color=np.where(win, "#86efac", "#fca5a5")),
                                   text=[f"{r:+.2f}R" for r in t["r_multiple"]], hovertemplate="exit %{y}<br>%{text}"))
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.06))
    return _layout(fig, height)


TRADE_COLS = ["direction", "entry_time", "entry_price", "exit_time", "exit_price", "exit_reason",
              "bars_held", "pnl_usd", "r_multiple", "equity_after"]


def trades_table(trades: pd.DataFrame) -> pd.DataFrame:
    if not len(trades):
        return trades
    t = trades[[c for c in TRADE_COLS if c in trades.columns]].copy()
    return t.rename(columns={"direction": "side", "entry_time": "entry", "exit_time": "exit",
                             "exit_reason": "why it closed", "bars_held": "bars", "pnl_usd": "P&L $",
                             "r_multiple": "R", "equity_after": "account after"})


def error_box(e: Exception) -> None:
    st.error(f"{type(e).__name__}: {e}")
    with st.expander("Details"):
        st.code("".join(traceback.format_exception(e)))
