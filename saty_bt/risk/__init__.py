"""
Position sizing and cost model.

Two sizing modes, per asset or overridden per run:
    fixed_fractional : risk `risk_pct` of equity to the initial stop
    fixed_margin     : `margin_usd` x `leverage` of notional per trade
                       (capped at current equity x leverage - you cannot
                       post margin you no longer have)

Quote-currency handling: for USD-quoted instruments (EURUSD, SPX, BTC,
equities) P&L in USD = units x price change. For JPY-quoted USDJPY the
units are USD and P&L in JPY is converted at the exit price.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Fill:
    price: float
    commission_usd: float


def cost_params(acfg: dict) -> dict:
    c = acfg.get("costs", {})
    return {"spread": float(c.get("spread", 0.0)), "slippage": float(c.get("slippage", 0.0)),
            "slippage_bps": float(c.get("slippage_bps", 0.0)),
            "commission_bps": float(c.get("commission_bps", 0.0)),
            "fee_fixed_usd": float(c.get("fee_fixed_usd", 0.0)),   # flat per transaction (e.g. gas)
            "holding_bps_per_hour": float(c.get("holding_bps_per_hour", 0.0)),   # funding/borrowing on notional while open
            # real FX rollover (saty_bt/data/rates.py); when "rate_diff" it REPLACES holding_bps_per_hour
            "financing": str(c.get("financing", "none")),
            "financing_markup_pct": float(c.get("financing_markup_pct", 0.0)),
            "financing_weekend": str(c.get("financing_weekend", "triple_wed")),
            "financing_day_count": int(c.get("financing_day_count", 365))}


def slip(price: float, cp: dict) -> float:
    return max(cp["slippage"], price * cp["slippage_bps"] / 1e4)


def fill_price(ref: float, direction: int, cp: dict, is_entry: bool) -> float:
    """Long entry / short exit pay up; short entry / long exit sell down."""
    adverse = direction if is_entry else -direction
    return ref + adverse * (slip(ref, cp) + cp["spread"] / 2.0)


def to_usd(amount_quote: float, price: float, quote_ccy: str) -> float:
    if quote_ccy == "USD":
        return amount_quote
    return amount_quote / price          # e.g. JPY -> USD at USDJPY


def notional_usd(units: float, price: float, quote_ccy: str) -> float:
    return units * price if quote_ccy == "USD" else units


def size_position(rcfg: dict, equity: float, entry: float, stop_dist: float,
                  quote_ccy: str) -> tuple[float, float, float]:
    """
    -> (units, notional_usd, risk_usd_to_stop). units are base units
    (EUR for EURUSD, USD for USDJPY, index points x1 for SPX, coins, shares).
    """
    mode = rcfg.get("sizing", "fixed_fractional")
    if equity <= 0 or entry <= 0:
        return 0.0, 0.0, 0.0
    if mode == "fixed_margin":
        lev = float(rcfg.get("leverage", 1))
        margin = min(float(rcfg.get("margin_usd", 0)), equity)
        notional = margin * lev
        units = notional / entry if quote_ccy == "USD" else notional
        risk = to_usd(units * stop_dist, entry, quote_ccy)
        return units, notional, risk
    # fixed fractional: risk_pct of equity to the stop, capped by leverage
    risk_usd = equity * float(rcfg.get("risk_pct", 1.0)) / 100.0
    if stop_dist <= 0:
        return 0.0, 0.0, 0.0
    risk_quote = risk_usd if quote_ccy == "USD" else risk_usd * entry
    units = risk_quote / stop_dist
    notional = notional_usd(units, entry, quote_ccy)
    cap = equity * float(rcfg.get("leverage", 1e9))
    if notional > cap:
        units *= cap / notional
        notional = cap
        risk_usd = to_usd(units * stop_dist, entry, quote_ccy)
    return units, notional, risk_usd
