"""Example custom indicator: buy dips in an uptrend, sell rallies in a downtrend."""
import pandas as pd

NAME = "EMA trend + RSI pullback"
DESCRIPTION = ("Long when price is above the trend EMA and RSI dips below the pullback level; "
               "short in the mirror case. Stays in until RSI recovers past 50.")
PARAMS = {"trend_length": 100, "rsi_length": 14, "pullback": 40}


def _rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def signal(df: pd.DataFrame, p: dict) -> pd.Series:
    trend = df["close"].ewm(span=p["trend_length"], adjust=False).mean()
    rsi = _rsi(df["close"], p["rsi_length"])
    up, down = df["close"] > trend, df["close"] < trend
    want = pd.Series(float("nan"), index=df.index)
    want[up & (rsi < p["pullback"])] = 1
    want[down & (rsi > 100 - p["pullback"])] = -1
    want[(want.isna()) & (((rsi > 50) & up) | ((rsi < 50) & down) | (~up & ~down))] = 0
    # hold the last decision until something changes it
    return want.ffill().fillna(0)
