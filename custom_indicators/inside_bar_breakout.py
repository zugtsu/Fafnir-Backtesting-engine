"""Example custom indicator: trade the break of an inside bar for a few bars."""
import numpy as np
import pandas as pd

NAME = "Inside bar breakout"
DESCRIPTION = ("After an inside bar (range inside the bar before it), go with the side that breaks first, "
               "then step aside after a set number of bars.")
PARAMS = {"hold_bars": 6}


def signal(df: pd.DataFrame, p: dict) -> pd.Series:
    prev_h, prev_l = df["high"].shift(1), df["low"].shift(1)
    inside = (df["high"] < prev_h) & (df["low"] > prev_l)
    mother_h, mother_l = df["high"].where(inside).shift(1), df["low"].where(inside).shift(1)
    side = np.where(df["close"] > mother_h, 1.0, np.where(df["close"] < mother_l, -1.0, np.nan))
    # keep the position for hold_bars bars after the break, then flat
    return pd.Series(side, index=df.index).ffill(limit=int(p["hold_bars"]) - 1).fillna(0)
