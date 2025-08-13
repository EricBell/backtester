"""
core/indicators.py - minimal pandas implementations of required indicators.
"""

import pandas as pd
import numpy as np

def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/n, adjust=False).mean()
    ma_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_tp_vol = (typical * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    return cum_tp_vol / cum_vol

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()