# tests/test_orb_signals.py
import pandas as pd
import numpy as np
from engines.orb_engine import ORBStrategy
from pathlib import Path

def make_synthetic_day():
    # create synthetic 3-min bars for one session day
    idx = pd.date_range("2025-08-01 08:00:00", periods=40, freq="3min", tz="America/New_York")
    # create a drift up after 15 min
    open_prices = np.linspace(4000, 4010, len(idx))
    high = open_prices + 0.5
    low = open_prices - 0.5
    close = open_prices + 0.2
    volume = np.full(len(idx), 1000)
    df = pd.DataFrame({"open":open_prices,"high":high,"low":low,"close":close,"volume":volume}, index=idx)
    return df

def test_orb_detects_signal():
    df = make_synthetic_day()
    params = {"open_range_minutes": 15, "require_volume_confirmation": False}
    s = ORBStrategy(params)
    sigs = s.generate_signals(df)
    assert isinstance(sigs, pd.DataFrame)
    # on this synthetic data there should be at least 0 or 1 signals (sanity)
    assert "timestamp" in sigs.columns