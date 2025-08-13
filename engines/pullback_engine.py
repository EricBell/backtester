"""
engines/pullback_engine.py - Pullback-to-EMA strategy stub.
"""

import pandas as pd

class PullbackStrategy:
    def __init__(self, params: dict):
        self.params = params or {}
        self.ema_fast = self.params.get("ema_fast", 9)
        self.ema_slow = self.params.get("ema_slow", 20)
        self.pullback_bars = self.params.get("pullback_bars", 3)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = ["timestamp", "side", "entry_price", "stop_price", "target_price", "meta"]
        return pd.DataFrame(columns=cols)