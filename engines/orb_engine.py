# engines/orb_engine.py
import pandas as pd
import numpy as np
from core import indicators
from datetime import timedelta, time

class ORBStrategy:
    def __init__(self, params: dict):
        self.params = params or {}
        self.open_range_minutes = int(self.params.get("open_range_minutes", 15))
        self.require_volume_confirmation = bool(self.params.get("require_volume_confirmation", True))
        self.volume_lookback = int(self.params.get("volume_lookback", 20))
        self.take_profit_r = list(self.params.get("take_profit_r", [1.0, 2.0]))
        self.stop_policy = self.params.get("stop_policy", "structure")
        self.atr_multiplier_stop = float(self.params.get("atr_multiplier_stop", 1.0))
        self.stop_fixed_points = float(self.params.get("stop_fixed_points", 4.0))

        # misc
        self.cancel_within_minutes = int(self.params.get("cancel_within_minutes", 60))

    def _day_groups(self, df: pd.DataFrame):
        # group by calendar date in the index's timezone
        # Use date() of the localized index
        df = df.copy()
        df["date_only"] = df.index.tz_convert(df.index.tz).date
        for date, group in df.groupby("date_only"):
            yield date, group.drop(columns=["date_only"])

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return signals DataFrame with columns:
        timestamp, side ('LONG'/'SHORT'), signal_price (breakout close), entry_price_hint (None),
        stop_price, target1, target2, meta (dict)
        """
        signals = []
        # ensure we have columns open/high/low/close/volume
        if not {"open","high","low","close","volume"}.issubset(df.columns):
            return pd.DataFrame(columns=[
                "timestamp","side","signal_price","entry_price_hint","stop_price","target1","target2","meta"
            ])

        # compute EMA(9) for confirmation across entire df (3-min bars)
        df = df.copy()
        df["ema9"] = indicators.ema(df["close"], 9)
        df["atr14"] = indicators.atr(df, 14)

        for date, day_df in self._day_groups(df):
            if day_df.empty:
                continue
            # define session start timestamp for the day (first index >= session_start)
            # open range window: first open_range_minutes starting at the day's first index
            # We assume df is already filtered to session times and resampled (3-min)
            session_start_ts = day_df.index[0]
            or_end_ts = session_start_ts + timedelta(minutes=self.open_range_minutes)
            or_bars = day_df.loc[(day_df.index > session_start_ts - pd.Timedelta(seconds=1)) & (day_df.index <= or_end_ts)]
            if or_bars.empty:
                # cannot define OR for this day
                continue
            or_high = or_bars["high"].max()
            or_low = or_bars["low"].min()

            # signals only after OR end, within cancel window
            signal_search_start = or_end_ts + pd.Timedelta(microseconds=1)
            signal_search_end = or_end_ts + pd.Timedelta(minutes=self.cancel_within_minutes)
            search_bars = day_df.loc[(day_df.index >= signal_search_start) & (day_df.index <= signal_search_end)]
            if search_bars.empty:
                continue

            # compute recent volume rolling average on day_df for volume confirmation
            if len(day_df) >= self.volume_lookback:
                vol_avg_series = day_df["volume"].rolling(self.volume_lookback, min_periods=1).mean()
            else:
                vol_avg_series = day_df["volume"].expanding().mean()

            # iterate bars in search_bars to detect breakout closes
            for ts, row in search_bars.iterrows():
                close = float(row["close"])
                high = float(row["high"])
                low = float(row["low"])
                ema9 = float(row["ema9"]) if pd.notna(row["ema9"]) else None
                vol = float(row["volume"])
                vol_avg = float(vol_avg_series.loc[ts]) if ts in vol_avg_series.index else float(vol_avg_series.iloc[-1])

                # LONG breakout
                if close > or_high:
                    # EMI: EMA slope or price > ema9 for confirmation
                    ema_ok = (ema9 is not None) and (close > ema9)
                    vol_ok = (not self.require_volume_confirmation) or (vol >= vol_avg)
                    if ema_ok and vol_ok:
                        # determine stop: breakout candle low (structure) or ATR-based if configured
                        if self.stop_policy == "structure":
                            stop_price = float(low)
                        elif self.stop_policy == "atr":
                            stop_price = close - self.atr_multiplier_stop * float(row["atr14"])
                        else:  # fixed
                            stop_price = close - self.stop_fixed_points
                        # initial R
                        initial_r = abs(close - stop_price)
                        t1 = close + self.take_profit_r[0] * initial_r
                        t2 = close + (self.take_profit_r[1] if len(self.take_profit_r) > 1 else (2.0 * initial_r))
                        meta = {"date": str(date), "or_high": float(or_high), "or_low": float(or_low), "reason": "ORB long"}
                        signals.append({
                            "timestamp": ts,
                            "side": "LONG",
                            "signal_price": close,
                            "entry_price_hint": None,   # backtester will enter at next bar open
                            "stop_price": float(stop_price),
                            "target1": float(t1),
                            "target2": float(t2),
                            "meta": meta
                        })
                        # one long signal per OR
                        break

                # SHORT breakout
                if close < or_low:
                    ema_ok = (ema9 is not None) and (close < ema9)
                    vol_ok = (not self.require_volume_confirmation) or (vol <= vol_avg)  # for short can accept lower vol on pull?
                    # keep same vol rule for shorts (depends on design); we'll accept vol_ok if vol >= vol_avg (consistent)
                    vol_ok = (not self.require_volume_confirmation) or (vol >= vol_avg)
                    if ema_ok and vol_ok:
                        if self.stop_policy == "structure":
                            stop_price = float(high)
                        elif self.stop_policy == "atr":
                            stop_price = close + self.atr_multiplier_stop * float(row["atr14"])
                        else:
                            stop_price = close + self.stop_fixed_points
                        initial_r = abs(close - stop_price)
                        t1 = close - self.take_profit_r[0] * initial_r
                        t2 = close - (self.take_profit_r[1] if len(self.take_profit_r) > 1 else (2.0 * initial_r))
                        meta = {"date": str(date), "or_high": float(or_high), "or_low": float(or_low), "reason": "ORB short"}
                        signals.append({
                            "timestamp": ts,
                            "side": "SHORT",
                            "signal_price": close,
                            "entry_price_hint": None,
                            "stop_price": float(stop_price),
                            "target1": float(t1),
                            "target2": float(t2),
                            "meta": meta
                        })
                        break

        if not signals:
            return pd.DataFrame(columns=[
                "timestamp","side","signal_price","entry_price_hint","stop_price","target1","target2","meta"
            ])
        return pd.DataFrame(signals).sort_values("timestamp").reset_index(drop=True)