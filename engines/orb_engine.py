# engines/orb_engine.py
"""
Opening Range Breakout (ORB) Strategy Engine

This module implements an Opening Range Breakout strategy that:
1. Defines an opening range based on the first N minutes of a trading session
2. Monitors for price breakouts above/below this range
3. Generates signals with appropriate stops and targets
4. Includes volume and EMA confirmation filters
"""

import pandas as pd
from core import indicators
from datetime import timedelta
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any


# Debug integration - can be enabled via environment variables
try:
    from engines.orb_debug import OrbDebug
    _debug_enabled = os.getenv("DEBUG_ORB", "0") not in ("0", "", "false", "False")
    _debug_outdir = Path(os.getenv("ORB_DEBUG_OUTDIR", "outputs/orb_debug"))
    orb_debugger = OrbDebug(enabled=_debug_enabled, outdir=_debug_outdir)
except ImportError:
    # Fallback if debug module not available
    class MockDebugger:
        def log(self, data): pass
        def flush(self): pass
    orb_debugger = MockDebugger()
    _debug_enabled = False


class ORBStrategy:
    """
    Opening Range Breakout Strategy.
    
    Generates long/short signals when price breaks above/below the opening range
    with optional volume and EMA confirmation.
    """
    
    def __init__(self, params: dict):
        """Initialize ORB strategy with parameters."""
        self.params = params or {}
        
        # Core ORB parameters
        self.open_range_minutes = int(self.params.get("open_range_minutes", 15))
        self.require_volume_confirmation = bool(self.params.get("require_volume_confirmation", True))
        self.volume_lookback = int(self.params.get("volume_lookback", 5))
        self.take_profit_r = list(self.params.get("take_profit_r", [1.0, 2.0]))
        
        # Stop loss policies and parameters
        self.stop_policy = self.params.get("stop_policy", "structure")
        self.atr_multiplier_stop = float(self.params.get("atr_multiplier_stop", 1.0))
        self.stop_fixed_points = float(self.params.get("stop_fixed_points", 4.0))
        
        # Signal management
        self.cancel_within_minutes = int(self.params.get("cancel_within_minutes", 60))

    def _ensure_datetime_index(self, df: pd.DataFrame, tz_target: str) -> pd.DataFrame:
        """Ensure DataFrame has a proper timezone-aware DatetimeIndex."""
        df = df.copy()
        
        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                raise ValueError(f"Index is not datetime-like and could not be parsed: {e}")
        
        # Handle timezone localization/conversion
        try:
            if df.index.tz is None:
                df.index = df.index.tz_localize(tz_target)
            else:
                df.index = df.index.tz_convert(tz_target)
        except Exception as e:
            # Fallback attempt
            try:
                df.index = pd.to_datetime(df.index).tz_localize(tz_target)
            except Exception:
                raise ValueError(f"Failed to localize/convert index to {tz_target}: {e}")
        
        return df

    def _apply_session_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply session time filtering if configured."""
        session_start = getattr(self, "session_start", None)
        session_end = getattr(self, "session_end", None)
        
        if not (session_start and session_end):
            return df
        
        try:
            return df.between_time(session_start, session_end)
        except Exception:
            # Fallback: manual time comparison
            start_time = pd.to_datetime(session_start).time()
            end_time = pd.to_datetime(session_end).time()
            return df[(df.index.time >= start_time) & (df.index.time <= end_time)]

    def _day_groups(self, df: pd.DataFrame, tz_target: str = "America/New_York"):
        """
        Yield (date, day_df) for each calendar date that has session bars.
        
        Args:
            df: DataFrame with OHLCV data
            tz_target: Target timezone for index conversion
            
        Yields:
            Tuple[date, pd.DataFrame]: Calendar date and corresponding day data
        """
        df = self._ensure_datetime_index(df, tz_target)
        
        # Group by calendar date
        for date_val, group in df.groupby(df.index.date):
            # Apply session filtering if configured
            session_df = self._apply_session_filter(group)
            
            if session_df.empty:
                continue
            
            # Sort and yield
            session_df = session_df.sort_index()
            yield date_val, session_df

    def _calculate_opening_range(self, day_df: pd.DataFrame) -> Tuple[float, float, pd.Timestamp]:
        """
        Calculate the opening range high/low and end timestamp.
        
        Args:
            day_df: Single day's OHLCV data
            
        Returns:
            Tuple of (or_high, or_low, or_end_timestamp)
        """
        if day_df.empty:
            raise ValueError("Cannot calculate opening range: empty day data")
        
        session_start_ts = day_df.index[0]
        or_end_ts = session_start_ts + timedelta(minutes=self.open_range_minutes)
        
        # Get opening range bars
        or_bars = day_df.loc[
            (day_df.index > session_start_ts - pd.Timedelta(seconds=1)) & 
            (day_df.index <= or_end_ts)
        ]
        
        if or_bars.empty:
            raise ValueError("No bars found in opening range period")
        
        or_high = or_bars["high"].max()
        or_low = or_bars["low"].min()
        
        return float(or_high), float(or_low), or_end_ts

    def _calculate_volume_average(self, day_df: pd.DataFrame) -> pd.Series:
        """Calculate rolling volume average for volume confirmation."""
        if len(day_df) >= self.volume_lookback:
            return day_df["volume"].rolling(self.volume_lookback, min_periods=1).mean()
        else:
            return day_df["volume"].expanding().mean()

    def _check_breakout_conditions(self, row: pd.Series, vol_avg_series: pd.Series, 
                                 or_high: float, or_low: float, timestamp: pd.Timestamp) -> Dict[str, Any]:
        """
        Check if current bar meets breakout conditions.
        
        Args:
            row: Current OHLCV bar
            vol_avg_series: Volume average series
            or_high: Opening range high
            or_low: Opening range low
            timestamp: Current bar timestamp
            
        Returns:
            Dict with breakout analysis results
        """
        close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        ema9 = float(row["ema9"]) if pd.notna(row["ema9"]) else None
        vol = float(row["volume"])
        
        # Get volume average for this timestamp
        vol_avg = float(vol_avg_series.loc[timestamp]) if timestamp in vol_avg_series.index else float(vol_avg_series.iloc[-1])
        
        result = {
            "close": close,
            "high": high,
            "low": low,
            "ema9": ema9,
            "volume": vol,
            "volume_avg": vol_avg,
            "breakout_type": None,
            "ema_confirmed": False,
            "volume_confirmed": False,
            "signal_valid": False
        }
        
        # Check for long breakout
        if close > or_high:
            result["breakout_type"] = "LONG"
            result["ema_confirmed"] = (ema9 is not None) and (close > ema9)
            result["volume_confirmed"] = (not self.require_volume_confirmation) or (vol >= vol_avg)
            result["signal_valid"] = result["ema_confirmed"] and result["volume_confirmed"]
        
        # Check for short breakout
        elif close < or_low:
            result["breakout_type"] = "SHORT"
            result["ema_confirmed"] = (ema9 is not None) and (close < ema9)
            result["volume_confirmed"] = (not self.require_volume_confirmation) or (vol >= vol_avg)
            result["signal_valid"] = result["ema_confirmed"] and result["volume_confirmed"]
        
        return result

    def _calculate_stop_price(self, row: pd.Series, close: float, side: str) -> float:
        """
        Calculate stop price based on configured stop policy.
        
        Args:
            row: Current OHLCV bar
            close: Close price
            side: "LONG" or "SHORT"
            
        Returns:
            Stop price
        """
        if self.stop_policy == "structure":
            # Use breakout bar's opposite extreme
            return float(row["low"]) if side == "LONG" else float(row["high"])
        
        elif self.stop_policy == "atr":
            # ATR-based stop
            atr = float(row["atr14"]) if pd.notna(row["atr14"]) else 2.0  # Fallback ATR
            if side == "LONG":
                return close - self.atr_multiplier_stop * atr
            else:
                return close + self.atr_multiplier_stop * atr
        
        else:  # "fixed"
            # Fixed points stop
            if side == "LONG":
                return close - self.stop_fixed_points
            else:
                return close + self.stop_fixed_points

    def _calculate_targets(self, close: float, stop_price: float, side: str) -> Tuple[float, float]:
        """
        Calculate profit targets based on initial risk.
        
        Args:
            close: Entry price
            stop_price: Stop loss price
            side: "LONG" or "SHORT"
            
        Returns:
            Tuple of (target1, target2)
        """
        initial_r = abs(close - stop_price)
        
        if side == "LONG":
            t1 = close + self.take_profit_r[0] * initial_r
            t2 = close + (self.take_profit_r[1] if len(self.take_profit_r) > 1 else 2.0) * initial_r
        else:  # SHORT
            t1 = close - self.take_profit_r[0] * initial_r
            t2 = close - (self.take_profit_r[1] if len(self.take_profit_r) > 1 else 2.0) * initial_r
        
        return float(t1), float(t2)

    def _create_signal(self, timestamp: pd.Timestamp, row: pd.Series, breakout_result: Dict[str, Any],
                      or_high: float, or_low: float, date: Any) -> Dict[str, Any]:
        """
        Create a trading signal dictionary.
        
        Args:
            timestamp: Signal timestamp
            row: Current OHLCV bar
            breakout_result: Results from breakout condition check
            or_high: Opening range high
            or_low: Opening range low
            date: Trading date
            
        Returns:
            Signal dictionary
        """
        side = breakout_result["breakout_type"]
        close = breakout_result["close"]
        
        # Calculate stop and targets
        stop_price = self._calculate_stop_price(row, close, side)
        t1, t2 = self._calculate_targets(close, stop_price, side)
        
        # Create signal
        signal = {
            "timestamp": timestamp,
            "side": side,
            "signal_price": close,
            "entry_price_hint": None,  # Backtester will enter at next bar open
            "stop_price": stop_price,
            "target1": t1,
            "target2": t2,
            "meta": {
                "date": str(date),
                "or_high": or_high,
                "or_low": or_low,
                "reason": f"ORB {side.lower()}"
            }
        }
        
        return signal

    def _log_breakout_candidate(self, timestamp: pd.Timestamp, row: pd.Series, 
                               breakout_result: Dict[str, Any], or_high: float, or_low: float):
        """Log breakout candidate for debugging purposes."""
        if not _debug_enabled:
            return
        
        try:
            # Build debug record
            candidate_row = {
                "timestamp": timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp),
                "local_time": timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp),
                "close": breakout_result["close"],
                "open": float(row.get("open", 0)),
                "high": breakout_result["high"],
                "low": breakout_result["low"],
                "volume": int(breakout_result["volume"]),
                "or_high": float(or_high),
                "or_low": float(or_low),
                "breakout_dir": breakout_result["breakout_type"].lower() if breakout_result["breakout_type"] else "none",
                "ema9": breakout_result["ema9"],
                "vwap": None,  # Not implemented in current version
                "volume_avg": breakout_result["volume_avg"],
                "ema_pass": breakout_result["ema_confirmed"],
                "vwap_pass": True,  # Not used currently
                "volume_pass": breakout_result["volume_confirmed"],
                "time_rule_pass": True,  # Always true in current implementation
                "decision": "ACCEPT" if breakout_result["signal_valid"] else "REJECT",
                "reject_reasons": self._get_reject_reasons(breakout_result),
                "planned_contracts": None,
                "planned_entry": None,
                "planned_stop": None,
                "planned_target": None,
            }
            
            orb_debugger.log(candidate_row)
        except Exception as e:
            print(f"ORB_DEBUG log failed: {e}")

    def _get_reject_reasons(self, breakout_result: Dict[str, Any]) -> str:
        """Get rejection reasons for failed breakout conditions."""
        reasons = []
        if not breakout_result["ema_confirmed"]:
            reasons.append("ema_fail")
        if not breakout_result["volume_confirmed"]:
            reasons.append("volume_fail")
        return ";".join(reasons)

    def _append_signal_dedup(self, signals_list: List[Dict], seen_keys: set, signal: Dict[str, Any]):
        """
        Append signal to list only if not a duplicate.
        
        Args:
            signals_list: List to append to
            seen_keys: Set of seen signal keys
            signal: Signal dictionary to add
        """
        try:
            ts = signal.get("timestamp")
            ts_key = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            side = signal.get("side")
            price = float(signal.get("signal_price"))
            key = (ts_key, side, price)
        except Exception:
            # If key generation fails, still append (be conservative)
            key = None
        
        if key is not None:
            if key in seen_keys:
                return
            seen_keys.add(key)
        
        signals_list.append(signal)

    def _process_day_signals(self, date: Any, day_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Process a single day's data for ORB signals.
        
        Args:
            date: Trading date
            day_df: Single day's OHLCV data with indicators
            
        Returns:
            List of signal dictionaries
        """
        signals = []
        
        if day_df.empty:
            return signals
        
        try:
            # Calculate opening range
            or_high, or_low, or_end_ts = self._calculate_opening_range(day_df)
        except ValueError:
            # Cannot define OR for this day
            return signals
        
        # Define signal search window
        signal_search_start = or_end_ts + pd.Timedelta(microseconds=1)
        signal_search_end = or_end_ts + pd.Timedelta(minutes=self.cancel_within_minutes)
        search_bars = day_df.loc[
            (day_df.index >= signal_search_start) & 
            (day_df.index <= signal_search_end)
        ]
        
        if search_bars.empty:
            return signals
        
        # Calculate volume averages
        vol_avg_series = self._calculate_volume_average(day_df)
        
        # Track signals per day (one per direction)
        long_signal_taken = False
        short_signal_taken = False
        
        # Process each bar in search window
        for timestamp, row in search_bars.iterrows():
            # Check breakout conditions
            breakout_result = self._check_breakout_conditions(
                row, vol_avg_series, or_high, or_low, timestamp
            )
            
            # Log for debugging
            if breakout_result["breakout_type"]:
                self._log_breakout_candidate(timestamp, row, breakout_result, or_high, or_low)
            
            # Generate signal if conditions met
            if breakout_result["signal_valid"]:
                if breakout_result["breakout_type"] == "LONG" and not long_signal_taken:
                    signal = self._create_signal(timestamp, row, breakout_result, or_high, or_low, date)
                    signals.append(signal)
                    long_signal_taken = True
                
                elif breakout_result["breakout_type"] == "SHORT" and not short_signal_taken:
                    signal = self._create_signal(timestamp, row, breakout_result, or_high, or_low, date)
                    signals.append(signal)
                    short_signal_taken = True
            
            # Break if both directions have signals
            if long_signal_taken and short_signal_taken:
                break
        
        return signals

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ORB trading signals from OHLCV data.
        
        Args:
            df: DataFrame with columns [open, high, low, close, volume]
            
        Returns:
            DataFrame with columns [timestamp, side, signal_price, entry_price_hint, 
                                  stop_price, target1, target2, meta]
        """
        # Validate required columns
        required_cols = {"open", "high", "low", "close", "volume"}
        if not required_cols.issubset(df.columns):
            return pd.DataFrame(columns=[
                "timestamp", "side", "signal_price", "entry_price_hint",
                "stop_price", "target1", "target2", "meta"
            ])
        
        if df.empty:
            return pd.DataFrame(columns=[
                "timestamp", "side", "signal_price", "entry_price_hint",
                "stop_price", "target1", "target2", "meta"
            ])
        
        # Calculate technical indicators
        df = df.copy()
        df["ema9"] = indicators.ema(df["close"], 9)
        df["atr14"] = indicators.atr(df, 14)
        
        # Process each trading day
        all_signals = []
        seen_keys = set()
        
        for date, day_df in self._day_groups(df):
            day_signals = self._process_day_signals(date, day_df)
            
            # Add signals with deduplication
            for signal in day_signals:
                self._append_signal_dedup(all_signals, seen_keys, signal)
            
            # Flush debug buffer at end of each day
            if _debug_enabled:
                orb_debugger.flush()
        
        # Return as DataFrame
        if not all_signals:
            return pd.DataFrame(columns=[
                "timestamp", "side", "signal_price", "entry_price_hint",
                "stop_price", "target1", "target2", "meta"
            ])
        
        return pd.DataFrame(all_signals).sort_values("timestamp").reset_index(drop=True)