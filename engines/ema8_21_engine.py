# engines/ema8_21_engine.py
"""
Ultra-Selective 8/21 EMA Strategy Engine for MES Futures

This module implements a highly selective strategy using 8/21 EMA crossovers with strong 
trend filters on 30-minute charts for MES futures day trading.

Key Features:
1. EMA 8/21 crossover signals with trend strength validation
2. RSI confirmation filters
3. Multiple take profit levels and trailing stops
4. Risk management with position sizing
5. Time-based session filtering
"""

import pandas as pd
from core import indicators
from datetime import timedelta, time
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


class EMA821Strategy:
    """
    Ultra-Selective 8/21 EMA Strategy for MES Futures.
    
    Generates long/short signals when EMA 8 crosses EMA 21 with strong trend confirmation.
    """
    
    def __init__(self, params: dict):
        """Initialize EMA 8-21 strategy with parameters."""
        self.params = params or {}
        
        # Strategy configuration from JSON
        self.strategy_name = "ema8-21"
        self.timeframe = "30m"
        
        # Trading session parameters
        self.session_start = self.params.get("session_start", "09:45")  # Exclude first 15 minutes
        self.session_end = self.params.get("session_end", "14:30")
        self.timezone = self.params.get("timezone", "America/New_York")
        
        # Indicator parameters
        self.fast_ema_period = int(self.params.get("fast_ema_period", 8))
        self.slow_ema_period = int(self.params.get("slow_ema_period", 21))
        self.rsi_period = int(self.params.get("rsi_period", 14))
        self.rsi_overbought = float(self.params.get("rsi_overbought", 60))
        self.rsi_oversold = float(self.params.get("rsi_oversold", 40))
        self.trend_strength_threshold = float(self.params.get("trend_strength_threshold", 0.8))
        
        # Exit parameters
        self.take_profit_levels = self.params.get("take_profit_levels", [4.0, 6.0])
        self.take_profit_sizes = self.params.get("take_profit_sizes", [50, 50])
        self.stop_loss_percentage = float(self.params.get("stop_loss_percentage", 2.5))
        self.trailing_stop_percentage = float(self.params.get("trailing_stop_percentage", 1.8))
        self.trailing_activation_rr = float(self.params.get("trailing_activation_rr", 1.0))
        
        # Risk management
        self.account_size = float(self.params.get("account_size", 2000))
        self.risk_per_trade_percentage = float(self.params.get("risk_per_trade_percentage", 2.5))
        self.max_daily_loss_percentage = float(self.params.get("max_daily_loss_percentage", 5.0))
        self.max_open_positions = int(self.params.get("max_open_positions", 2))
        
        # Position sizing
        self.default_position_size = int(self.params.get("default_position_size", 1))
        self.max_position_size = int(self.params.get("max_position_size", 2))
        
        # Execution
        self.commission_per_contract = float(self.params.get("commission_per_contract", 2.0))
        self.point_value = float(self.params.get("point_value", 5.0))
        self.slippage_points = float(self.params.get("slippage_points", 1.0))

    def _ensure_datetime_index(self, df: pd.DataFrame, tz_target: str) -> pd.DataFrame:
        """Ensure DataFrame has a proper timezone-aware DatetimeIndex."""
        df = df.copy()
        
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                raise ValueError(f"Index is not datetime-like and could not be parsed: {e}")
        
        try:
            if df.index.tz is None:
                df.index = df.index.tz_localize(tz_target)
            else:
                df.index = df.index.tz_convert(tz_target)
        except Exception as e:
            try:
                df.index = pd.to_datetime(df.index).tz_localize(tz_target)
            except Exception:
                raise ValueError(f"Failed to localize/convert index to {tz_target}: {e}")
        
        return df

    def _apply_session_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply session time filtering."""
        try:
            return df.between_time(self.session_start, self.session_end)
        except Exception:
            start_time = pd.to_datetime(self.session_start).time()
            end_time = pd.to_datetime(self.session_end).time()
            return df[(df.index.time >= start_time) & (df.index.time <= end_time)]

    def _day_groups(self, df: pd.DataFrame, tz_target: str = "America/New_York"):
        """Yield (date, day_df) for each calendar date that has session bars."""
        df = self._ensure_datetime_index(df, tz_target)
        
        for date_val, group in df.groupby(df.index.date):
            session_df = self._apply_session_filter(group)
            
            if session_df.empty:
                continue
            
            session_df = session_df.sort_index()
            yield date_val, session_df

    def _calculate_trend_strength(self, fast_ema: float, slow_ema: float, close: float) -> float:
        """Calculate trend strength as percentage difference between EMAs relative to price."""
        if close == 0:
            return 0.0
        return abs(fast_ema - slow_ema) / close * 100

    def _check_ema_crossover(self, current_fast: float, current_slow: float, 
                           prev_fast: float, prev_slow: float) -> Optional[str]:
        """Check for EMA crossover and return direction."""
        # Bullish crossover: fast EMA crosses above slow EMA
        if prev_fast <= prev_slow and current_fast > current_slow:
            return "LONG"
        
        # Bearish crossover: fast EMA crosses below slow EMA
        elif prev_fast >= prev_slow and current_fast < current_slow:
            return "SHORT"
        
        return None

    def _check_entry_conditions(self, row: pd.Series, prev_row: pd.Series) -> Dict[str, Any]:
        """Check if current bar meets entry conditions for EMA 8-21 strategy."""
        close = float(row["close"])
        fast_ema = float(row["ema8"])
        slow_ema = float(row["ema21"])
        rsi = float(row["rsi14"])
        
        prev_fast_ema = float(prev_row["ema8"])
        prev_slow_ema = float(prev_row["ema21"])
        
        # Calculate trend strength
        trend_strength = self._calculate_trend_strength(fast_ema, slow_ema, close)
        
        # Check for crossover
        crossover_direction = self._check_ema_crossover(
            fast_ema, slow_ema, prev_fast_ema, prev_slow_ema
        )
        
        result = {
            "close": close,
            "fast_ema": fast_ema,
            "slow_ema": slow_ema,
            "rsi": rsi,
            "trend_strength": trend_strength,
            "crossover_direction": crossover_direction,
            "signal_valid": False,
            "signal_type": None
        }
        
        if crossover_direction == "LONG":
            # Long entry conditions
            conditions_met = (
                trend_strength > self.trend_strength_threshold and
                rsi > self.rsi_oversold and
                close > fast_ema and
                close > slow_ema
            )
            
            if conditions_met:
                result["signal_valid"] = True
                result["signal_type"] = "LONG"
        
        elif crossover_direction == "SHORT":
            # Short entry conditions
            conditions_met = (
                trend_strength > self.trend_strength_threshold and
                rsi < self.rsi_overbought and
                close < fast_ema and
                close < slow_ema
            )
            
            if conditions_met:
                result["signal_valid"] = True
                result["signal_type"] = "SHORT"
        
        return result

    def _calculate_position_size(self, entry_price: float, stop_price: float) -> int:
        """Calculate position size based on risk management rules."""
        # Use fixed position size for better profits
        return self.default_position_size

    def _calculate_stop_loss(self, entry_price: float, side: str) -> float:
        """Calculate stop loss based on percentage."""
        stop_distance = entry_price * (self.stop_loss_percentage / 100)
        
        if side == "LONG":
            return entry_price - stop_distance
        else:  # SHORT
            return entry_price + stop_distance

    def _calculate_take_profit_levels(self, entry_price: float, stop_price: float, side: str) -> List[float]:
        """Calculate multiple take profit levels."""
        initial_risk = abs(entry_price - stop_price)
        targets = []
        
        for tp_percentage in self.take_profit_levels:
            if side == "LONG":
                target = entry_price + (initial_risk * tp_percentage / 100)
            else:  # SHORT
                target = entry_price - (initial_risk * tp_percentage / 100)
            
            targets.append(float(target))
        
        return targets

    def _create_signal(self, timestamp: pd.Timestamp, row: pd.Series, 
                      entry_conditions: Dict[str, Any], date: Any) -> Dict[str, Any]:
        """Create a trading signal dictionary."""
        side = entry_conditions["signal_type"]
        entry_price = entry_conditions["close"]
        
        # Calculate stop loss
        stop_price = self._calculate_stop_loss(entry_price, side)
        
        # Calculate position size
        position_size = self._calculate_position_size(entry_price, stop_price)
        
        # Calculate take profit levels
        targets = self._calculate_take_profit_levels(entry_price, stop_price, side)
        
        # Create signal
        signal = {
            "timestamp": timestamp,
            "side": side,
            "signal_price": entry_price,
            "entry_price_hint": None,  # Backtester will enter at next bar open
            "stop_price": stop_price,
            "target1": targets[0] if len(targets) > 0 else None,
            "target2": targets[1] if len(targets) > 1 else None,
            "position_size": position_size,
            "meta": {
                "date": str(date),
                "strategy": "ema8-21",
                "fast_ema": entry_conditions["fast_ema"],
                "slow_ema": entry_conditions["slow_ema"],
                "rsi": entry_conditions["rsi"],
                "trend_strength": entry_conditions["trend_strength"],
                "reason": f"EMA 8/21 crossover {side.lower()}"
            }
        }
        
        return signal

    def _process_day_signals(self, date: Any, day_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Process a single day's data for EMA 8-21 signals."""
        signals = []
        
        if len(day_df) < 2:  # Need at least 2 bars to check crossover
            return signals
        
        # Track signals per day (allow multiple per direction)
        signals_taken = 0
        max_signals_per_day = 4
        
        # Process each bar starting from the second one
        for i in range(1, len(day_df)):
            current_row = day_df.iloc[i]
            prev_row = day_df.iloc[i-1]
            timestamp = day_df.index[i]
            
            # Skip if any required indicators are NaN
            required_fields = ["ema8", "ema21", "rsi14"]
            if any(pd.isna(current_row[field]) or pd.isna(prev_row[field]) for field in required_fields):
                continue
            
            # Check entry conditions
            entry_conditions = self._check_entry_conditions(current_row, prev_row)
            
            # Generate signal if conditions met
            if entry_conditions["signal_valid"] and signals_taken < max_signals_per_day:
                signal = self._create_signal(timestamp, current_row, entry_conditions, date)
                signals.append(signal)
                signals_taken += 1
            
            # Break if max signals reached
            if signals_taken >= max_signals_per_day:
                break
        
        return signals

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate EMA 8-21 trading signals from OHLCV data.
        
        Args:
            df: DataFrame with columns [open, high, low, close, volume]
            
        Returns:
            DataFrame with trading signals
        """
        # Validate required columns
        required_cols = {"open", "high", "low", "close", "volume"}
        if not required_cols.issubset(df.columns):
            return pd.DataFrame(columns=[
                "timestamp", "side", "signal_price", "entry_price_hint",
                "stop_price", "target1", "target2", "position_size", "meta"
            ])
        
        if df.empty:
            return pd.DataFrame(columns=[
                "timestamp", "side", "signal_price", "entry_price_hint",
                "stop_price", "target1", "target2", "position_size", "meta"
            ])
        
        # Calculate technical indicators
        df = df.copy()
        df["ema8"] = indicators.ema(df["close"], self.fast_ema_period)
        df["ema21"] = indicators.ema(df["close"], self.slow_ema_period)
        df["rsi14"] = indicators.rsi(df["close"], self.rsi_period)
        
        # Process each trading day
        all_signals = []
        
        for date, day_df in self._day_groups(df, self.timezone):
            day_signals = self._process_day_signals(date, day_df)
            all_signals.extend(day_signals)
        
        # Return as DataFrame
        if not all_signals:
            return pd.DataFrame(columns=[
                "timestamp", "side", "signal_price", "entry_price_hint",
                "stop_price", "target1", "target2", "position_size", "meta"
            ])
        
        return pd.DataFrame(all_signals).sort_values("timestamp").reset_index(drop=True)