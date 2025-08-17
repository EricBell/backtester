"""
engines/pullback_engine.py - Pullback-to-EMA strategy implementation.

Strategy Logic:
1. Identify trend using fast EMA vs slow EMA
2. Wait for pullback to fast EMA 
3. Enter when price resumes trend direction
4. Use ATR-based stops and risk-reward targets
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class PullbackStrategy:
    """
    Pullback-to-EMA Strategy
    
    Enters trades when price pulls back to moving average after establishing
    a trend, then resumes the original direction.
    """
    
    def __init__(self, params: dict = None):
        """Initialize the Pullback strategy with parameters"""
        self.params = params or {}
        
        # EMA parameters
        self.ema_fast = self.params.get("ema_fast", 9)
        self.ema_slow = self.params.get("ema_slow", 20)
        
        # Pullback detection
        self.pullback_bars = self.params.get("pullback_bars", 3)
        
        # Risk management
        self.atr_period = self.params.get("atr_period", 14)
        self.atr_multiplier_stop = self.params.get("atr_multiplier_stop", 1.2)
        self.target_r = self.params.get("target_r", 2.0)
        self.stop_fixed_points = self.params.get("stop_fixed_points", 4.0)
        
        # Filters
        self.require_bias_by_ema = self.params.get("require_bias_by_ema", True)
        self.round_stops_to_tick = self.params.get("round_stops_to_tick", True)
        self.tick_size = self.params.get("tick_size", 0.25)
        
        # Session filter
        self.session_start = self.params.get("session_start", "08:00")
        self.session_end = self.params.get("session_end", "12:00")

    def in_session(self, dt) -> bool:
        """Check if a given datetime is within the trading session"""
        time = pd.to_datetime(dt).time()
        start_time = pd.to_datetime(self.session_start).time()
        end_time = pd.to_datetime(self.session_end).time()
        return start_time <= time <= end_time

    def round_to_tick(self, price: float, direction: str = "nearest") -> float:
        """Round price to nearest tick size"""
        if not self.round_stops_to_tick or self.tick_size <= 0:
            return price
        
        if direction == "nearest":
            return round(price / self.tick_size) * self.tick_size
        elif direction == "up":
            return np.ceil(price / self.tick_size) * self.tick_size
        elif direction == "down":
            return np.floor(price / self.tick_size) * self.tick_size
        else:
            return round(price / self.tick_size) * self.tick_size

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on the Pullback strategy
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with trading signals
        """
        # Ensure DataFrame has required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Create a copy to avoid modifying original DataFrame
        df = df.copy()
        
        # Calculate EMAs
        df[f'ema_{self.ema_fast}'] = df['close'].ewm(span=self.ema_fast, adjust=False).mean()
        df[f'ema_{self.ema_slow}'] = df['close'].ewm(span=self.ema_slow, adjust=False).mean()
        
        # Calculate ATR for stop placement
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(self.atr_period).mean()
        
        # Determine trend direction
        df['trend_bullish'] = df[f'ema_{self.ema_fast}'] > df[f'ema_{self.ema_slow}']
        df['trend_bearish'] = df[f'ema_{self.ema_fast}'] < df[f'ema_{self.ema_slow}']
        
        # Track pullback states
        df['price_below_fast_ema'] = df['close'] < df[f'ema_{self.ema_fast}']
        df['price_above_fast_ema'] = df['close'] > df[f'ema_{self.ema_fast}']
        
        # Pullback detection: price has been away from fast EMA for at least pullback_bars periods recently
        df['bullish_pullback_active'] = (
            df['trend_bullish'] & 
            # df['price_below_fast_ema'].rolling(self.pullback_bars).sum() >= 1  # At least 1 period below in last N bars
            df['price_below_fast_ema'].rolling(self.pullback_bars).sum() >= (self.pullback_bars // 2)  # At least 1 period below in last N bars
        )
        
        df['bearish_pullback_active'] = (
            df['trend_bearish'] & 
            # df['price_above_fast_ema'].rolling(self.pullback_bars).sum() >= 1  # At least 1 period above in last N bars
            df['price_above_fast_ema'].rolling(self.pullback_bars).sum() >= (self.pullback_bars // 2)  # At least 1 period above in last N bars
        )
        
        # Create empty signals list
        signals = []
        
        # Loop through data (skip first few bars for indicator warmup)
        warmup = max(self.ema_slow, self.atr_period, self.pullback_bars) + 1
        
        for i in range(warmup, len(df) - 1):
            # Skip if not in trading session
            if not self.in_session(df.index[i]):
                continue
                
            current = df.iloc[i]
            prev = df.iloc[i-1]
            next_bar = df.iloc[i+1]  # For entry price
            
            # Long signal: bullish trend, pullback completed, price back above fast EMA
            long_setup = (
                current['trend_bullish'] and
                prev['bullish_pullback_active'] and
                current['close'] > current[f'ema_{self.ema_fast}'] and
                prev['close'] <= prev[f'ema_{self.ema_fast}']  # Just crossed back above
            )
            
            
            # Additional bias confirmation if required
            if self.require_bias_by_ema:
                long_setup = long_setup and (current['close'] > current[f'ema_{self.ema_slow}'])
            
            # Short signal: bearish trend, pullback completed, price back below fast EMA  
            short_setup = (
                current['trend_bearish'] and
                prev['bearish_pullback_active'] and
                current['close'] < current[f'ema_{self.ema_fast}'] and
                prev['close'] >= prev[f'ema_{self.ema_fast}']  # Just crossed back below
            )
            
            # Additional bias confirmation if required
            if self.require_bias_by_ema:
                short_setup = short_setup and (current['close'] < current[f'ema_{self.ema_slow}'])
            
            if long_setup:
                # Entry price would be next bar's open
                entry_price = next_bar['open']
                
                # Calculate stop loss
                if not pd.isna(current['atr']) and current['atr'] > 0:
                    stop_distance = current['atr'] * self.atr_multiplier_stop
                else:
                    stop_distance = self.stop_fixed_points
                
                stop_price = entry_price - stop_distance
                stop_price = self.round_to_tick(stop_price, "down")
                
                # Calculate target
                risk = entry_price - stop_price
                target_price = entry_price + (risk * self.target_r)
                target_price = self.round_to_tick(target_price, "up")
                
                # Create signal
                signal = {
                    "timestamp": df.index[i+1],  # Signal activated on next bar
                    "side": "LONG",
                    "entry_price": entry_price,
                    "stop_price": stop_price,
                    "target_price": target_price,
                    "meta": {
                        "setup": "pullback_long",
                        "fast_ema": current[f'ema_{self.ema_fast}'],
                        "slow_ema": current[f'ema_{self.ema_slow}'],
                        "atr": current['atr'],
                        "stop_distance": stop_distance,
                        "risk_points": risk,
                        "target_points": target_price - entry_price
                    }
                }
                signals.append(signal)
                
            elif short_setup:
                # Entry price would be next bar's open
                entry_price = next_bar['open']
                
                # Calculate stop loss
                if not pd.isna(current['atr']) and current['atr'] > 0:
                    stop_distance = current['atr'] * self.atr_multiplier_stop
                else:
                    stop_distance = self.stop_fixed_points
                
                stop_price = entry_price + stop_distance
                stop_price = self.round_to_tick(stop_price, "up")
                
                # Calculate target
                risk = stop_price - entry_price
                target_price = entry_price - (risk * self.target_r)
                target_price = self.round_to_tick(target_price, "down")
                
                # Create signal
                signal = {
                    "timestamp": df.index[i+1],  # Signal activated on next bar
                    "side": "SHORT",
                    "entry_price": entry_price,
                    "stop_price": stop_price,
                    "target_price": target_price,
                    "meta": {
                        "setup": "pullback_short",
                        "fast_ema": current[f'ema_{self.ema_fast}'],
                        "slow_ema": current[f'ema_{self.ema_slow}'],
                        "atr": current['atr'],
                        "stop_distance": stop_distance,
                        "risk_points": risk,
                        "target_points": entry_price - target_price
                    }
                }
                signals.append(signal)
        
        # Convert signals list to DataFrame
        return pd.DataFrame(signals)