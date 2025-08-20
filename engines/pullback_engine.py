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
        
        # Stop Method Toggles
        self.use_atr_stops = self.params.get("use_atr_stops", True)
        self.use_fixed_stops = self.params.get("use_fixed_stops", True)
        self.use_sr_stops = self.params.get("use_sr_stops", False)
        
        # Support/Resistance Stop Settings
        self.sr_lookback_bars = self.params.get("sr_lookback_bars", 20)
        self.sr_min_touches = self.params.get("sr_min_touches", 2)
        self.sr_buffer_points = self.params.get("sr_buffer_points", 1)
        self.round_number_intervals = self.params.get("round_number_intervals", [25, 50, 100])
        
        # Stop Selection Method
        self.stop_selection = self.params.get("stop_selection", "tightest")

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
    
    def _calculate_atr_stop(self, entry_price: float, atr: float, side: str) -> float:
        """Calculate ATR-based stop loss"""
        if pd.isna(atr) or atr <= 0:
            return None
        
        stop_distance = atr * self.atr_multiplier_stop
        
        if side == "LONG":
            return entry_price - stop_distance
        else:  # SHORT
            return entry_price + stop_distance
    
    def _calculate_fixed_stop(self, entry_price: float, side: str) -> float:
        """Calculate fixed point stop loss"""
        if side == "LONG":
            return entry_price - self.stop_fixed_points
        else:  # SHORT
            return entry_price + self.stop_fixed_points
    
    def _find_support_resistance_levels(self, df: pd.DataFrame, current_idx: int) -> list:
        """Find nearby support and resistance levels"""
        levels = []
        
        # Look back from current position
        start_idx = max(0, current_idx - self.sr_lookback_bars)
        lookback_data = df.iloc[start_idx:current_idx + 1]
        
        if len(lookback_data) < 3:
            return levels
        
        # Find swing highs and lows
        for i in range(1, len(lookback_data) - 1):
            current_bar = lookback_data.iloc[i]
            prev_bar = lookback_data.iloc[i - 1]
            next_bar = lookback_data.iloc[i + 1]
            
            # Swing high (resistance)
            if (current_bar['high'] > prev_bar['high'] and 
                current_bar['high'] > next_bar['high']):
                levels.append({
                    'price': current_bar['high'],
                    'type': 'resistance',
                    'strength': 1
                })
            
            # Swing low (support)
            if (current_bar['low'] < prev_bar['low'] and 
                current_bar['low'] < next_bar['low']):
                levels.append({
                    'price': current_bar['low'],
                    'type': 'support',
                    'strength': 1
                })
        
        # Add round number levels
        current_price = df.iloc[current_idx]['close']
        
        for interval in self.round_number_intervals:
            # Find nearest round numbers above and below current price
            lower_round = (int(current_price / interval)) * interval
            upper_round = lower_round + interval
            
            # Check if these levels are within reasonable range
            price_range = current_price * 0.05  # 5% range
            
            if abs(lower_round - current_price) <= price_range:
                levels.append({
                    'price': lower_round,
                    'type': 'support' if lower_round < current_price else 'resistance',
                    'strength': 2  # Round numbers get higher strength
                })
            
            if abs(upper_round - current_price) <= price_range:
                levels.append({
                    'price': upper_round,
                    'type': 'support' if upper_round < current_price else 'resistance',
                    'strength': 2
                })
        
        # Remove duplicates and sort by distance from current price
        unique_levels = []
        for level in levels:
            # Check if this price level already exists (within 1 tick)
            duplicate = False
            for existing in unique_levels:
                if abs(existing['price'] - level['price']) <= self.tick_size:
                    # Keep the one with higher strength
                    if level['strength'] > existing['strength']:
                        unique_levels.remove(existing)
                    else:
                        duplicate = True
                    break
            
            if not duplicate:
                unique_levels.append(level)
        
        # Sort by distance from current price
        unique_levels.sort(key=lambda x: abs(x['price'] - current_price))
        
        return unique_levels
    
    def _calculate_sr_stop(self, entry_price: float, df: pd.DataFrame, current_idx: int, side: str) -> float:
        """Calculate support/resistance based stop loss"""
        levels = self._find_support_resistance_levels(df, current_idx)
        
        if not levels:
            return None
        
        # Find the nearest relevant level
        relevant_levels = []
        
        for level in levels:
            if side == "LONG":
                # For long trades, look for support levels below entry
                if level['type'] == 'support' and level['price'] < entry_price:
                    relevant_levels.append(level)
            else:  # SHORT
                # For short trades, look for resistance levels above entry
                if level['type'] == 'resistance' and level['price'] > entry_price:
                    relevant_levels.append(level)
        
        if not relevant_levels:
            return None
        
        # Use the nearest relevant level
        nearest_level = relevant_levels[0]
        
        if side == "LONG":
            # Stop just below support
            return nearest_level['price'] - self.sr_buffer_points
        else:  # SHORT
            # Stop just above resistance
            return nearest_level['price'] + self.sr_buffer_points
    
    def calculate_stop_price(self, entry_price: float, atr: float, df: pd.DataFrame, current_idx: int, side: str) -> float:
        """Calculate stop price using the selected method(s)"""
        stops = []
        
        # Calculate each type of stop if enabled
        if self.use_atr_stops:
            atr_stop = self._calculate_atr_stop(entry_price, atr, side)
            if atr_stop is not None:
                stops.append(('atr', atr_stop))
        
        if self.use_fixed_stops:
            fixed_stop = self._calculate_fixed_stop(entry_price, side)
            stops.append(('fixed', fixed_stop))
        
        if self.use_sr_stops:
            sr_stop = self._calculate_sr_stop(entry_price, df, current_idx, side)
            if sr_stop is not None:
                stops.append(('sr', sr_stop))
        
        if not stops:
            # Fallback to fixed stop if no methods enabled
            return self._calculate_fixed_stop(entry_price, side)
        
        # Select stop based on configuration
        stop_prices = [stop[1] for stop in stops]
        
        if self.stop_selection == "tightest":
            if side == "LONG":
                # Tightest stop for long = highest stop price (closest to entry)
                selected_stop = max(stop_prices)
            else:  # SHORT
                # Tightest stop for short = lowest stop price (closest to entry)
                selected_stop = min(stop_prices)
        elif self.stop_selection == "loosest":
            if side == "LONG":
                # Loosest stop for long = lowest stop price (furthest from entry)
                selected_stop = min(stop_prices)
            else:  # SHORT
                # Loosest stop for short = highest stop price (furthest from entry)
                selected_stop = max(stop_prices)
        else:  # "average"
            selected_stop = sum(stop_prices) / len(stop_prices)
        
        return selected_stop

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
                
                # Calculate stop loss using flexible method
                stop_price = self.calculate_stop_price(
                    entry_price, current['atr'], df, i, "LONG"
                )
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
                        "stop_distance": risk,  # Distance from entry to stop
                        "risk_points": risk,
                        "target_points": target_price - entry_price
                    }
                }
                signals.append(signal)
                
            elif short_setup:
                # Entry price would be next bar's open
                entry_price = next_bar['open']
                
                # Calculate stop loss using flexible method
                stop_price = self.calculate_stop_price(
                    entry_price, current['atr'], df, i, "SHORT"
                )
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
                        "stop_distance": risk,  # Distance from entry to stop
                        "risk_points": risk,
                        "target_points": entry_price - target_price
                    }
                }
                signals.append(signal)
        
        # Convert signals list to DataFrame
        return pd.DataFrame(signals)