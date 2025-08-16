# """
# engines/vwap_engine.py - VWAP fade strategy stub.
# """

'''
This implementation:

#Follows the strategy skeleton structure

#Implements the VWAP Fade / Mean Reversion strategy as described in your project snapshot

#Handles the required calculations for:

##VWAP calculation
##Price extension detection
##Volume analysis for pullback confirmation
##Swing high/low detection for stop placement
##Risk-based target calculation

#Returns signals in the required format with:

##timestamp (when the trade would be entered)
##side (LONG or SHORT)
##entry_price (next bar's open)
##stop_price (beyond recent swing)
##target_price (based on R-multiple)
##meta (additional information about the trade setup)

The strategy looks for price extensions away from VWAP with a subsequent low-volume pullback, which provides a mean-reversion opportunity back to the VWAP.

'''

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class VWAPFadeStrategy:
    """
    VWAP Fade / Mean Reversion Strategy
    
    Entry: Price extended from VWAP on 3-min with low-volume pullback
    Stop: Small stop beyond recent swing
    Targets: Quick scalps 0.5-1R; repeatable
    Notes: Controlled stops, lower R:R
    """
    
    def __init__(self, params: dict = None):
        """Initialize the VWAP Fade strategy with parameters"""
        self.params = params or {}
        
        # VWAP extension parameters
        self.vwap_extension = self.params.get("vwap_extension", 2.0)  # Points from VWAP to consider extended
        self.extension_periods = self.params.get("extension_periods", 3)  # How many periods price should be extended
        
        # Volume parameters
        self.volume_threshold = self.params.get("volume_threshold", 0.7)  # Ratio to average volume to consider low volume
        self.volume_lookback = self.params.get("volume_lookback", 20)  # Periods to calculate average volume
        
        # Swing detection
        self.swing_lookback = self.params.get("swing_lookback", 5)  # Periods to identify recent swing
        
        # Trade management
        self.stop_buffer = self.params.get("stop_buffer", 0.5)  # Additional buffer beyond swing for stop
        self.target_r = self.params.get("target_r", 1.0)  # Target risk/reward ratio
        self.stop_r = self.params.get("stop_r", 0.75)  # Stop loss as R multiple
        
        # Session filter
        self.session_start = self.params.get("session_start", "08:00")
        self.session_end = self.params.get("session_end", "12:00")

    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate VWAP (Volume Weighted Average Price)"""
        df = df.copy()
        df['pv'] = df['close'] * df['volume']
        df['cumulative_pv'] = df['pv'].cumsum()
        df['cumulative_volume'] = df['volume'].cumsum()
        vwap = df['cumulative_pv'] / df['cumulative_volume']
        return vwap
    
    def in_session(self, dt) -> bool:
        """Check if a given datetime is within the trading session"""
        time = pd.to_datetime(dt).time()
        start_time = pd.to_datetime(self.session_start).time()
        end_time = pd.to_datetime(self.session_end).time()
        return start_time <= time <= end_time
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on the VWAP Fade strategy
        
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
        
        # Create empty signals DataFrame
        cols = ["timestamp", "side", "entry_price", "stop_price", "target_price", "meta"]
        signals = []
        
        # Calculate VWAP
        df['vwap'] = self.calculate_vwap(df)
        
        # Calculate distance from VWAP
        df['vwap_distance'] = df['close'] - df['vwap']
        df['vwap_distance_abs'] = df['vwap_distance'].abs()
        
        # Calculate average volume
        df['volume_sma'] = df['volume'].rolling(window=self.volume_lookback).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Identify price extensions from VWAP
        df['extended_above'] = (df['vwap_distance'] > self.vwap_extension) & \
                              (df['vwap_distance'].shift(1) > self.vwap_extension) & \
                              (df['vwap_distance'].shift(2) > self.vwap_extension)
        
        df['extended_below'] = (df['vwap_distance'] < -self.vwap_extension) & \
                              (df['vwap_distance'].shift(1) < -self.vwap_extension) & \
                              (df['vwap_distance'].shift(2) < -self.vwap_extension)
        
        # Identify low volume pullbacks
        df['low_volume'] = df['volume_ratio'] < self.volume_threshold
        
        # Find recent swings
        df['swing_high'] = df['high'].rolling(window=self.swing_lookback).max()
        df['swing_low'] = df['low'].rolling(window=self.swing_lookback).min()
        
        # Loop through data (skip first few bars for indicator warmup)
        warmup = max(self.volume_lookback, self.swing_lookback, self.extension_periods)
        
        for i in range(warmup, len(df) - 1):
            if not self.in_session(df.index[i]):
                continue
            
            current = df.iloc[i]
            next_bar = df.iloc[i + 1]  # For entry confirmation
            
            # Identify short fade opportunity (price extended above VWAP with low volume pullback)
            short_setup = (
                current['extended_above'] and
                current['low_volume'] and
                current['close'] > current['vwap'] and
                current['close'] < current['high'].shift(1)  # Pullback from high
            )
            
            # Identify long fade opportunity (price extended below VWAP with low volume pullback)
            long_setup = (
                current['extended_below'] and
                current['low_volume'] and
                current['close'] < current['vwap'] and
                current['close'] > current['low'].shift(1)  # Pullback from low
            )
            
            if short_setup:
                # Entry price would be next bar's open
                entry_price = next_bar['open']
                
                # Stop price would be just above the recent swing high
                stop_price = current['swing_high'] + self.stop_buffer
                
                # Calculate target based on risk/reward ratio
                risk = stop_price - entry_price
                target_price = entry_price - (risk * self.target_r)
                
                # Create signal
                signal = {
                    "timestamp": df.index[i + 1],  # Signal activated on next bar
                    "side": "SHORT",
                    "entry_price": entry_price,
                    "stop_price": stop_price,
                    "target_price": target_price,
                    "meta": {
                        "setup": "vwap_fade_short",
                        "vwap": current['vwap'],
                        "vwap_distance": current['vwap_distance'],
                        "volume_ratio": current['volume_ratio'],
                        "swing_high": current['swing_high'],
                        "risk_points": risk,
                        "target_points": entry_price - target_price
                    }
                }
                signals.append(signal)
                
            elif long_setup:
                # Entry price would be next bar's open
                entry_price = next_bar['open']
                
                # Stop price would be just below the recent swing low
                stop_price = current['swing_low'] - self.stop_buffer
                
                # Calculate target based on risk/reward ratio
                risk = entry_price - stop_price
                target_price = entry_price + (risk * self.target_r)
                
                # Create signal
                signal = {
                    "timestamp": df.index[i + 1],  # Signal activated on next bar
                    "side": "LONG",
                    "entry_price": entry_price,
                    "stop_price": stop_price,
                    "target_price": target_price,
                    "meta": {
                        "setup": "vwap_fade_long",
                        "vwap": current['vwap'],
                        "vwap_distance": current['vwap_distance'],
                        "volume_ratio": current['volume_ratio'],
                        "swing_low": current['swing_low'],
                        "risk_points": risk,
                        "target_points": target_price - entry_price
                    }
                }
                signals.append(signal)
        
        # Convert signals list to DataFrame
        return pd.DataFrame(signals)






# import pandas as pd

# class VWAPStrategy:
#     def __init__(self, params: dict):
#         self.params = params or {}
#         self.threshold_points = self.params.get("threshold_points", 2.0)
#         self.target_r = self.params.get("target_r", 1.0)
#         self.stop_r = self.params.get("stop_r", 0.75)

#     def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
#         cols = ["timestamp", "side", "entry_price", "stop_price", "target_price", "meta"]
#         return pd.DataFrame(columns=cols)