import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class MESScalpingStrategy:
    """
    MES Scalping Strategy
    
    Uses EMA crossovers with RSI and MACD confirmation for quick scalp trades
    during the 08:00-12:00 ET session, focusing on 3-minute charts with 1-min entries
    and 10-min bias.


    This MES Scalping Strategy implementation:

    #Follows your skeleton structure exactly
    #Focuses on scalping the MES contract during your preferred 08:00-12:00 ET session
    #Uses EMA crossovers as the primary entry signal
    #Adds RSI and MACD confirmation to filter out weak setups
    #Calculates stops based on ATR or fixed tick values
    #Sets targets based on risk-reward ratio
    #Returns signals in the required format (timestamp, side, entry_price, stop_price, target_price, meta)
    
    The strategy generates signals when:

    #For longs: Fast EMA crosses above Medium EMA with RSI > 50 and increasing MACD histogram
    #For shorts: Fast EMA crosses below Medium EMA with RSI < 50 and decreasing MACD histogram
    
    Entry is on the next bar's open with stops based on ATR or tick distance and targets based on your preferred R-multiple.



    """
    
    def __init__(self, params: dict = None):
        """Initialize the MES Scalping strategy with parameters"""
        self.params = params or {}
        
        # EMA parameters
        self.fast_ema = self.params.get("fast_ema", 9)
        self.medium_ema = self.params.get("medium_ema", 21)
        self.slow_ema = self.params.get("slow_ema", 50)
        
        # RSI parameters
        self.rsi_period = self.params.get("rsi_period", 14)
        self.rsi_overbought = self.params.get("rsi_overbought", 70)
        self.rsi_oversold = self.params.get("rsi_oversold", 30)
        self.rsi_midline = self.params.get("rsi_midline", 50)
        
        # MACD parameters
        self.macd_fast = self.params.get("macd_fast", 12)
        self.macd_slow = self.params.get("macd_slow", 26)
        self.macd_signal = self.params.get("macd_signal", 9)
        
        # Trade management
        self.threshold_points = self.params.get("threshold_points", 2.0)  # Points for signal confirmation
        self.target_r = self.params.get("target_r", 1.5)  # Target risk/reward ratio
        self.stop_r = self.params.get("stop_r", 0.75)  # Stop loss as R multiple
        self.take_profit_ticks = self.params.get("take_profit_ticks", 10)  # Take profit in ticks
        self.stop_loss_ticks = self.params.get("stop_loss_ticks", 5)  # Stop loss in ticks
        
        # Contract specifications
        self.tick_size = self.params.get("tick_size", 0.25)  # MES tick size is 0.25 points
        
        # Session filter (required)
        self.session_start = self.params.get("session_start")
        self.session_end = self.params.get("session_end")
        if not self.session_start or not self.session_end:
            raise ValueError("Scalping strategy requires session_start and session_end in config")

    def in_session(self, dt) -> bool:
        """Check if a given datetime is within the trading session"""
        time = pd.to_datetime(dt).time()
        start_time = pd.to_datetime(self.session_start).time()
        end_time = pd.to_datetime(self.session_end).time()
        return start_time <= time <= end_time
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on the MES Scalping strategy
        
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
        
        # Create a copy to avoid modifying the original DataFrame
        df = df.copy()
        
        # Calculate indicators
        # EMAs
        df[f'ema_{self.fast_ema}'] = df['close'].ewm(span=self.fast_ema, adjust=False).mean()
        df[f'ema_{self.medium_ema}'] = df['close'].ewm(span=self.medium_ema, adjust=False).mean()
        df[f'ema_{self.slow_ema}'] = df['close'].ewm(span=self.slow_ema, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['macd_line'] = df['close'].ewm(span=self.macd_fast, adjust=False).mean() - \
                          df['close'].ewm(span=self.macd_slow, adjust=False).mean()
        df['macd_signal'] = df['macd_line'].ewm(span=self.macd_signal, adjust=False).mean()
        df['macd_histogram'] = df['macd_line'] - df['macd_signal']
        df['macd_histogram_diff'] = df['macd_histogram'].diff()
        
        # Calculate ATR for stop placement
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # Create empty signals DataFrame
        cols = ["timestamp", "side", "entry_price", "stop_price", "target_price", "meta"]
        signals = []
        
        # Loop through data (skip first few bars for indicator warmup)
        warmup = max(self.slow_ema, self.rsi_period, self.macd_slow + self.macd_signal)
        
        for i in range(warmup + 1, len(df) - 1):
            # Skip if not in trading session
            if not self.in_session(df.index[i]):
                continue
                
            current = df.iloc[i]
            prev = df.iloc[i-1]
            next_bar = df.iloc[i+1]  # For entry confirmation
            
            # Fast EMA crossover of Medium EMA for potential long setup
            long_ema_cross = (prev[f'ema_{self.fast_ema}'] <= prev[f'ema_{self.medium_ema}']) and \
                             (current[f'ema_{self.fast_ema}'] > current[f'ema_{self.medium_ema}'])
            
            # Additional confirmation for long entry
            long_confirmation = (
                current['rsi'] > self.rsi_midline and  # RSI above midline (bullish)
                current['macd_histogram_diff'] > 0 and  # MACD histogram increasing (momentum)
                current['close'] > current[f'ema_{self.fast_ema}']  # Price above fast EMA
            )
            
            # Fast EMA crossover of Medium EMA for potential short setup
            short_ema_cross = (prev[f'ema_{self.fast_ema}'] >= prev[f'ema_{self.medium_ema}']) and \
                              (current[f'ema_{self.fast_ema}'] < current[f'ema_{self.medium_ema}'])
            
            # Additional confirmation for short entry
            short_confirmation = (
                current['rsi'] < self.rsi_midline and  # RSI below midline (bearish)
                current['macd_histogram_diff'] < 0 and  # MACD histogram decreasing (momentum)
                current['close'] < current[f'ema_{self.fast_ema}']  # Price below fast EMA
            )
            
            # Generate long signals
            if long_ema_cross and long_confirmation:
                # Entry price would be next bar's open
                entry_price = next_bar['open']
                
                # Stop price - use ATR or fixed tick-based stop
                if 'atr' in current and not pd.isna(current['atr']):
                    stop_distance = max(self.stop_loss_ticks * self.tick_size, current['atr'] * 0.5)
                else:
                    stop_distance = self.stop_loss_ticks * self.tick_size
                
                stop_price = entry_price - stop_distance
                
                # Target price - based on R multiple
                risk = entry_price - stop_price
                target_price = entry_price + (risk * self.target_r)
                
                # Create signal
                signal = {
                    "timestamp": df.index[i+1],  # Signal activated on next bar
                    "side": "LONG",
                    "entry_price": entry_price,
                    "stop_price": stop_price,
                    "target_price": target_price,
                    "meta": {
                        "setup": "mes_scalp_long_ema_cross",
                        "rsi": current['rsi'],
                        "macd_histogram": current['macd_histogram'],
                        "macd_histogram_diff": current['macd_histogram_diff'],
                        "fast_ema": current[f'ema_{self.fast_ema}'],
                        "medium_ema": current[f'ema_{self.medium_ema}'],
                        "risk_points": risk,
                        "target_points": target_price - entry_price
                    }
                }
                signals.append(signal)
            
            # Generate short signals
            elif short_ema_cross and short_confirmation:
                # Entry price would be next bar's open
                entry_price = next_bar['open']
                
                # Stop price - use ATR or fixed tick-based stop
                if 'atr' in current and not pd.isna(current['atr']):
                    stop_distance = max(self.stop_loss_ticks * self.tick_size, current['atr'] * 0.5)
                else:
                    stop_distance = self.stop_loss_ticks * self.tick_size
                
                stop_price = entry_price + stop_distance
                
                # Target price - based on R multiple
                risk = stop_price - entry_price
                target_price = entry_price - (risk * self.target_r)
                
                # Create signal
                signal = {
                    "timestamp": df.index[i+1],  # Signal activated on next bar
                    "side": "SHORT",
                    "entry_price": entry_price,
                    "stop_price": stop_price,
                    "target_price": target_price,
                    "meta": {
                        "setup": "mes_scalp_short_ema_cross",
                        "rsi": current['rsi'],
                        "macd_histogram": current['macd_histogram'],
                        "macd_histogram_diff": current['macd_histogram_diff'],
                        "fast_ema": current[f'ema_{self.fast_ema}'],
                        "medium_ema": current[f'ema_{self.medium_ema}'],
                        "risk_points": risk,
                        "target_points": entry_price - target_price
                    }
                }
                signals.append(signal)
                
        # Convert signals list to DataFrame
        return pd.DataFrame(signals)

# {
#   "strategy_name": "MES Scalping Strategy",
#   "asset": "MES",
#   "timeframes": {
#     "primary": "3min",
#     "entry": "1min",
#     "bias": "10min"
#   },
#   "trading_hours": {
#     "start": "08:00",
#     "end": "12:00",
#     "timezone": "ET"
#   },
#   "indicators": {
#     "moving_averages": {
#       "fast_ema": 9,
#       "medium_ema": 21,
#       "slow_ema": 50
#     },
#     "momentum": {
#       "rsi": {
#         "period": 14,
#         "overbought": 70,
#         "oversold": 30
#       },
#       "macd": {
#         "fast_period": 12,
#         "slow_period": 26,
#         "signal_period": 9
#       }
#     },
#     "volatility": {
#       "atr": {
#         "period": 14
#       },
#       "bollinger_bands": {
#         "period": 20,
#         "std_dev": 2
#       }
#     }
#   },
#   "entry_rules": {
#     "long": [
#       "price crosses above fast_ema",
#       "rsi > 50",
#       "macd histogram increasing"
#     ],
#     "short": [
#       "price crosses below fast_ema",
#       "rsi < 50",
#       "macd histogram decreasing"
#     ]
#   },
#   "exit_rules": {
#     "take_profit": {
#       "ticks": 10
#     },
#     "stop_loss": {
#       "ticks": 5
#     },
#     "trailing_stop": {
#       "activation_ticks": 8,
#       "trail_ticks": 4
#     },
#     "time_based": {
#       "max_trade_duration_minutes": 30
#     }
#   },
#   "risk_management": {
#     "max_daily_loss": 200,
#     "max_position_size": 2,
#     "risk_per_trade_dollars": 40,
#     "daily_profit_target": 100
#   },
#   "position_sizing": {
#     "initial_contracts": 1,
#     "max_contracts": 2,
#     "scaling": "none"
#   },
#   "initial_capital": 2000,
#   "commission_per_contract": 2.50,
#   "slippage_ticks": 1
# }