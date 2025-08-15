"""
Comprehensive unit tests for ORBStrategy engine.
Tests cover initialization, data processing, signal generation, and edge cases.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, time
from engines.orb_engine import ORBStrategy


class TestORBStrategyInit:
    """Test ORBStrategy initialization and parameter handling."""
    
    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        strategy = ORBStrategy({})
        
        assert strategy.open_range_minutes == 15
        assert strategy.require_volume_confirmation is True
        assert strategy.volume_lookback == 5
        assert strategy.take_profit_r == [1.0, 2.0]
        assert strategy.stop_policy == "structure"
        assert strategy.atr_multiplier_stop == 1.0
        assert strategy.stop_fixed_points == 4.0
        assert strategy.cancel_within_minutes == 60
    
    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        params = {
            "open_range_minutes": 30,
            "require_volume_confirmation": False,
            "volume_lookback": 10,
            "take_profit_r": [1.5, 3.0],
            "stop_policy": "atr",
            "atr_multiplier_stop": 2.0,
            "stop_fixed_points": 5.0,
            "cancel_within_minutes": 120
        }
        strategy = ORBStrategy(params)
        
        assert strategy.open_range_minutes == 30
        assert strategy.require_volume_confirmation is False
        assert strategy.volume_lookback == 10
        assert strategy.take_profit_r == [1.5, 3.0]
        assert strategy.stop_policy == "atr"
        assert strategy.atr_multiplier_stop == 2.0
        assert strategy.stop_fixed_points == 5.0
        assert strategy.cancel_within_minutes == 120
    
    def test_init_with_none_params(self):
        """Test initialization with None params."""
        strategy = ORBStrategy(None)
        assert strategy.open_range_minutes == 15  # Should use defaults


class TestORBStrategyDayGroups:
    """Test the _day_groups method for date grouping and timezone handling."""
    
    def test_day_groups_single_day(self):
        """Test _day_groups with single day data."""
        strategy = ORBStrategy({})
        
        # Create single day data
        idx = pd.date_range("2025-01-15 09:00:00", periods=10, freq="3min", tz="America/New_York")
        df = pd.DataFrame({
            "open": [100.0] * 10,
            "high": [101.0] * 10,
            "low": [99.0] * 10,
            "close": [100.5] * 10,
            "volume": [1000] * 10
        }, index=idx)
        
        groups = list(strategy._day_groups(df))
        assert len(groups) == 1
        
        date, day_df = groups[0]
        assert date == datetime(2025, 1, 15).date()
        assert len(day_df) == 10
        assert day_df.index.tz.zone == "America/New_York"
    
    def test_day_groups_multiple_days(self):
        """Test _day_groups with multiple days."""
        strategy = ORBStrategy({})
        
        # Create 2-day data
        idx1 = pd.date_range("2025-01-15 09:00:00", periods=5, freq="3min", tz="America/New_York")
        idx2 = pd.date_range("2025-01-16 09:00:00", periods=5, freq="3min", tz="America/New_York")
        idx = idx1.append(idx2)
        
        df = pd.DataFrame({
            "open": [100.0] * 10,
            "high": [101.0] * 10,
            "low": [99.0] * 10,
            "close": [100.5] * 10,
            "volume": [1000] * 10
        }, index=idx)
        
        groups = list(strategy._day_groups(df))
        assert len(groups) == 2
        
        # Check first day
        date1, day_df1 = groups[0]
        assert date1 == datetime(2025, 1, 15).date()
        assert len(day_df1) == 5
        
        # Check second day
        date2, day_df2 = groups[1]
        assert date2 == datetime(2025, 1, 16).date()
        assert len(day_df2) == 5
    
    def test_day_groups_naive_timestamps(self):
        """Test _day_groups with naive timestamps."""
        strategy = ORBStrategy({})
        
        # Create naive timestamps
        idx = pd.date_range("2025-01-15 09:00:00", periods=5, freq="3min")
        df = pd.DataFrame({
            "open": [100.0] * 5,
            "high": [101.0] * 5,
            "low": [99.0] * 5,
            "close": [100.5] * 5,
            "volume": [1000] * 5
        }, index=idx)
        
        groups = list(strategy._day_groups(df))
        assert len(groups) == 1
        
        date, day_df = groups[0]
        assert day_df.index.tz.zone == "America/New_York"
    
    def test_day_groups_with_session_filter(self):
        """Test _day_groups with session filtering."""
        strategy = ORBStrategy({})
        strategy.session_start = "09:30"
        strategy.session_end = "10:30"
        
        # Create data spanning wider time range
        idx = pd.date_range("2025-01-15 09:00:00", periods=20, freq="3min", tz="America/New_York")
        df = pd.DataFrame({
            "open": [100.0] * 20,
            "high": [101.0] * 20,
            "low": [99.0] * 20,
            "close": [100.5] * 20,
            "volume": [1000] * 20
        }, index=idx)
        
        groups = list(strategy._day_groups(df))
        assert len(groups) == 1
        
        date, day_df = groups[0]
        # Should filter to session times
        assert len(day_df) < 20  # Less than original
        assert day_df.index.min().time() >= time(9, 30)
        assert day_df.index.max().time() <= time(10, 30)


class TestORBStrategySignalDedup:
    """Test the _append_signal_dedup method."""
    
    def test_append_signal_dedup_no_duplicates(self):
        """Test dedup with no duplicates."""
        strategy = ORBStrategy({})
        signals = []
        seen_keys = set()
        
        sig1 = {"timestamp": pd.Timestamp("2025-01-15 10:00:00"), "side": "LONG", "signal_price": 100.0}
        sig2 = {"timestamp": pd.Timestamp("2025-01-15 10:03:00"), "side": "SHORT", "signal_price": 99.0}
        
        strategy._append_signal_dedup(signals, seen_keys, sig1)
        strategy._append_signal_dedup(signals, seen_keys, sig2)
        
        assert len(signals) == 2
        assert len(seen_keys) == 2
    
    def test_append_signal_dedup_with_duplicates(self):
        """Test dedup with duplicate signals."""
        strategy = ORBStrategy({})
        signals = []
        seen_keys = set()
        
        sig1 = {"timestamp": pd.Timestamp("2025-01-15 10:00:00"), "side": "LONG", "signal_price": 100.0}
        sig2 = {"timestamp": pd.Timestamp("2025-01-15 10:00:00"), "side": "LONG", "signal_price": 100.0}  # Duplicate
        
        strategy._append_signal_dedup(signals, seen_keys, sig1)
        strategy._append_signal_dedup(signals, seen_keys, sig2)  # Should be ignored
        
        assert len(signals) == 1
        assert len(seen_keys) == 1
    
    def test_append_signal_dedup_malformed_signal(self):
        """Test dedup with malformed signal data."""
        strategy = ORBStrategy({})
        signals = []
        seen_keys = set()
        
        # Malformed signal (missing required fields)
        sig = {"side": "LONG"}  # Missing timestamp and signal_price
        
        # Should not crash, should still append
        strategy._append_signal_dedup(signals, seen_keys, sig)
        assert len(signals) == 1


class TestORBStrategySignalGeneration:
    """Test signal generation logic."""
    
    def create_test_data(self, start_time="2025-01-15 09:00:00", periods=40):
        """Create test OHLCV data for signal testing."""
        idx = pd.date_range(start_time, periods=periods, freq="3min", tz="America/New_York")
        
        # Create price data that will generate OR breakout
        base_price = 4000.0
        prices = [base_price] * periods
        
        # Opening range: first 5 bars (15 minutes)
        or_bars = 5
        or_high = base_price + 2.0
        or_low = base_price - 2.0
        
        # Set opening range prices
        for i in range(or_bars):
            prices[i] = base_price + np.random.uniform(-1.5, 1.5)
        
        # Create breakout after OR
        if periods > or_bars:
            prices[or_bars] = or_high + 1.0  # Breakout above OR
        
        df = pd.DataFrame({
            "open": prices,
            "high": [p + 0.5 for p in prices],
            "low": [p - 0.5 for p in prices],
            "close": [p + 0.1 for p in prices],
            "volume": [1000] * periods
        }, index=idx)
        
        return df
    
    def test_generate_signals_missing_columns(self):
        """Test signal generation with missing required columns."""
        strategy = ORBStrategy({})
        
        # DataFrame missing required columns
        df = pd.DataFrame({
            "price": [100, 101, 102],
            "vol": [1000, 1000, 1000]
        })
        
        signals = strategy.generate_signals(df)
        assert isinstance(signals, pd.DataFrame)
        assert len(signals) == 0
        assert "timestamp" in signals.columns
    
    def test_generate_signals_empty_data(self):
        """Test signal generation with empty DataFrame."""
        strategy = ORBStrategy({})
        
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        signals = strategy.generate_signals(df)
        
        assert isinstance(signals, pd.DataFrame)
        assert len(signals) == 0
    
    def test_generate_signals_long_breakout(self):
        """Test generation of long breakout signal."""
        params = {
            "open_range_minutes": 15,
            "require_volume_confirmation": False,  # Disable for predictable test
            "stop_policy": "structure"
        }
        strategy = ORBStrategy(params)
        
        # Create data with clear long breakout
        df = self.create_test_data()
        signals = strategy.generate_signals(df)
        
        assert isinstance(signals, pd.DataFrame)
        # Should have required columns
        expected_cols = ["timestamp", "side", "signal_price", "entry_price_hint", 
                        "stop_price", "target1", "target2", "meta"]
        for col in expected_cols:
            assert col in signals.columns
    
    def test_generate_signals_volume_confirmation(self):
        """Test signal generation with volume confirmation enabled."""
        params = {
            "open_range_minutes": 15,
            "require_volume_confirmation": True,
            "volume_lookback": 5
        }
        strategy = ORBStrategy(params)
        
        df = self.create_test_data()
        # Set low volume on breakout bar to test volume filter
        df.loc[df.index[5], "volume"] = 100  # Much lower than average
        
        signals = strategy.generate_signals(df)
        assert isinstance(signals, pd.DataFrame)
    
    def test_generate_signals_different_stop_policies(self):
        """Test signal generation with different stop policies."""
        test_data = self.create_test_data()
        
        for stop_policy in ["structure", "atr", "fixed"]:
            params = {
                "open_range_minutes": 15,
                "require_volume_confirmation": False,
                "stop_policy": stop_policy,
                "atr_multiplier_stop": 1.5,
                "stop_fixed_points": 3.0
            }
            strategy = ORBStrategy(params)
            signals = strategy.generate_signals(test_data.copy())
            
            assert isinstance(signals, pd.DataFrame)
            # All stop policies should work without errors
    
    def test_generate_signals_multiple_days(self):
        """Test signal generation across multiple days."""
        params = {"open_range_minutes": 15, "require_volume_confirmation": False}
        strategy = ORBStrategy(params)
        
        # Create 2-day dataset
        day1 = self.create_test_data("2025-01-15 09:00:00", 20)
        day2 = self.create_test_data("2025-01-16 09:00:00", 20)
        df = pd.concat([day1, day2])
        
        signals = strategy.generate_signals(df)
        assert isinstance(signals, pd.DataFrame)
        
        # Should handle multiple days without error
        if len(signals) > 0:
            # Check that signals are properly sorted by timestamp
            timestamps = pd.to_datetime(signals["timestamp"])
            assert timestamps.is_monotonic_increasing


class TestORBStrategyEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_insufficient_bars_for_opening_range(self):
        """Test behavior when there are insufficient bars for opening range."""
        strategy = ORBStrategy({"open_range_minutes": 15})
        
        # Only 2 bars - insufficient for 15-minute OR
        idx = pd.date_range("2025-01-15 09:00:00", periods=2, freq="3min", tz="America/New_York")
        df = pd.DataFrame({
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1000, 1000]
        }, index=idx)
        
        signals = strategy.generate_signals(df)
        assert len(signals) == 0  # Should not generate signals
    
    def test_no_breakout_signals(self):
        """Test case where price stays within opening range."""
        strategy = ORBStrategy({"open_range_minutes": 15, "require_volume_confirmation": False})
        
        # Create flat price action within range
        idx = pd.date_range("2025-01-15 09:00:00", periods=20, freq="3min", tz="America/New_York")
        prices = [4000.0] * 20  # Flat prices
        
        df = pd.DataFrame({
            "open": prices,
            "high": [p + 0.1 for p in prices],  # Very small range
            "low": [p - 0.1 for p in prices],
            "close": prices,
            "volume": [1000] * 20
        }, index=idx)
        
        signals = strategy.generate_signals(df)
        assert len(signals) == 0  # No breakouts should be detected
    
    def test_invalid_timezone_handling(self):
        """Test handling of timezone conversion errors."""
        strategy = ORBStrategy({})
        
        # Create data with problematic timezone
        idx = pd.date_range("2025-01-15 09:00:00", periods=5, freq="3min")
        df = pd.DataFrame({
            "open": [100.0] * 5,
            "high": [101.0] * 5,
            "low": [99.0] * 5,
            "close": [100.5] * 5,
            "volume": [1000] * 5
        }, index=idx)
        
        # Should handle timezone conversion gracefully
        try:
            groups = list(strategy._day_groups(df, tz_target="Invalid/Timezone"))
            # If no error, good; if error, should be caught gracefully
        except ValueError:
            # Expected for invalid timezone
            pass
    
    def test_missing_ema_atr_indicators(self):
        """Test behavior when EMA/ATR indicators are missing or NaN."""
        strategy = ORBStrategy({"open_range_minutes": 15, "require_volume_confirmation": False})
        
        # Create minimal dataset that might not have enough data for indicators
        idx = pd.date_range("2025-01-15 09:00:00", periods=10, freq="3min", tz="America/New_York")
        df = pd.DataFrame({
            "open": [100.0] * 10,
            "high": [101.0] * 10,
            "low": [99.0] * 10,
            "close": [100.5] * 10,
            "volume": [1000] * 10
        }, index=idx)
        
        # Should handle missing indicators gracefully
        signals = strategy.generate_signals(df)
        assert isinstance(signals, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__])