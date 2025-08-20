"""
Comprehensive unit tests for core/backtester.py

Tests cover helper functions, TradeRecord dataclass, and Backtester class
including signal processing, trade execution simulation, and metrics calculation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import tempfile
import math

from core.backtester import (
    round_to_tick, 
    get_config_int, 
    TradeRecord, 
    Backtester
)


class TestHelperFunctions:
    """Test standalone helper functions."""
    
    def test_round_to_tick_nearest(self):
        """Test rounding to nearest tick."""
        # Test actual behavior of round_to_tick function
        assert round_to_tick(100.125, 0.25) == 100.0   # rounds down
        assert round_to_tick(100.25, 0.25) == 100.25   # exact
        assert round_to_tick(100.0, 0.25) == 100.0     # exact
        assert round_to_tick(100.375, 0.25) == 100.5   # rounds to nearest (up)
        assert round_to_tick(100.4, 0.25) == 100.5     # rounds up
    
    def test_round_to_tick_up(self):
        """Test rounding up to tick."""
        assert round_to_tick(100.01, 0.25, "up") == 100.25
        assert round_to_tick(100.0, 0.25, "up") == 100.0
        assert round_to_tick(99.99, 0.25, "up") == 100.0
    
    def test_round_to_tick_down(self):
        """Test rounding down to tick."""
        assert round_to_tick(100.24, 0.25, "down") == 100.0
        assert round_to_tick(100.25, 0.25, "down") == 100.25
        assert round_to_tick(100.26, 0.25, "down") == 100.25
    
    def test_round_to_tick_zero_tick_size(self):
        """Test with zero tick size (should return original price)."""
        assert round_to_tick(100.123, 0) == 100.123
        assert round_to_tick(100.123, -1) == 100.123
    
    def test_round_to_tick_invalid_direction(self):
        """Test with invalid direction (should default to nearest)."""
        assert round_to_tick(100.124, 0.25, "invalid") == 100.0
    
    def test_get_config_int_simple_int(self):
        """Test getting integer from simple config value."""
        cfg = {"contracts": 5}
        assert get_config_int(cfg, "contracts") == 5
    
    def test_get_config_int_string_convertible(self):
        """Test getting integer from string value."""
        cfg = {"contracts": "3"}
        assert get_config_int(cfg, "contracts") == 3
    
    def test_get_config_int_dict_with_preferred_key(self):
        """Test getting integer from dict with preferred key."""
        cfg = {"contracts": {"value": 2, "other": 10}}
        assert get_config_int(cfg, "contracts") == 2
    
    def test_get_config_int_dict_with_contracts_key(self):
        """Test getting integer from dict with 'contracts' key."""
        cfg = {"position": {"contracts": 4, "other": 99}}
        assert get_config_int(cfg, "position") == 4
    
    def test_get_config_int_dict_fallback_to_first_value(self):
        """Test fallback to first dict value when no preferred keys."""
        cfg = {"setting": {"some_key": 7}}
        assert get_config_int(cfg, "setting") == 7
    
    def test_get_config_int_missing_key_uses_default(self):
        """Test using default when key is missing."""
        cfg = {}
        assert get_config_int(cfg, "missing_key", default=3) == 3
    
    def test_get_config_int_invalid_value_uses_default(self):
        """Test using default when value cannot be converted."""
        cfg = {"contracts": "invalid"}
        assert get_config_int(cfg, "contracts", default=2) == 2
    
    def test_get_config_int_none_value_uses_default(self):
        """Test using default when value is None."""
        cfg = {"contracts": None}
        assert get_config_int(cfg, "contracts", default=1) == 1


class TestTradeRecord:
    """Test TradeRecord dataclass."""
    
    def test_trade_record_creation(self):
        """Test creating a TradeRecord instance."""
        trade = TradeRecord(
            trade_id=1,
            entry_timestamp=datetime(2025, 1, 15, 10, 0),
            exit_timestamp=datetime(2025, 1, 15, 10, 30),
            side="LONG",
            entry_price=100.0,
            exit_price=102.0,
            contracts=1,
            gross_pnl=10.0,
            commission=2.0,
            slippage_cost=1.0,
            net_pnl=7.0,
            r_multiple=1.4,
            setup="ORB",
            params_snapshot={"test": "data"},
            contract="MES"
        )
        
        assert trade.trade_id == 1
        assert trade.side == "LONG"
        assert trade.entry_price == 100.0
        assert trade.exit_price == 102.0
        assert trade.net_pnl == 7.0
        assert trade.contract == "MES"
    
    def test_trade_record_fields_accessible(self):
        """Test that all fields are accessible."""
        trade = TradeRecord(
            trade_id=2,
            entry_timestamp=datetime.now(),
            exit_timestamp=datetime.now(),
            side="SHORT",
            entry_price=50.0,
            exit_price=48.0,
            contracts=2,
            gross_pnl=20.0,
            commission=3.0,
            slippage_cost=2.0,
            net_pnl=15.0,
            r_multiple=0.75,
            setup="PULLBACK",
            params_snapshot={},
            contract="ES"
        )
        
        # Test all fields are accessible
        assert hasattr(trade, 'trade_id')
        assert hasattr(trade, 'entry_timestamp')
        assert hasattr(trade, 'exit_timestamp')
        assert hasattr(trade, 'side')
        assert hasattr(trade, 'entry_price')
        assert hasattr(trade, 'exit_price')
        assert hasattr(trade, 'contracts')
        assert hasattr(trade, 'gross_pnl')
        assert hasattr(trade, 'commission')
        assert hasattr(trade, 'slippage_cost')
        assert hasattr(trade, 'net_pnl')
        assert hasattr(trade, 'r_multiple')
        assert hasattr(trade, 'setup')
        assert hasattr(trade, 'params_snapshot')
        assert hasattr(trade, 'contract')


class MockStrategy:
    """Mock strategy for testing Backtester."""
    
    def __init__(self, signals_to_return=None):
        if signals_to_return is None:
            self.signals_to_return = pd.DataFrame()
        elif isinstance(signals_to_return, pd.DataFrame):
            self.signals_to_return = signals_to_return
        else:
            self.signals_to_return = pd.DataFrame()
    
    def generate_signals(self, df):
        return self.signals_to_return


class TestBacktester:
    """Test Backtester class initialization and basic functionality."""
    
    def create_test_bars(self, start_time="2025-01-15 09:00:00", periods=20, base_price=100.0):
        """Create test OHLCV data."""
        idx = pd.date_range(start_time, periods=periods, freq="3min", tz="America/New_York")
        
        # Create realistic OHLCV data
        prices = [base_price + i * 0.1 for i in range(periods)]
        df = pd.DataFrame({
            "Open": prices,
            "High": [p + 0.5 for p in prices],
            "Low": [p - 0.5 for p in prices],
            "Close": [p + 0.1 for p in prices],
            "Volume": [1000] * periods
        }, index=idx)
        
        return df
    
    def create_test_config(self):
        """Create test configuration."""
        return {
            "contract_meta": {
                "dollars_per_point": 5.0,
                "tick_size": 0.25
            },
            "contracts": 1,
            "slippage_points": 0.1,
            "commission_roundtrip": 2.0,
            "resolved_contract": "MES",
            "orb": {
                "round_stops_to_tick": True
            }
        }
    
    def test_backtester_initialization(self):
        """Test Backtester initialization."""
        strategy = MockStrategy()
        bars_df = self.create_test_bars()
        config = self.create_test_config()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            bt = Backtester(strategy, bars_df, config, outdir)
            
            assert bt.strategy is strategy
            assert len(bt.bars) == 20
            assert bt.config == config
            assert bt.outdir == outdir
            assert bt.trades == []
            assert bt.trade_id_seq == 1
            assert bt.dollars_per_point == 5.0
    
    def test_backtester_initialization_missing_contract_meta(self):
        """Test initialization fails gracefully with missing contract metadata."""
        strategy = MockStrategy()
        bars_df = self.create_test_bars()
        config = {}  # Missing contract_meta
        
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            with pytest.raises(KeyError):
                Backtester(strategy, bars_df, config, outdir)
    
    def test_backtester_run_no_signals(self):
        """Test backtester run with no signals."""
        strategy = MockStrategy()  # Returns empty DataFrame
        bars_df = self.create_test_bars()
        config = self.create_test_config()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            bt = Backtester(strategy, bars_df, config, outdir)
            bt.run()
            
            assert len(bt.trades) == 0
            assert bt.trade_id_seq == 1
    
    def test_backtester_run_with_signal(self):
        """Test backtester run with a signal."""
        # Create a signal DataFrame
        signal_time = pd.Timestamp("2025-01-15 09:06:00", tz="America/New_York")
        signals = pd.DataFrame([{
            "timestamp": signal_time,
            "side": "LONG",
            "signal_price": 100.0,
            "entry_price_hint": None,
            "stop_price": 99.0,
            "target1": 102.0,
            "target2": 104.0,
            "meta": {"test": "data"}
        }])
        
        strategy = MockStrategy(signals)
        bars_df = self.create_test_bars()
        config = self.create_test_config()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            bt = Backtester(strategy, bars_df, config, outdir)
            bt.run()
            
            # Should have executed at least one trade
            assert len(bt.trades) > 0
            assert bt.trade_id_seq > 1
    
    def test_backtester_column_case_handling(self):
        """Test that backtester handles different column cases correctly."""
        strategy = MockStrategy()
        bars_df = self.create_test_bars()
        config = self.create_test_config()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            bt = Backtester(strategy, bars_df, config, outdir)
            
            # Verify original bars have Title case
            assert "Open" in bt.bars.columns
            assert "High" in bt.bars.columns
            
            # Run should convert to lowercase for strategy
            bt.run()
            
            # Verify original bars still have Title case
            assert "Open" in bt.bars.columns
            assert "High" in bt.bars.columns
    
    def test_compute_metrics_no_trades(self):
        """Test metrics computation with no trades."""
        strategy = MockStrategy()
        bars_df = self.create_test_bars()
        config = self.create_test_config()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            bt = Backtester(strategy, bars_df, config, outdir)
            
            metrics = bt.compute_metrics()
            
            assert metrics["total_trades"] == 0
            assert metrics["gross_pnl"] == 0.0
            assert metrics["net_pnl"] == 0.0
            assert metrics["wins"] == 0
            assert metrics["losses"] == 0
            assert metrics["win_rate"] == 0.0
    
    def test_compute_metrics_with_trades(self):
        """Test metrics computation with sample trades."""
        strategy = MockStrategy()
        bars_df = self.create_test_bars()
        config = self.create_test_config()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            bt = Backtester(strategy, bars_df, config, outdir)
            
            # Add sample trades
            bt.trades = [
                TradeRecord(
                    trade_id=1, entry_timestamp=datetime.now(), exit_timestamp=datetime.now(),
                    side="LONG", entry_price=100.0, exit_price=102.0, contracts=1,
                    gross_pnl=10.0, commission=2.0, slippage_cost=1.0, net_pnl=7.0,
                    r_multiple=1.4, setup="ORB", params_snapshot={}, contract="MES"
                ),
                TradeRecord(
                    trade_id=2, entry_timestamp=datetime.now(), exit_timestamp=datetime.now(),
                    side="SHORT", entry_price=100.0, exit_price=102.0, contracts=1,
                    gross_pnl=-10.0, commission=2.0, slippage_cost=1.0, net_pnl=-13.0,
                    r_multiple=-0.65, setup="ORB", params_snapshot={}, contract="MES"
                )
            ]
            
            metrics = bt.compute_metrics()
            performance = metrics["performance"]
            
            assert performance["total_trades"] == 2
            assert performance["gross_pnl"] == 0.0  # 10 + (-10)
            assert performance["net_pnl"] == -6.0   # 7 + (-13)
            assert performance["wins"] == 1
            assert performance["losses"] == 1
            assert performance["win_rate"] == 0.5
    
    def test_save_results_no_trades(self):
        """Test saving results with no trades."""
        strategy = MockStrategy()
        bars_df = self.create_test_bars()
        config = self.create_test_config()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            bt = Backtester(strategy, bars_df, config, outdir)
            bt.save_results()
            
            # Check files are created
            trades_file = outdir / "trades.csv"
            summary_file = outdir / "summary.json"
            
            assert trades_file.exists()
            assert summary_file.exists()
            
            # Check trades CSV has header
            with open(trades_file, 'r') as f:
                header = f.readline().strip()
                assert "trade_id" in header
                assert "entry_timestamp" in header
                assert "net_pnl" in header
            
            # Check summary JSON
            with open(summary_file, 'r') as f:
                summary = json.load(f)
                assert summary["total_trades"] == 0
                assert summary["net_pnl"] == 0.0


class TestBacktesterSignalProcessing:
    """Test signal processing and trade execution logic."""
    
    def create_test_bars_with_movement(self):
        """Create test bars with price movement for signal testing."""
        idx = pd.date_range("2025-01-15 09:00:00", periods=10, freq="3min", tz="America/New_York")
        
        # Create bars where price moves up to hit targets
        prices = [100.0, 100.1, 100.2, 101.0, 102.5, 103.0, 102.0, 101.5, 101.0, 100.5]
        df = pd.DataFrame({
            "Open": prices,
            "High": [p + 1.0 for p in prices],  # Higher highs for target hits
            "Low": [p - 0.5 for p in prices],
            "Close": prices,
            "Volume": [1000] * 10
        }, index=idx)
        
        return df
    
    def test_signal_processing_long_trade(self):
        """Test processing a long signal that hits target."""
        # Create signal that should execute
        signal_time = pd.Timestamp("2025-01-15 09:00:00", tz="America/New_York")  # Before first bar
        signals = pd.DataFrame([{
            "timestamp": signal_time,
            "side": "LONG",
            "signal_price": 100.0,
            "entry_price_hint": None,
            "stop_price": 99.0,
            "target1": 101.5,
            "target2": 103.0,
            "meta": {"test": "long_signal"}
        }])
        
        strategy = MockStrategy(signals)
        bars_df = self.create_test_bars_with_movement()
        config = {
            "contract_meta": {"dollars_per_point": 5.0, "tick_size": 0.25},
            "contracts": 1,
            "slippage_points": 0.1,
            "commission_roundtrip": 2.0,
            "resolved_contract": "MES",
            "orb": {"round_stops_to_tick": False}
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            bt = Backtester(strategy, bars_df, config, outdir)
            bt.run()
            
            # Should have generated trades
            assert len(bt.trades) > 0
            
            # Check first trade is LONG
            first_trade = bt.trades[0]
            assert first_trade.side == "LONG"
            assert first_trade.contracts > 0
    
    def test_signal_processing_short_trade(self):
        """Test processing a short signal."""
        signal_time = pd.Timestamp("2025-01-15 09:00:00", tz="America/New_York")
        signals = pd.DataFrame([{
            "timestamp": signal_time,
            "side": "SHORT",
            "signal_price": 100.0,
            "entry_price_hint": None,
            "stop_price": 101.0,
            "target1": 98.5,
            "target2": 97.0,
            "meta": {"test": "short_signal"}
        }])
        
        strategy = MockStrategy(signals)
        bars_df = self.create_test_bars_with_movement()
        config = {
            "contract_meta": {"dollars_per_point": 5.0, "tick_size": 0.25},
            "contracts": 1,
            "slippage_points": 0.1,
            "commission_roundtrip": 2.0,
            "resolved_contract": "MES",
            "orb": {"round_stops_to_tick": False}
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            bt = Backtester(strategy, bars_df, config, outdir)
            bt.run()
            
            # Should have generated trades
            assert len(bt.trades) > 0
            
            # Check first trade is SHORT
            first_trade = bt.trades[0]
            assert first_trade.side == "SHORT"
    
    def test_no_next_bar_available(self):
        """Test signal at end of data with no next bar."""
        # Signal at the very last timestamp
        signal_time = pd.Timestamp("2025-01-15 09:27:00", tz="America/New_York")  # Last bar
        signals = pd.DataFrame([{
            "timestamp": signal_time,
            "side": "LONG", 
            "signal_price": 100.0,
            "entry_price_hint": None,
            "stop_price": 99.0,
            "target1": 101.0,
            "target2": 102.0,
            "meta": {"test": "late_signal"}
        }])
        
        strategy = MockStrategy(signals)
        bars_df = self.create_test_bars_with_movement()
        config = {
            "contract_meta": {"dollars_per_point": 5.0, "tick_size": 0.25},
            "contracts": 1,
            "slippage_points": 0.1,
            "commission_roundtrip": 2.0,
            "resolved_contract": "MES"
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            bt = Backtester(strategy, bars_df, config, outdir)
            bt.run()
            
            # Should not generate trades (no next bar to enter)
            assert len(bt.trades) == 0


class TestBacktesterEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_none_signals(self):
        """Test handling when strategy returns None."""
        strategy = MockStrategy(None)
        bars_df = pd.DataFrame({"Open": [100], "High": [101], "Low": [99], "Close": [100], "Volume": [1000]})
        config = {"contract_meta": {"dollars_per_point": 5.0}}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            bt = Backtester(strategy, bars_df, config, outdir)
            bt.run()
            
            assert len(bt.trades) == 0
    
    def test_empty_bars_df(self):
        """Test with empty bars DataFrame."""
        strategy = MockStrategy()
        bars_df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        config = {"contract_meta": {"dollars_per_point": 5.0}}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            bt = Backtester(strategy, bars_df, config, outdir)
            bt.run()
            
            assert len(bt.trades) == 0
    
    def test_malformed_signal_data(self):
        """Test handling malformed signal data."""
        # Signal missing required fields
        signals = pd.DataFrame([{
            "timestamp": pd.Timestamp("2025-01-15 09:00:00", tz="America/New_York"),
            "side": "LONG",
            # Missing signal_price, stop_price, etc.
        }])
        
        strategy = MockStrategy(signals)
        # Create bars with datetime index to match timestamp comparison
        idx = pd.date_range("2025-01-15 08:00:00", periods=2, freq="1h", tz="America/New_York")
        bars_df = pd.DataFrame({
            "Open": [100, 101], "High": [101, 102], "Low": [99, 100], 
            "Close": [100, 101], "Volume": [1000, 1000]
        }, index=idx)
        config = {"contract_meta": {"dollars_per_point": 5.0}}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            bt = Backtester(strategy, bars_df, config, outdir)
            
            # Should not crash, but may not generate valid trades due to missing fields
            bt.run()
            # Test passes if no exception is raised
    
    def test_config_with_dict_contracts(self):
        """Test config with contracts as dict (using get_config_int)."""
        strategy = MockStrategy()
        bars_df = pd.DataFrame({
            "Open": [100], "High": [101], "Low": [99], "Close": [100], "Volume": [1000]
        })
        config = {
            "contract_meta": {"dollars_per_point": 5.0},
            "contracts": {"value": 2}  # Dict format
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            bt = Backtester(strategy, bars_df, config, outdir)
            bt.run()
            
            # Should handle dict contracts via get_config_int
            assert len(bt.trades) == 0  # No signals, but no crash


if __name__ == "__main__":
    pytest.main([__file__])