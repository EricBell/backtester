"""
core/backtester.py - Backtester that integrates Strategy objects and simulates execution.

This module provides:
1. TradeRecord dataclass for storing individual trade results
2. Helper functions for price rounding and config parsing
3. Backtester class that simulates trade execution from strategy signals
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import math


def round_to_tick(price: float, tick_size: float, direction: str = "nearest") -> float:
    """
    Round price to nearest tick size.
    
    Args:
        price: Price to round
        tick_size: Minimum tick size
        direction: "nearest", "up", or "down"
        
    Returns:
        Rounded price
    """
    if tick_size <= 0:
        return price
    
    if direction == "nearest":
        return round(price / tick_size) * tick_size
    elif direction == "up":
        return math.ceil(price / tick_size) * tick_size
    elif direction == "down":
        return math.floor(price / tick_size) * tick_size
    else:
        return round(price / tick_size) * tick_size


def get_config_int(cfg: dict, key: str, default: int = 1, 
                   preferred_keys: tuple = ("value", "val", "contracts", "qty", "quantity", "n", "count")) -> int:
    """
    Safely extract integer from config, handling both simple values and nested dicts.
    
    Args:
        cfg: Configuration dictionary
        key: Key to extract
        default: Default value if extraction fails
        preferred_keys: Preferred keys to check in nested dicts
        
    Returns:
        Integer value or default
    """
    raw = cfg.get(key, default)
    
    if isinstance(raw, dict):
        # Try preferred keys first
        for k in preferred_keys:
            if k in raw:
                raw = raw[k]
                break
        else:
            # Fallback to first available value
            try:
                raw = next(iter(raw.values()))
            except Exception:
                raw = default
    
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


@dataclass
class TradeRecord:
    """Record of a completed trade with all relevant metrics."""
    trade_id: int
    entry_timestamp: datetime
    exit_timestamp: datetime
    side: str  # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    contracts: int
    gross_pnl: float
    commission: float
    slippage_cost: float
    net_pnl: float
    r_multiple: float  # Risk-reward ratio
    setup: str  # Strategy name
    params_snapshot: dict
    contract: str


class Backtester:
    """
    Backtester that simulates trade execution from strategy signals.
    
    Takes signals from a strategy and simulates realistic trade execution including:
    - Entry at next bar open with slippage
    - Stop loss and profit target monitoring
    - Partial fills and position sizing
    - Commission and slippage costs
    """
    
    def __init__(self, strategy, bars_df: pd.DataFrame, config: Dict[str, Any], outdir: Path):
        """
        Initialize backtester.
        
        Args:
            strategy: Strategy object that implements generate_signals()
            bars_df: OHLCV price data with DatetimeIndex
            config: Configuration dictionary with contract metadata and settings
            outdir: Output directory for results files
        """
        self.strategy = strategy
        self.bars = bars_df
        self.config = config
        self.outdir = outdir
        self.trades: List[TradeRecord] = []
        self.trade_id_seq = 1
        
        # Extract key configuration values
        self.dollars_per_point = config["contract_meta"]["dollars_per_point"]
        self.tick_size = float(config["contract_meta"].get("tick_size", 0.0))
        # Use command line override first, then contract-specific, then global, then default
        if config.get("position_contracts_override") is not None:
            contracts_raw = config.get("position_contracts_override")
        else:
            contracts_raw = config.get("contract_meta", {}).get("position_contracts", 
                                       get_config_int(config, "contracts", default=1))
        self.contracts = max(1, int(contracts_raw))
        self.slippage_points = float(config.get("slippage_points", 0.0))
        self.contract_name = config.get("resolved_contract", "UNKNOWN")
        # Use contract-specific commission, fallback to global if not found
        self.commission_rt = float(config.get("contract_meta", {}).get("commission_roundtrip", 
                                             config.get("commission_roundtrip", 0.0)))
        
        # Rounding settings
        self.round_to_tick_enabled = config.get("orb", {}).get("round_stops_to_tick", False)

    def _prepare_strategy_data(self) -> pd.DataFrame:
        """Prepare data for strategy signal generation."""
        bars_for_strategy = self.bars.copy()
        bars_for_strategy.columns = [c.lower() for c in bars_for_strategy.columns]
        return bars_for_strategy

    def _normalize_signal(self, sig: pd.Series) -> Dict[str, Any]:
        """Convert signal Series to normalized dictionary with lowercase keys."""
        return {(k.lower() if isinstance(k, str) else k): v for k, v in sig.items()}

    def _apply_tick_rounding(self, price: float, direction: str = "nearest") -> float:
        """Apply tick rounding if enabled in config."""
        if self.round_to_tick_enabled and self.tick_size > 0:
            return round_to_tick(price, self.tick_size, direction)
        return price

    def _find_entry_bar(self, signal_time: pd.Timestamp) -> Optional[pd.Series]:
        """Find the next available bar for trade entry after signal time."""
        next_bars = self.bars.loc[self.bars.index > signal_time]
        return next_bars.iloc[0] if not next_bars.empty else None

    def _calculate_entry_price(self, entry_bar: pd.Series, side: str) -> float:
        """Calculate entry price with slippage."""
        next_open = float(entry_bar.get("open", entry_bar.get("Open")))
        sign = 1 if side == "LONG" else -1
        entry_fill = next_open + sign * self.slippage_points
        return self._apply_tick_rounding(entry_fill)

    def _extract_signal_prices(self, sig: Dict[str, Any]) -> Tuple[float, float, float]:
        """Extract and round stop and target prices from signal."""
        stop_price = float(sig["stop_price"])
        
        # Handle both single target_price and dual target1/target2 formats
        if "target1" in sig and "target2" in sig:
            # Dual target format (ORB engine)
            t1 = float(sig["target1"])
            t2 = float(sig["target2"])
        elif "target_price" in sig:
            # Single target format (scalping, vwap, pullback engines)
            target_price = float(sig["target_price"])
            t1 = target_price  # Use single target as first target
            t2 = target_price  # Use same target as second target (single exit at t1)
        else:
            raise ValueError("Signal must contain either 'target_price' or both 'target1' and 'target2'")
        
        if self.round_to_tick_enabled:
            stop_price = self._apply_tick_rounding(stop_price)
            t1 = self._apply_tick_rounding(t1)
            t2 = self._apply_tick_rounding(t2)
            
        return stop_price, t1, t2

    def _create_trade_record(self, trade_id: int, entry_timestamp: datetime, exit_timestamp: datetime,
                           side: str, entry_price: float, exit_price: float, contracts: int,
                           gross_pnl: float, commission: float, slippage_cost: float,
                           stop_price: float, setup: str = "UNKNOWN") -> TradeRecord:
        """Create a TradeRecord with calculated metrics."""
        # Validate contracts > 0
        if contracts <= 0:
            raise ValueError(f"Invalid contracts value: {contracts}. Must be > 0.")
        
        net_pnl = gross_pnl - commission - slippage_cost
        
        # Calculate R-multiple (risk-reward ratio)
        risk_amount = abs((entry_price - stop_price) * self.dollars_per_point)
        r_multiple = net_pnl / max(1.0, risk_amount)
        
        return TradeRecord(
            trade_id=trade_id,
            entry_timestamp=entry_timestamp,
            exit_timestamp=exit_timestamp,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            contracts=contracts,
            gross_pnl=gross_pnl,
            commission=commission,
            slippage_cost=slippage_cost,
            net_pnl=net_pnl,
            r_multiple=r_multiple,
            setup=setup,
            params_snapshot=self.config,
            contract=self.contract_name
        )

    def _execute_long_trade_exit(self, high: float, low: float, entry_price: float, 
                               stop_price: float, t1: float, t2: float, remaining_contracts: int,
                               first_exit_done: bool, entry_timestamp: datetime, 
                               exit_timestamp: datetime) -> Tuple[List[TradeRecord], int, bool, bool]:
        """Execute long trade exit logic (stop loss or profit targets)."""
        trades = []
        exited = False
        
        # Check stop loss first (adverse move)
        if low <= stop_price:
            exit_price = self._apply_tick_rounding(stop_price - self.slippage_points)
            gross_pnl = (exit_price - entry_price) * remaining_contracts * self.dollars_per_point
            slippage_cost = self.slippage_points * self.dollars_per_point * remaining_contracts
            
            trade = self._create_trade_record(
                self.trade_id_seq, entry_timestamp, exit_timestamp, "LONG",
                entry_price, exit_price, remaining_contracts, gross_pnl,
                self.commission_rt, slippage_cost, stop_price, "ORB"
            )
            trades.append(trade)
            self.trade_id_seq += 1
            exited = True
            
        # Check first target
        elif not first_exit_done and high >= t1:
            # For single target strategies (t1 == t2), exit entire position
            # For dual target strategies, exit partial position
            if t1 == t2:
                exit_contracts = remaining_contracts  # Exit entire position for single target
            elif remaining_contracts >= 2:
                exit_contracts = max(1, math.floor(remaining_contracts / 2))
            else:
                exit_contracts = remaining_contracts  # Exit all remaining if < 2 contracts
            exit_price = self._apply_tick_rounding(t1 - self.slippage_points)
            gross_pnl = (exit_price - entry_price) * exit_contracts * self.dollars_per_point
            slippage_cost = self.slippage_points * self.dollars_per_point * exit_contracts
            
            trade = self._create_trade_record(
                self.trade_id_seq, entry_timestamp, exit_timestamp, "LONG",
                entry_price, exit_price, exit_contracts, gross_pnl,
                self.commission_rt, slippage_cost, stop_price, "ORB"  # Commission on all exits
            )
            trades.append(trade)
            self.trade_id_seq += 1
            remaining_contracts = max(0, remaining_contracts - exit_contracts)
            first_exit_done = True
            
            # If we exited entire position (single target strategy), mark as exited
            if remaining_contracts == 0:
                exited = True
            
        # Check second target (only after first target hit)
        elif first_exit_done and high >= t2:
            exit_price = self._apply_tick_rounding(t2 - self.slippage_points)
            gross_pnl = (exit_price - entry_price) * remaining_contracts * self.dollars_per_point
            slippage_cost = self.slippage_points * self.dollars_per_point * remaining_contracts
            
            trade = self._create_trade_record(
                self.trade_id_seq, entry_timestamp, exit_timestamp, "LONG",
                entry_price, exit_price, remaining_contracts, gross_pnl,
                self.commission_rt, slippage_cost, stop_price, "ORB"
            )
            trades.append(trade)
            self.trade_id_seq += 1
            exited = True
            
        return trades, remaining_contracts, first_exit_done, exited

    def _execute_short_trade_exit(self, high: float, low: float, entry_price: float,
                                stop_price: float, t1: float, t2: float, remaining_contracts: int,
                                first_exit_done: bool, entry_timestamp: datetime,
                                exit_timestamp: datetime) -> Tuple[List[TradeRecord], int, bool, bool]:
        """Execute short trade exit logic (stop loss or profit targets)."""
        trades = []
        exited = False
        
        # Check stop loss first (adverse move)
        if high >= stop_price:
            exit_price = self._apply_tick_rounding(stop_price + self.slippage_points)
            gross_pnl = (entry_price - exit_price) * remaining_contracts * self.dollars_per_point
            slippage_cost = self.slippage_points * self.dollars_per_point * remaining_contracts
            
            trade = self._create_trade_record(
                self.trade_id_seq, entry_timestamp, exit_timestamp, "SHORT",
                entry_price, exit_price, remaining_contracts, gross_pnl,
                self.commission_rt, slippage_cost, stop_price, "ORB"
            )
            trades.append(trade)
            self.trade_id_seq += 1
            exited = True
            
        # Check first target
        elif not first_exit_done and low <= t1:
            # For single target strategies (t1 == t2), exit entire position
            # For dual target strategies, exit partial position
            if t1 == t2:
                exit_contracts = remaining_contracts  # Exit entire position for single target
            elif remaining_contracts >= 2:
                exit_contracts = max(1, math.floor(remaining_contracts / 2))
            else:
                exit_contracts = remaining_contracts  # Exit all remaining if < 2 contracts
            exit_price = self._apply_tick_rounding(t1 + self.slippage_points)
            gross_pnl = (entry_price - exit_price) * exit_contracts * self.dollars_per_point
            slippage_cost = self.slippage_points * self.dollars_per_point * exit_contracts
            
            trade = self._create_trade_record(
                self.trade_id_seq, entry_timestamp, exit_timestamp, "SHORT",
                entry_price, exit_price, exit_contracts, gross_pnl,
                self.commission_rt, slippage_cost, stop_price, "ORB"  # Commission on all exits
            )
            trades.append(trade)
            self.trade_id_seq += 1
            remaining_contracts = max(0, remaining_contracts - exit_contracts)
            first_exit_done = True
            
            # If we exited entire position (single target strategy), mark as exited
            if remaining_contracts == 0:
                exited = True
            
        # Check second target (only after first target hit)
        elif first_exit_done and low <= t2:
            exit_price = self._apply_tick_rounding(t2 + self.slippage_points)
            gross_pnl = (entry_price - exit_price) * remaining_contracts * self.dollars_per_point
            slippage_cost = self.slippage_points * self.dollars_per_point * remaining_contracts
            
            trade = self._create_trade_record(
                self.trade_id_seq, entry_timestamp, exit_timestamp, "SHORT",
                entry_price, exit_price, remaining_contracts, gross_pnl,
                self.commission_rt, slippage_cost, stop_price, "ORB"
            )
            trades.append(trade)
            self.trade_id_seq += 1
            exited = True
            
        return trades, remaining_contracts, first_exit_done, exited

    def _execute_market_close_exit(self, last_bar: pd.Series, side: str, entry_price: float,
                                 remaining_contracts: int, entry_timestamp: datetime,
                                 stop_price: float) -> TradeRecord:
        """Execute market-on-close exit for remaining position."""
        exit_price = float(last_bar.get("close", last_bar.get("Close")))
        sign = 1 if side == "LONG" else -1
        exit_price = self._apply_tick_rounding(exit_price - sign * self.slippage_points)
        
        if side == "LONG":
            gross_pnl = (exit_price - entry_price) * remaining_contracts * self.dollars_per_point
        else:
            gross_pnl = (entry_price - exit_price) * remaining_contracts * self.dollars_per_point
            
        slippage_cost = self.slippage_points * self.dollars_per_point * remaining_contracts
        
        return self._create_trade_record(
            self.trade_id_seq, entry_timestamp, last_bar.name.to_pydatetime(), side,
            entry_price, exit_price, remaining_contracts, gross_pnl,
            self.commission_rt, slippage_cost, stop_price, "ORB"
        )

    def _process_signal(self, idx: int, sig: pd.Series) -> None:
        """Process a single trading signal and execute trades."""
        # Normalize signal to dictionary
        sig_dict = self._normalize_signal(sig)
        
        # Extract signal data
        signal_time = pd.to_datetime(sig_dict["timestamp"])
        side = sig_dict["side"]
        
        # Find entry bar
        entry_bar = self._find_entry_bar(signal_time)
        if entry_bar is None:
            return  # No next bar available
        
        # Calculate entry price
        entry_price = self._calculate_entry_price(entry_bar, side)
        entry_timestamp = entry_bar.name.to_pydatetime()
        
        # Extract stop and target prices
        stop_price, t1, t2 = self._extract_signal_prices(sig_dict)
        
        # Initialize position tracking
        remaining_contracts = self.contracts
        if remaining_contracts <= 0:
            print(f"Warning: Invalid contract size {remaining_contracts} for signal, skipping")
            return
        first_exit_done = False
        
        # Process subsequent bars for exits
        next_bars = self.bars.loc[self.bars.index > signal_time]
        exited = False
        
        for bar_time, bar in next_bars.iterrows():
            high = float(bar.get("high", bar.get("High")))
            low = float(bar.get("low", bar.get("Low")))
            exit_timestamp = bar_time.to_pydatetime()
            
            # Execute exit logic based on side
            if side == "LONG":
                new_trades, remaining_contracts, first_exit_done, exited = self._execute_long_trade_exit(
                    high, low, entry_price, stop_price, t1, t2, remaining_contracts,
                    first_exit_done, entry_timestamp, exit_timestamp
                )
            else:  # SHORT
                new_trades, remaining_contracts, first_exit_done, exited = self._execute_short_trade_exit(
                    high, low, entry_price, stop_price, t1, t2, remaining_contracts,
                    first_exit_done, entry_timestamp, exit_timestamp
                )
            
            # Add any new trades
            self.trades.extend(new_trades)
            
            if exited:
                break
        
        # Handle market close exit if position still open
        if not exited and remaining_contracts > 0:
            last_bar = next_bars.iloc[-1] if not next_bars.empty else entry_bar
            market_close_trade = self._execute_market_close_exit(
                last_bar, side, entry_price, remaining_contracts, entry_timestamp, stop_price
            )
            self.trades.append(market_close_trade)
            self.trade_id_seq += 1

    def run(self) -> None:
        """
        Execute backtesting simulation.
        
        Generates signals from strategy and simulates trade execution including:
        - Entry at next bar open with slippage
        - Stop loss and profit target monitoring  
        - Partial position exits
        - Commission and slippage costs
        """
        # Prepare data for strategy
        bars_for_strategy = self._prepare_strategy_data()
        
        # Generate signals
        signals = self.strategy.generate_signals(bars_for_strategy)
        if signals is None or signals.empty:
            return
        
        # Process each signal
        for idx, sig in signals.iterrows():
            try:
                self._process_signal(idx, sig)
            except Exception as e:
                # Log error but continue processing other signals
                print(f"Error processing signal {idx}: {e}")
                continue

    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute summary statistics from completed trades.
        
        Returns:
            Dictionary with performance metrics including total trades,
            P&L figures, win rate, and other key statistics
        """
        total = len(self.trades)
        if total == 0:
            return {
                "total_trades": 0,
                "gross_pnl": 0.0,
                "net_pnl": 0.0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
            }
        
        gross_pnl = sum(t.gross_pnl for t in self.trades)
        net_pnl = sum(t.net_pnl for t in self.trades)
        wins = sum(1 for t in self.trades if t.net_pnl > 0)
        losses = sum(1 for t in self.trades if t.net_pnl <= 0)
        win_rate = wins / total if total > 0 else 0.0
        
        # Extract strategy parameters from config for run identification
        strategy_name = self.strategy.__class__.__name__
        strategy_params = {}
        
        # Get strategy-specific parameters
        if hasattr(self.strategy, 'params') and self.strategy.params:
            strategy_params = dict(self.strategy.params)
        
        # Extract key run parameters
        run_params = {
            "strategy": strategy_name,
            "contract": self.contract_name,
            "contracts": self.contracts,
            "dollars_per_point": self.dollars_per_point,
            "commission_roundtrip": self.commission_rt,
            "slippage_points": self.slippage_points,
            "session_start": self.config.get("session_start"),
            "session_end": self.config.get("session_end"),
            "resample_minutes": self.config.get("resample_minutes"),
        }
        
        return {
            "run_parameters": run_params,
            "strategy_parameters": strategy_params,
            "performance": {
                "total_trades": total,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "wins": wins,
                "losses": losses,
                "win_rate": round(win_rate, 4),
            }
        }

    def save_results(self) -> None:
        """Save backtest results to output files."""
        self.outdir.mkdir(parents=True, exist_ok=True)
        
        # Save trades CSV
        trades_file = self.outdir / "trades.csv"
        if not self.trades:
            # Write empty CSV with headers
            with open(trades_file, "w") as f:
                f.write(",".join([
                    "trade_id", "entry_timestamp", "exit_timestamp", "side", "entry_price", "exit_price",
                    "contracts", "gross_pnl", "commission", "slippage_cost", "net_pnl", "r_multiple",
                    "setup", "params_snapshot", "contract"
                ]) + "\n")
        else:
            # Write trades to CSV
            import csv
            with open(trades_file, "w", newline="") as f:
                writer = csv.writer(f)
                
                # Write header
                writer.writerow([
                    "trade_id", "entry_timestamp", "exit_timestamp", "side", "entry_price", "exit_price",
                    "contracts", "gross_pnl", "commission", "slippage_cost", "net_pnl", "r_multiple",
                    "setup", "params_snapshot", "contract"
                ])
                
                # Write trade data
                for trade in self.trades:
                    writer.writerow([
                        trade.trade_id,
                        trade.entry_timestamp.isoformat(),
                        trade.exit_timestamp.isoformat(),
                        trade.side,
                        trade.entry_price,
                        trade.exit_price,
                        trade.contracts,
                        trade.gross_pnl,
                        trade.commission,
                        trade.slippage_cost,
                        trade.net_pnl,
                        trade.r_multiple,
                        trade.setup,
                        json.dumps(trade.params_snapshot),
                        trade.contract
                    ])
        
        # Save summary JSON
        summary_file = self.outdir / "summary.json"
        summary = self.compute_metrics()
        with open(summary_file, "w") as f:
            json.dump(summary, f, default=str, indent=2)