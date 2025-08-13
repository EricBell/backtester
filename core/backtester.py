"""
core/backtester.py - skeleton backtester that integrates Strategy objects and simulates execution.
"""

from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

@dataclass
class TradeRecord:
    # simplified trade record dataclass for illustration
    trade_id: int
    entry_timestamp: datetime
    exit_timestamp: datetime
    side: str
    entry_price: float
    exit_price: float
    contracts: int
    gross_pnl: float
    commission: float
    slippage_cost: float
    net_pnl: float
    r_multiple: float
    setup: str
    params_snapshot: dict
    contract: str

class Backtester:
    def __init__(self, strategy, bars_df: pd.DataFrame, config: Dict[str, Any], outdir: Path):
        self.strategy = strategy
        self.bars = bars_df
        self.config = config
        self.outdir = outdir
        self.trades = []  # list of TradeRecord
        self.trade_id_seq = 1
        # dollars_per_point resolved from contract meta
        self.dollars_per_point = config["contract_meta"]["dollars_per_point"]

    def run(self):
        """
        Skeleton:
        - Ask strategy to generate signals (DataFrame of signals)
        - For each signal simulate a fill according to rules (slippage, commission)
        - Record trades
        """
        signals = self.strategy.generate_signals(self.bars)
        # For skeleton, we won't implement fills; we'll create no trades.
        # Real implementation: iterate signals and simulate fills.
        return

    def compute_metrics(self):
        # Compute summary statistics from self.trades
        # Return a dict that will be saved as summary.json
        return {
            "total_trades": len(self.trades),
            "wins": sum(1 for t in self.trades if t.net_pnl > 0),
            "losses": sum(1 for t in self.trades if t.net_pnl <= 0),
        }

    def save_results(self):
        # Save trades.csv (empty or stub), summary.json, and a placeholder equity_curve.png (not generated in skeleton)
        trades_out = self.outdir / "trades.csv"
        summary_out = self.outdir / "summary.json"
        daily_out = self.outdir / "daily_pnl.csv"

        # write empty trades CSV header if no trades
        if not self.trades:
            with open(trades_out, "w") as f:
                f.write(",".join([
                    "trade_id","entry_timestamp","exit_timestamp","side","entry_price","exit_price",
                    "contracts","gross_pnl","commission","slippage_cost","net_pnl","r_multiple","setup","params_snapshot","contract"
                ]) + "\n")
        else:
            # Real implementation would serialize dataclass list to CSV
            pass

        summary = self.compute_metrics()
        with open(summary_out, "w") as f:
            json.dump(summary, f, default=str, indent=2)