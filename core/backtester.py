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
    # __init__ unchanged

    def run(self):
        """
        - Ask strategy to generate signals
        - For each signal attempt to execute (entry at next bar open) and monitor for stop/targets
        - Record filled trades into self.trades
        """
        signals = self.strategy.generate_signals(self.bars)
        if signals is None or signals.empty:
            return

        # default contracts
        contracts = int(self.config.get("contracts", 1))
        slippage_points = float(self.config.get("slippage_points", 0.0))
        commission_rt = float(self.config.get("commission_roundtrip", 0.0))
        tick_size = float(self.config["contract_meta"].get("tick_size", 0.0))
        dollars_per_point = float(self.dollars_per_point)

        for idx, sig in signals.iterrows():
            ts = pd.to_datetime(sig["timestamp"])
            side = sig["side"]
            signal_time = ts
            # find next bar after signal_time in bars index
            next_bars = self.bars.loc[self.bars.index > signal_time]
            if next_bars.empty:
                continue
            entry_bar = next_bars.iloc[0]
            next_open = float(entry_bar["open"])

            sign = 1 if side == "LONG" else -1
            entry_fill = next_open + sign * slippage_points  # slippage in points
            # optional rounding
            if self.config.get("orb", {}).get("round_stops_to_tick", False):
                # ensure entry rounded to tick for consistency
                entry_fill = round_to_tick(entry_fill, tick_size, direction="nearest")

            # determine stop/targets
            stop_price = float(sig["stop_price"])
            t1 = float(sig["target1"])
            t2 = float(sig["target2"])
            # apply rounding if requested
            if self.config.get("orb", {}).get("round_stops_to_tick", False):
                stop_price = round_to_tick(stop_price, tick_size, direction="nearest")
                t1 = round_to_tick(t1, tick_size, direction="nearest")
                t2 = round_to_tick(t2, tick_size, direction="nearest")

            # now scan subsequent bars to detect hits (intrabar using high/low)
            remaining_contracts = contracts
            open_contracts = contracts
            entry_timestamp = entry_bar.name.to_pydatetime()
            entry_price = entry_fill

            # current position: we assume full position opened at entry_fill
            # We will allow 2-stage exits: first T1 (partial) then T2 for remainder
            first_exit_done = False

            # walk bars starting from entry_bar
            post_bars = next_bars  # includes entry_bar and onwards
            exited = False
            for j, prow in post_bars.iterrows():
                high = float(prow["high"])
                low = float(prow["low"])
                # check targets/stops depending on side
                if side == "LONG":
                    # check stop first (adverse)
                    if low <= stop_price:
                        exit_price = stop_price - slippage_points  # assume slippage on stop hit
                        exit_price = round_to_tick(exit_price, tick_size, direction="nearest") if tick_size > 0 else exit_price
                        gross = (exit_price - entry_price) * remaining_contracts * dollars_per_point
                        commission = commission_rt  # apply round trip at close; simple model
                        slippage_cost = slippage_points * dollars_per_point * remaining_contracts
                        net = gross - commission - slippage_cost
                        tr = TradeRecord(
                            trade_id=self.trade_id_seq,
                            entry_timestamp=entry_timestamp,
                            exit_timestamp=j.to_pydatetime(),
                            side=side,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            contracts=remaining_contracts,
                            gross_pnl=gross,
                            commission=commission,
                            slippage_cost=slippage_cost,
                            net_pnl=net,
                            r_multiple= net / max(1.0, abs((entry_price - stop_price) * dollars_per_point)),
                            setup="ORB",
                            params_snapshot=self.config,
                            contract=self.config.get("resolved_contract")
                        )
                        self.trades.append(tr)
                        self.trade_id_seq += 1
                        exited = True
                        break
                    # target1
                    if (not first_exit_done) and (high >= t1):
                        # partial exit of half (if >1 contracts) else full
                        exit_contracts = max(1, math.floor(remaining_contracts / 2))
                        exit_price = t1 - slippage_points  # assume slippage
                        exit_price = round_to_tick(exit_price, tick_size, direction="nearest") if tick_size > 0 else exit_price
                        gross = (exit_price - entry_price) * exit_contracts * dollars_per_point
                        commission = 0.0  # round-trip applied at final close; to keep simple we will apply commission_rt when the whole trade closes
                        slippage_cost = slippage_points * dollars_per_point * exit_contracts
                        net = gross - (commission) - slippage_cost
                        tr = TradeRecord(
                            trade_id=self.trade_id_seq,
                            entry_timestamp=entry_timestamp,
                            exit_timestamp=j.to_pydatetime(),
                            side=side,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            contracts=exit_contracts,
                            gross_pnl=gross,
                            commission=commission,
                            slippage_cost=slippage_cost,
                            net_pnl=net,
                            r_multiple= net / max(1.0, abs((entry_price - stop_price) * dollars_per_point)),
                            setup="ORB",
                            params_snapshot=self.config,
                            contract=self.config.get("resolved_contract")
                        )
                        self.trades.append(tr)
                        self.trade_id_seq += 1
                        remaining_contracts -= exit_contracts
                        first_exit_done = True
                        # continue looking for second target/stop for remaining contracts
                        # Note: commission for these partial exits will be aggregated at final close below for simplicity

                    # target2 for remainder
                    if first_exit_done and (high >= t2):
                        exit_price = t2 - slippage_points
                        exit_price = round_to_tick(exit_price, tick_size, direction="nearest") if tick_size > 0 else exit_price
                        gross = (exit_price - entry_price) * remaining_contracts * dollars_per_point
                        commission = commission_rt
                        slippage_cost = slippage_points * dollars_per_point * remaining_contracts
                        net = gross - commission - slippage_cost
                        tr = TradeRecord(
                            trade_id=self.trade_id_seq,
                            entry_timestamp=entry_timestamp,
                            exit_timestamp=j.to_pydatetime(),
                            side=side,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            contracts=remaining_contracts,
                            gross_pnl=gross,
                            commission=commission,
                            slippage_cost=slippage_cost,
                            net_pnl=net,
                            r_multiple= net / max(1.0, abs((entry_price - stop_price) * dollars_per_point)),
                            setup="ORB",
                            params_snapshot=self.config,
                            contract=self.config.get("resolved_contract")
                        )
                        self.trades.append(tr)
                        self.trade_id_seq += 1
                        exited = True
                        break

                else:  # SHORT
                    if high >= stop_price:
                        exit_price = stop_price + slippage_points
                        exit_price = round_to_tick(exit_price, tick_size, direction="nearest") if tick_size > 0 else exit_price
                        gross = (entry_price - exit_price) * remaining_contracts * dollars_per_point
                        commission = commission_rt
                        slippage_cost = slippage_points * dollars_per_point * remaining_contracts
                        net = gross - commission - slippage_cost
                        tr = TradeRecord(
                            trade_id=self.trade_id_seq,
                            entry_timestamp=entry_timestamp,
                            exit_timestamp=j.to_pydatetime(),
                            side=side,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            contracts=remaining_contracts,
                            gross_pnl=gross,
                            commission=commission,
                            slippage_cost=slippage_cost,
                            net_pnl=net,
                            r_multiple= net / max(1.0, abs((entry_price - stop_price) * dollars_per_point)),
                            setup="ORB",
                            params_snapshot=self.config,
                            contract=self.config.get("resolved_contract")
                        )
                        self.trades.append(tr)
                        self.trade_id_seq += 1
                        exited = True
                        break
                    if (not first_exit_done) and (low <= t1):
                        exit_contracts = max(1, math.floor(remaining_contracts / 2))
                        exit_price = t1 + slippage_points
                        exit_price = round_to_tick(exit_price, tick_size, direction="nearest") if tick_size > 0 else exit_price
                        gross = (entry_price - exit_price) * exit_contracts * dollars_per_point
                        commission = 0.0
                        slippage_cost = slippage_points * dollars_per_point * exit_contracts
                        net = gross - commission - slippage_cost
                        tr = TradeRecord(
                            trade_id=self.trade_id_seq,
                            entry_timestamp=entry_timestamp,
                            exit_timestamp=j.to_pydatetime(),
                            side=side,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            contracts=exit_contracts,
                            gross_pnl=gross,
                            commission=commission,
                            slippage_cost=slippage_cost,
                            net_pnl=net,
                            r_multiple= net / max(1.0, abs((entry_price - stop_price) * dollars_per_point)),
                            setup="ORB",
                            params_snapshot=self.config,
                            contract=self.config.get("resolved_contract")
                        )
                        self.trades.append(tr)
                        self.trade_id_seq += 1
                        remaining_contracts -= exit_contracts
                        first_exit_done = True
                    if first_exit_done and (low <= t2):
                        exit_price = t2 + slippage_points
                        exit_price = round_to_tick(exit_price, tick_size, direction="nearest") if tick_size > 0 else exit_price
                        gross = (entry_price - exit_price) * remaining_contracts * dollars_per_point
                        commission = commission_rt
                        slippage_cost = slippage_points * dollars_per_point * remaining_contracts
                        net = gross - commission - slippage_cost
                        tr = TradeRecord(
                            trade_id=self.trade_id_seq,
                            entry_timestamp=entry_timestamp,
                            exit_timestamp=j.to_pydatetime(),
                            side=side,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            contracts=remaining_contracts,
                            gross_pnl=gross,
                            commission=commission,
                            slippage_cost=slippage_cost,
                            net_pnl=net,
                            r_multiple= net / max(1.0, abs((entry_price - stop_price) * dollars_per_point)),
                            setup="ORB",
                            params_snapshot=self.config,
                            contract=self.config.get("resolved_contract")
                        )
                        self.trades.append(tr)
                        self.trade_id_seq += 1
                        exited = True
                        break

            # if not exited by targets or stop and bars exhausted, close at last bar close (market-on-close)
            if not exited:
                last_bar = post_bars.iloc[-1]
                exit_price = float(last_bar["close"])
                # apply slippage
                exit_price = exit_price - sign * slippage_points
                exit_price = round_to_tick(exit_price, tick_size, direction="nearest") if tick_size > 0 else exit_price
                gross = ((exit_price - entry_price) if side == "LONG" else (entry_price - exit_price)) * remaining_contracts * dollars_per_point
                commission = commission_rt
                slippage_cost = slippage_points * dollars_per_point * remaining_contracts
                net = gross - commission - slippage_cost
                tr = TradeRecord(
                    trade_id=self.trade_id_seq,
                    entry_timestamp=entry_timestamp,
                    exit_timestamp=last_bar.name.to_pydatetime(),
                    side=side,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    contracts=remaining_contracts,
                    gross_pnl=gross,
                    commission=commission,
                    slippage_cost=slippage_cost,
                    net_pnl=net,
                    r_multiple= net / max(1.0, abs((entry_price - stop_price) * dollars_per_point)),
                    setup="ORB",
                    params_snapshot=self.config,
                    contract=self.config.get("resolved_contract")
                )
                self.trades.append(tr)
                self.trade_id_seq += 1

        # run complete

    # keep existing compute_metrics and save_results implementations (unchanged)
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