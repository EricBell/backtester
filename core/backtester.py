"""
core/backtester.py - skeleton backtester that integrates Strategy objects and simulates execution.
"""


from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd
from pathlib import Path
import json
from datetime import datetime


# def get_config_int(cfg, key, default=1, preferred_keys=("value", "val", "contracts", "qty", "quantity", "n", "count")):
#     """
#     Safely get an integer from cfg[key]. If cfg[key] is a dict, try common internal keys
#     or fall back to the first value. Returns int(default) on failure.
#     """
#     raw = cfg.get(key, default)
#     if isinstance(raw, dict):
#         for k in preferred_keys:
#             if k in raw:
#                 raw = raw[k]
#                 break
#         else:
#             try:
#                 raw = next(iter(raw.values()))
#             except Exception:
#                 raw = default
#     try:
#         return int(raw)
#     except (TypeError, ValueError):
#         return int(default)

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

   


    # --- add near top of file (after imports) ---
    def get_config_int(cfg, key, default=1, preferred_keys=("value", "val", "contracts", "qty", "quantity", "n", "count")):
        """
        Safely get an integer from cfg[key]. If cfg[key] is a dict, try common internal keys
        or fall back to the first value. Returns int(default) on failure.
        """
        raw = cfg.get(key, default)
        # If the config value is a dict, try preferred keys then fall back to first value
        if isinstance(raw, dict):
            for k in preferred_keys:
                if k in raw:
                    raw = raw[k]
                    break
            else:
                # fallback: first value or default
                try:
                    raw = next(iter(raw.values()))
                except Exception:
                    raw = default
        # Convert to int, with safe fallback
        try:
            return int(raw)
        except (TypeError, ValueError):
            return int(default)

 



    def run(self):
        """
        - Ask strategy to generate signals
        - For each signal attempt to execute (entry at next bar open) and monitor for stop/targets
        - Record filled trades into self.trades
        """
        # Give the strategy a lowercase-column copy of the bars so it can find expected columns.
        # This preserves self.bars (Title-case) for the execution simulator below while
        # providing the strategy the lowercase columns it expects (open/high/low/close/volume).
        bars_for_strategy = self.bars.copy()
        bars_for_strategy.columns = [c.lower() for c in bars_for_strategy.columns]
        signals = self.strategy.generate_signals(bars_for_strategy)
        if signals is None or signals.empty:
            return


        import logging, pprint
        logging.basicConfig(level=logging.DEBUG)
        raw_contracts = self.config.get("contracts", 1)




        logging.debug("backtester: raw_contracts repr=%r type=%s", raw_contracts, type(raw_contracts))
        pprint.pprint(raw_contracts)


        # default contracts
            # contracts = int(self.config.get("contracts", 1))
        # --- replace the original contracts assignment (around line ~50) with this ---
        contracts = get_config_int(self.config, "contracts", default=1)








        slippage_points = float(self.config.get("slippage_points", 0.0))
        commission_rt = float(self.config.get("commission_roundtrip", 0.0))
        tick_size = float(self.config["contract_meta"].get("tick_size", 0.0))
        dollars_per_point = float(self.dollars_per_point)

        for idx, sig in signals.iterrows():

            # sig.index = [k.lower() if isinstance(k, str) else k for k in sig.index]
            # normalize the signal row to a plain dict with lowercase string keys
            sig = { (k.lower() if isinstance(k, str) else k): v for k, v in sig.items() }
            # DEBUG: inspect signal row to find missing keys (insert immediately after "for idx, sig in signals.iterrows():")
            import pprint
            print(f"[DEBUG_SIGNAL {idx}] -- START SIGNAL DEBUG --")
            print(f"[DEBUG] signals.columns = {list(signals.columns)}")
            try:
                keys = list(sig.index)
            except Exception:
                keys = None
            print(f"[DEBUG_SIGNAL {idx}] sig.index keys = {keys}")
            try:
                sig_dict = sig.to_dict()
                print(f"[DEBUG_SIGNAL {idx}] sig as dict:")
                pprint.pprint(sig_dict)
            except Exception as e:
                print(f"[DEBUG_SIGNAL {idx}] sig.to_dict() failed: {e}")
            print(f"[DEBUG_SIGNAL {idx}] -- END SIGNAL DEBUG --")


            ts = pd.to_datetime(sig["timestamp"])
            side = sig["side"]
            signal_time = ts


            # --- DEBUG: per-signal diagnostics (insert here) ---
            try:
                # parse planned prices safely
                sig_price = sig.get("signal_price")
                stop_price = float(sig["stop_price"]) if "stop_price" in sig and pd.notna(sig["stop_price"]) else None
                t1 = float(sig["target1"]) if "target1" in sig and pd.notna(sig["target1"]) else None
                t2 = float(sig["target2"]) if "target2" in sig and pd.notna(sig["target2"]) else None
            except Exception:
                sig_price = sig.get("signal_price")
                stop_price = t1 = t2 = None

            # print key runtime settings and signal values
            # --- inside the signals loop, replace the debug print that called int(...) with this safe print ---
            raw_contracts = self.config.get("contracts", 1)
            print(f"[DEBUG_SIGNAL {idx}] ts={signal_time} side={side} sig_price={sig_price} stop={stop_price} t1={t1} t2={t2}")
            print(f"[DEBUG_SIGNAL {idx}] config_contracts_raw={repr(raw_contracts)} type={type(raw_contracts)} slippage_points={slippage_points} commission_rt={commission_rt} tick_size={tick_size} dollars_per_point={dollars_per_point}")            
            # print(f"[DEBUG_SIGNAL {idx}] ts={signal_time} side={side} sig_price={sig_price} stop={stop_price} t1={t1} t2={t2}")
            # print(f"[DEBUG_SIGNAL {idx}] config_contracts={int(self.config.get('contracts',1))} slippage_points={slippage_points} commission_rt={commission_rt} tick_size={tick_size} dollars_per_point={dollars_per_point}")

            # locate the next bar (same logic Backtester uses) but print context before deciding
            next_bars = self.bars.loc[self.bars.index > signal_time]
            if next_bars.empty:
                before = self.bars[self.bars.index <= signal_time].tail(3)
                after = self.bars[self.bars.index > signal_time].head(3)
                print(f"[DEBUG_SIGNAL {idx}] NO next bar found after {signal_time}")
                print("[DEBUG_SIGNAL {idx}] nearest before (<=ts):")
                if not before.empty:
                    print(before.to_string())
                else:
                    print("  <none>")
                print("[DEBUG_SIGNAL {idx}] nearest after (>ts):")
                if not after.empty:
                    print(after.to_string())
                else:
                    print("  <none>")
                # keep behavior: skip this signal
                continue

            # If next_bar exists, print the entry bar open we will use
            entry_bar = next_bars.iloc[0]
            try:
                next_open = float(entry_bar["open"])
            except Exception:
                next_open = None
            print(f"[DEBUG_SIGNAL {idx}] next_bar={entry_bar.name} next_open={next_open}")
            # --- end DEBUG block ---


            # find next bar after signal_time in bars index
            next_bars = self.bars.loc[self.bars.index > signal_time]
            if next_bars.empty:
                continue
            entry_bar = next_bars.iloc[0]
            # next_open = float(entry_bar["open"])
            next_open = float(entry_bar.get("open", entry_bar.get("Open")))

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

    def compute_metrics(self):
        """
        Compute summary statistics from self.trades.
        Returns a dict with at least:
          total_trades, gross_pnl, net_pnl, wins, losses, win_rate
        """
        total = len(self.trades)
        gross = sum(t.gross_pnl for t in self.trades) if total else 0.0
        net = sum(t.net_pnl for t in self.trades) if total else 0.0
        wins = sum(1 for t in self.trades if t.net_pnl > 0)
        losses = sum(1 for t in self.trades if t.net_pnl <= 0)
        win_rate = (wins / total) if total else 0.0

        # preserve existing fields where applicable; this dict can be expanded later
        return {
            "total_trades": total,
            "gross_pnl": gross,
            "net_pnl": net,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 4),
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
