# engines/orb_engine.py
import pandas as pd
import numpy as np
from core import indicators
from datetime import timedelta, time

# at top of engines/orb_engine.py (or near other imports)
import os
from pathlib import Path
from engines.orb_debug import OrbDebug

# enable via env var DEBUG_ORB=1 and optional ORB_DEBUG_OUTDIR path
_debug_enabled = os.getenv("DEBUG_ORB", "0") not in ("0", "", "false", "False")
_debug_outdir = Path(os.getenv("ORB_DEBUG_OUTDIR", "outputs/orb_debug"))
orb_debugger = OrbDebug(enabled=_debug_enabled, outdir=_debug_outdir)
print("ORB_DEBUG enabled =", _debug_enabled, "outdir =", _debug_outdir)
# right after orb_debugger = OrbDebug(...)
# orb_debugger.log({"timestamp":"init-test", "note":"init-called"})
# orb_debugger.flush()
# print("ORB_DEBUG test log written (init).")

# debug probe — add immediately after orb_debugger = OrbDebug(...)
import traceback, sys
try:
    print("DEBUG: orb_debugger type:", type(orb_debugger))
    print("DEBUG: orb_debugger public attrs:", [n for n in dir(orb_debugger) if not n.startswith('_')])
    # orb_debugger.log({"timestamp":"init-test", "note":"init-called"})
    # orb_debugger.flush()
    # print("ORB_DEBUG init test succeeded - file should be written to outdir")
except Exception as exc:
    print("ORB_DEBUG init test FAILED:", repr(exc))
    traceback.print_exc(file=sys.stdout)





class ORBStrategy:
    def __init__(self, params: dict):
        self.params = params or {}
        self.open_range_minutes = int(self.params.get("open_range_minutes", 15))
        self.require_volume_confirmation = bool(self.params.get("require_volume_confirmation", True))
        self.volume_lookback = int(self.params.get("volume_lookback", 5))
        self.take_profit_r = list(self.params.get("take_profit_r", [1.0, 2.0]))
        self.stop_policy = self.params.get("stop_policy", "structure")
        self.atr_multiplier_stop = float(self.params.get("atr_multiplier_stop", 1.0))
        self.stop_fixed_points = float(self.params.get("stop_fixed_points", 4.0))

        # misc
        self.cancel_within_minutes = int(self.params.get("cancel_within_minutes", 60))

    def _day_groups(self, df: pd.DataFrame, tz_target: str = "America/New_York"):
        """
        Yield (date, day_df) for each calendar date that has session bars.
        - Ensures df.index is a DatetimeIndex and converted/localized to tz_target.
        - Optionally filters to session_start/session_end if those attrs exist on self.
        """
        df = df.copy()

        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                raise ValueError(f"_day_groups: df.index is not datetime-like and could not be parsed: {e}")

        # Localize or convert to tz_target
        try:
            if df.index.tz is None:
                # Treat naive timestamps as tz_target (explicit). If your source is UTC, convert at load time instead.
                df.index = df.index.tz_localize(tz_target)
            else:
                df.index = df.index.tz_convert(tz_target)
        except Exception as e:
            # fallback attempt
            try:
                df.index = pd.to_datetime(df.index).tz_localize(tz_target)
            except Exception:
                raise ValueError(f"_day_groups: failed to localize/convert index to {tz_target}: {e}")

        # Session window strings, if configured on self (optional)
        session_start = getattr(self, "session_start", None)
        session_end = getattr(self, "session_end", None)

        # Group by calendar date (in tz_target)
        for date_val, group in df.groupby(df.index.date):
            # Optionally filter to session window (if configured)
            if session_start and session_end:
                try:
                    session_df = group.between_time(session_start, session_end)
                except Exception:
                    # fallback: manual time comparison
                    start_time = pd.to_datetime(session_start).time()
                    end_time = pd.to_datetime(session_end).time()
                    session_df = group[(group.index.time >= start_time) & (group.index.time <= end_time)]
            else:
                session_df = group

            if session_df.empty:
                # skip days that have no session bars
                continue

            # sort/clean and yield
            session_df = session_df.sort_index()
            yield date_val, session_df
            """
            Yield (date, day_df) for each calendar date that has session bars.
            - Ensures df.index is a DatetimeIndex and converted/localized to tz_target.
            - Optionally filters to session_start/session_end if those attrs exist on self.
            """
            df = df.copy()

            # Ensure DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception as e:
                    raise ValueError(f"_day_groups: df.index is not datetime-like and could not be parsed: {e}")

            # Localize or convert to tz_target
            try:
                if df.index.tz is None:
                    # Treat naive timestamps as tz_target (explicit). If your source is UTC, change at load time.
                    df.index = df.index.tz_localize(tz_target)
                else:
                    df.index = df.index.tz_convert(tz_target)
            except Exception as e:
                # defensively attempt a safe conversion/localization
                try:
                    df.index = pd.to_datetime(df.index).tz_localize(tz_target)
                except Exception:
                    raise ValueError(f"_day_groups: failed to localize/convert index to {tz_target}: {e}")

            # Session window strings, if configured on self (optional)
            session_start = getattr(self, "session_start", None)
            session_end = getattr(self, "session_end", None)

            # Group by calendar date (in tz_target)
            # Use df.index.date which yields date objects in local tz
            for date_val, group in df.groupby(df.index.date):
                # Optionally filter to session window (if configured)
                if session_start and session_end:
                    try:
                        session_df = group.between_time(session_start, session_end)
                    except Exception:
                        # fallback: manual time comparison
                        start_time = pd.to_datetime(session_start).time()
                        end_time = pd.to_datetime(session_end).time()
                        session_df = group[(group.index.time >= start_time) & (group.index.time <= end_time)]
                else:
                    session_df = group

                if session_df.empty:
                    # skip days that have no session bars
                    continue

                # sort/clean and yield
                session_df = session_df.sort_index()
                yield date_val, session_df


    def _append_signal_dedup(self, signals_list, seen_keys, sig):
        """
        Append sig (dict) to signals_list only if its key is not in seen_keys.
        Key is (timestamp_iso, side, float(signal_price)).
        Preserves first occurrence.
        """
        try:
            ts = sig.get("timestamp")
            # normalize timestamp to string key
            ts_key = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            side = sig.get("side")
            price = float(sig.get("signal_price"))
            key = (ts_key, side, price)
        except Exception:
            # if anything is malformed, still attempt to append (be conservative)
            key = None

        if key is not None:
            if key in seen_keys:
                return
            seen_keys.add(key)
        signals_list.append(sig)                

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return signals DataFrame with columns:
        timestamp, side ('LONG'/'SHORT'), signal_price (breakout close), entry_price_hint (None),
        stop_price, target1, target2, meta (dict)
        """

        print('generate_signals called with df rows:', len(df))

        signals = []
        seen_keys = set()

        # ensure we have columns open/high/low/close/volume
        if not {"open","high","low","close","volume"}.issubset(df.columns):
            return pd.DataFrame(columns=[
                "timestamp","side","signal_price","entry_price_hint","stop_price","target1","target2","meta"
            ])

        # compute EMA(9) for confirmation across entire df (3-min bars)
        df = df.copy()
        df["ema9"] = indicators.ema(df["close"], 9)
        df["atr14"] = indicators.atr(df, 14)
        

        # Quick inspect before grouping
        print("GENERATE_SIGNALS: df rows =", len(df))
        print("GENERATE_SIGNALS: cols =", df.columns.tolist())
        try:
            print("GENERATE_SIGNALS: index min/max:", df.index.min(), df.index.max())
            print("GENERATE_SIGNALS: index dtype:", df.index.dtype)
            print("GENERATE_SIGNALS: tzinfo (if any):", getattr(df.index, 'tz', None))
        except Exception as e:
            print("GENERATE_SIGNALS: index inspect failed:", e)

        # Call _day_groups to enumerate groups (non-destructive)
        try:
            groups_preview = []
            for i, (d, g) in enumerate(self._day_groups(df)):
                if i < 3:
                    groups_preview.append((d, len(g), g.index.min(), g.index.max()))
                else:
                    break
            print("GENERATE_SIGNALS: _day_groups preview (first 3):", groups_preview, "total_groups_estimate:", "unknown")
        except Exception as e:
            print("GENERATE_SIGNALS: _day_groups iter failed:", e)


        for date, day_df in self._day_groups(df):
            if day_df.empty:
                continue



            # DEBUG: inspect day_df before OR calc
            print("DEBUG_DAY: rows=", len(day_df))
            try:
                print("DEBUG_DAY: idx min/max:", day_df.index.min(), day_df.index.max())
                print("DEBUG_DAY: sample head:", day_df.head().to_string(index=True, max_rows=5))
                print("DEBUG_DAY: sample tail:", day_df.tail().to_string(index=True, max_rows=5))
            except Exception as e:
                print("DEBUG_DAY: inspect failed:", e)

            # If you have a configured session window (string), print it
            try:
                print("DEBUG_SESSION_WINDOW:", getattr(self, "session_start", "08:00"), "->", getattr(self, "session_end", "12:00"))
            except Exception:
                print("DEBUG_SESSION_WINDOW: unknown")


            # define session start timestamp for the day (first index >= session_start)
            # open range window: first open_range_minutes starting at the day's first index
            # We assume df is already filtered to session times and resampled (3-min)
            session_start_ts = day_df.index[0]
            or_end_ts = session_start_ts + timedelta(minutes=self.open_range_minutes)
            or_bars = day_df.loc[(day_df.index > session_start_ts - pd.Timedelta(seconds=1)) & (day_df.index <= or_end_ts)]
            if or_bars.empty:
                # cannot define OR for this day
                continue
            or_high = or_bars["high"].max()
            or_low = or_bars["low"].min()

            # signals only after OR end, within cancel window
            signal_search_start = or_end_ts + pd.Timedelta(microseconds=1)
            signal_search_end = or_end_ts + pd.Timedelta(minutes=self.cancel_within_minutes)
            search_bars = day_df.loc[(day_df.index >= signal_search_start) & (day_df.index <= signal_search_end)]
            if search_bars.empty:
                continue

            # compute recent volume rolling average on day_df for volume confirmation
            if len(day_df) >= self.volume_lookback:
                vol_avg_series = day_df["volume"].rolling(self.volume_lookback, min_periods=1).mean()
            else:
                vol_avg_series = day_df["volume"].expanding().mean()


            # DEBUG PROBE - insert immediately before the loop
            print("ORB_SEARCH_BARS: len=", len(search_bars))
            try:
                print("ORB_SEARCH_BARS idx min/max:", search_bars.index.min(), search_bars.index.max())
            except Exception:
                print("ORB_SEARCH_BARS: index not datetime or empty")


            # iterate bars in search_bars to detect breakout closes
            for ts, row in search_bars.iterrows():
                close = float(row["close"])
                high = float(row["high"])
                low = float(row["low"])
                ema9 = float(row["ema9"]) if pd.notna(row["ema9"]) else None
                vol = float(row["volume"])
                vol_avg = float(vol_avg_series.loc[ts]) if ts in vol_avg_series.index else float(vol_avg_series.iloc[-1])

                # LONG breakout
                if close > or_high:
                    # EMI: EMA slope or price > ema9 for confirmation
                    ema_ok = (ema9 is not None) and (close > ema9)
                    vol_ok = (not self.require_volume_confirmation) or (vol >= vol_avg)
                    
                    # Log breakout candidate evaluation
                    reject_reasons = []
                    if not ema_ok:
                        reject_reasons.append("ema_fail")
                    if not vol_ok:
                        reject_reasons.append("volume_fail")
                    
                    # --- Build candidate debug row (defensive) ---
                    # Ensure timestamp is an ISO string (pd.Timestamp or datetime)
                    try:
                        ts_iso = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
                    except Exception:
                        ts_iso = str(ts)

                    # Robustly get 'open' value (handle different column casing)
                    def _get_row_val(key_list, default=None):
                        for k in key_list:
                            if k in row.index:
                                return row[k]
                        return default

                    open_val = _get_row_val(['open', 'Open', 'OPEN'], None)
                    try:
                        open_f = float(open_val) if open_val is not None else None
                    except Exception:
                        open_f = None

                    # vwap - use existing var if present in scope, otherwise None
                    vwap_val = vwap if 'vwap' in locals() else None

                    # decision flags
                    ema_pass = bool(ema_ok)
                    vol_pass = bool(vol_ok)
                    vwap_pass = True if vwap_val is None else True  # leave True if unused; set logic if you require vwap

                    final_decision = "ACCEPT" if (ema_pass and vol_pass and vwap_pass) else "REJECT"

                    candidate_row = {
                        "timestamp": ts_iso,
                        "local_time": ts_iso,
                        "close": float(close),
                        "open": open_f,
                        "high": float(high),
                        "low": float(low),
                        "volume": int(vol),
                        "or_high": float(or_high),
                        "or_low": float(or_low) if 'or_low' in locals() else None,
                        "breakout_dir": "long",
                        "ema9": float(ema9) if ema9 is not None else None,
                        "vwap": float(vwap_val) if vwap_val is not None else None,
                        "volume_avg": float(vol_avg),
                        # per-check flags (explicit)
                        "ema_pass": ema_pass,
                        "vwap_pass": bool(vwap_pass),
                        "volume_pass": vol_pass,
                        "time_rule_pass": True,  # update if you compute a real time rule
                        # two decision fields: probe vs final (keeps timeline of checks)
                        "decision_probe": "ACCEPT" if (ema_pass and vol_pass) else "REJECT",
                        "decision": final_decision,
                        "reject_reasons": ";".join(reject_reasons) if reject_reasons else "",
                        # optional sizing/plan fields (fill later)
                        "planned_contracts": None,
                        "planned_entry": None,
                        "planned_stop": None,
                        "planned_target": None,
                    }

                    # Non-fatal logging
                    try:
                        orb_debugger.log(candidate_row)
                    except Exception as e:
                        print("ORB_DEBUG log failed:", e)
                    
                    if ema_ok and vol_ok:
                        # determine stop: breakout candle low (structure) or ATR-based if configured
                        if self.stop_policy == "structure":
                            stop_price = float(low)
                        elif self.stop_policy == "atr":
                            stop_price = close - self.atr_multiplier_stop * float(row["atr14"])
                        else:  # fixed
                            stop_price = close - self.stop_fixed_points
                        # initial R
                        initial_r = abs(close - stop_price)
                        t1 = close + self.take_profit_r[0] * initial_r
                        t2 = close + (self.take_profit_r[1] if len(self.take_profit_r) > 1 else (2.0 * initial_r))
                        meta = {"date": str(date), "or_high": float(or_high), "or_low": float(or_low), "reason": "ORB long"}
                        signals_dict =   {
                            "timestamp": ts,
                            "side": "LONG",
                            "signal_price": close,
                            "entry_price_hint": None,   # backtester will enter at next bar open
                            "stop_price": float(stop_price),
                            "target1": float(t1),
                            "target2": float(t2),
                            "meta": meta
                        }
                        self._append_signal_dedup(signals, seen_keys, signals_dict)
                        # one long signal per OR
                        break

                # SHORT breakout
                if close < or_low:
                    ema_ok = (ema9 is not None) and (close < ema9)
                    vol_ok = (not self.require_volume_confirmation) or (vol <= vol_avg)  # for short can accept lower vol on pull?
                    # keep same vol rule for shorts (depends on design); we'll accept vol_ok if vol >= vol_avg (consistent)
                    vol_ok = (not self.require_volume_confirmation) or (vol >= vol_avg)
                    
                    # Log breakout candidate evaluation
                    reject_reasons = []
                    if not ema_ok:
                        reject_reasons.append("ema_fail")
                    if not vol_ok:
                        reject_reasons.append("volume_fail")
                    
                    candidate_row = {
                        "timestamp": ts,
                        "local_time": ts.isoformat(),
                        "close": float(close),
                        "open": float(row["open"]),
                        "high": float(high),
                        "low": float(low),
                        "volume": int(vol),
                        "or_high": float(or_high),
                        "or_low": float(or_low),
                        "breakout_dir": "short",
                        "ema9": float(ema9) if ema9 is not None else None,
                        "vwap": None,
                        "volume_avg": float(vol_avg),
                        "ema_pass": bool(ema_ok),
                        "vwap_pass": True,
                        "volume_pass": bool(vol_ok),
                        "time_rule_pass": True,
                        "decision": "ACCEPT" if (ema_ok and vol_ok) else "REJECT",
                        "reject_reasons": ";".join(reject_reasons) if reject_reasons else "",
                        "planned_contracts": None,
                        "planned_entry": None,
                        "planned_stop": None,
                        "planned_target": None,
                    }
                    orb_debugger.log(candidate_row)
                    
                    if ema_ok and vol_ok:
                        if self.stop_policy == "structure":
                            stop_price = float(high)
                        elif self.stop_policy == "atr":
                            stop_price = close + self.atr_multiplier_stop * float(row["atr14"])
                        else:
                            stop_price = close + self.stop_fixed_points
                        initial_r = abs(close - stop_price)
                        t1 = close - self.take_profit_r[0] * initial_r
                        t2 = close - (self.take_profit_r[1] if len(self.take_profit_r) > 1 else (2.0 * initial_r))
                        meta = {"date": str(date), "or_high": float(or_high), "or_low": float(or_low), "reason": "ORB short"}
                        signals_dict =   {
                            "timestamp": ts,
                            "side": "SHORT",
                            "signal_price": close,
                            "entry_price_hint": None,
                            "stop_price": float(stop_price),
                            "target1": float(t1),
                            "target2": float(t2),
                            "meta": meta
                        }
                        self._append_signal_dedup(signals, seen_keys, signals_dict)
                        break
            
            # Flush debug buffer at end of each day
            orb_debugger.flush()

        if not signals:
            return pd.DataFrame(columns=[
                "timestamp","side","signal_price","entry_price_hint","stop_price","target1","target2","meta"
            ])
        return pd.DataFrame(signals).sort_values("timestamp").reset_index(drop=True)