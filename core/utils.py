"""
core/utils.py - helpers for config, CSV loading, date parsing, session filtering, contract validation.
"""

from pathlib import Path
import yaml
import pandas as pd
from datetime import datetime, time
from typing import Tuple, Optional, Dict, Any
import json

def get_builtin_defaults() -> Dict[str, Any]:
    return {
        "timezone": "America/New_York",
        "commission_roundtrip": 1.75,
        "slippage_points": 0.25,
        "session_start": "08:00",
        "session_end": "12:00",
        "resample_minutes": 3,
        "max_daily_loss": 200.0,
        "max_trades_per_day": 6,
        "meta_file_support_enabled": False,
        "contracts": {},
    }


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def save_yaml(obj: dict, path: Path):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def merge_overrides(config: dict, overrides: dict) -> dict:
    merged = dict(get_builtin_defaults())
    if config:
        merged.update(config)
    # apply overrides (only non-None)
    for k, v in overrides.items():
        if v is not None:
            merged[k] = v
    return merged


def missing_contract_error(supported_symbols: list) -> str:
    return (
        "ERROR: --contract is required and was not provided.\n"
        f"Supported contract symbols: {', '.join(supported_symbols)}\n"
        "Example: --contract MES\n"
        "Please specify the contract symbol to ensure correct P&L scaling."
    )


def invalid_contract_error(symbol: str, supported_symbols: list) -> str:
    return (
        f"ERROR: Unknown contract '{symbol}'. Supported contract symbols: {', '.join(supported_symbols)}\n"
        "Please add the contract to your config.yaml under 'contracts' or use a supported symbol.\n"
        "Example: --contract MES"
    )


def validate_contract_symbol(symbol: str, merged_config: dict) -> bool:
    ctrs = merged_config.get("contracts", {}) or {}
    return symbol in ctrs


def load_csv_parse_datetime(path: Path, tz: Optional[str]) -> pd.DataFrame:
    """
    Load CSV with expected columns [date, time, open, high, low, close, volume].
    Combine date+time into a single timezone-aware datetime index.
    """
    df = pd.read_csv(path)
    # Expect date,time columns; try to be flexible
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"])
    else:
        # Some CSVs use date+time columns
        if "date" in df.columns and "time" in df.columns:
            dt = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
        else:
            # Fallback: try to parse the index or first column
            try:
                first_col = df.columns[0]
                dt = pd.to_datetime(df[first_col])
            except Exception as e:
                raise ValueError("Could not find date/time columns in CSV. Expected 'date' and 'time' columns or 'datetime' column.") from e

    # Localize/convert timezone
    if dt.dt.tz is None:
        # naive -> localize to tz if provided
        if tz:
            dt = dt.dt.tz_localize(tz)
        else:
            dt = dt.dt.tz_localize("UTC")  # fallback
    else:
        # convert to tz if provided
        if tz:
            dt = dt.dt.tz_convert(tz)

    df["datetime"] = dt
    df = df.set_index("datetime").sort_index()
    return df


def derive_data_range(df) -> Tuple[datetime, datetime]:
    idx = df.index
    return idx.min().to_pydatetime(), idx.max().to_pydatetime()


def resolve_effective_date_range(data_start, data_end, cli_start: Optional[str], cli_end: Optional[str]):
    # cli_start and cli_end are strings YYYY-MM-DD or None
    if cli_start is None and cli_end is None:
        return data_start.date().isoformat(), data_end.date().isoformat()
    # Validate provided values are within data range
    from datetime import datetime
    try:
        s = datetime.fromisoformat(cli_start).date() if cli_start else data_start.date()
        e = datetime.fromisoformat(cli_end).date() if cli_end else data_end.date()
    except Exception:
        return None, None
    if s < data_start.date() or e > data_end.date():
        return None, None
    return s.isoformat(), e.isoformat()


def filter_date_range(df, start_iso: str, end_iso: str):
    # start_iso/end_iso are dates in YYYY-MM-DD
    start_ts = pd.to_datetime(start_iso).tz_localize(df.index.tz)
    end_ts = pd.to_datetime(end_iso).tz_localize(df.index.tz) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return df.loc[(df.index >= start_ts) & (df.index <= end_ts)]


def session_filter(df, session_start: str = "08:00", session_end: str = "12:00"):
    # session_start/ session_end strings "HH:MM" in same timezone as df index
    start_h, start_m = [int(x) for x in session_start.split(":")]
    end_h, end_m = [int(x) for x in session_end.split(":")]
    def in_session(ts):
        local = ts.tz_convert(ts.tz) if ts.tz is not None else ts
        tod = local.timetz()
        return (time(start_h, start_m) <= tod <= time(end_h, end_m))
    mask = df.index.map(in_session)
    return df.loc[mask]


def resample_to_n_minutes(df, n: int):
    # Simple resample on the datetime index to n-minute bars. Label/closed set to right by default.
    # Handle both lowercase and capitalized column names
    available_cols = df.columns.tolist()
    
    # Find the correct column names (case-insensitive)
    ohlc = {}
    for target, agg_func in [("open", "first"), ("high", "max"), ("low", "min"), ("close", "last"), ("volume", "sum")]:
        # Look for exact match first, then case-insensitive
        col_name = None
        if target in available_cols:
            col_name = target
        elif target.capitalize() in available_cols:
            col_name = target.capitalize()
        elif target.upper() in available_cols:
            col_name = target.upper()
        
        if col_name:
            ohlc[col_name] = agg_func
    res = df.resample(f"{n}min", label="right", closed="right").agg(ohlc)
    res = res.dropna(how="any")
    return res


def save_json(obj: dict, path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)

def validate_config_basic(cfg: dict):
    errors = []
    # Globals
    if "timezone" not in cfg or not cfg["timezone"]:
        errors.append("timezone is missing")
    if cfg.get("commission_roundtrip", -1) < 0:
        errors.append("commission_roundtrip must be >= 0")
    if cfg.get("resample_minutes", 0) < 1:
        errors.append("resample_minutes must be >= 1")
    if cfg.get("require_contract", False) and not cfg.get("contracts"):
        errors.append("require_contract is true but no contracts are defined")
    # Contract entries
    ctrs = cfg.get("contracts", {})
    for k, v in ctrs.items():
        if "dollars_per_point" not in v or v["dollars_per_point"] <= 0:
            errors.append(f"contracts.{k}.dollars_per_point must be > 0")
        if "tick_size" not in v or v["tick_size"] <= 0:
            errors.append(f"contracts.{k}.tick_size must be > 0")
    return errors        

def validate_config(cfg: dict) -> list:
    """
    Validate config dict. Returns list of error messages (empty if valid).
    Call this early in main() after merging CLI/config/defaults.
    """
    errors = []

    # Basic globals
    if not cfg.get("timezone"):
        errors.append("timezone is missing or empty")
    if cfg.get("commission_roundtrip", -1) < 0:
        errors.append("commission_roundtrip must be >= 0")
    if cfg.get("slippage_points", -1) < 0:
        errors.append("slippage_points must be >= 0")
    if cfg.get("resample_minutes", 0) < 1:
        errors.append("resample_minutes must be >= 1")
    if cfg.get("min_trades_for_stats_warning", 0) < 1:
        errors.append("min_trades_for_stats_warning must be >= 1")

    # require_contract logic
    if cfg.get("require_contract", False) and not cfg.get("contracts"):
        errors.append("require_contract is true but 'contracts' mapping is missing or empty")

    # Contracts mapping checks
    contracts = cfg.get("contracts", {})
    if not isinstance(contracts, dict):
        errors.append("'contracts' must be a mapping")
    else:
        for sym, meta in contracts.items():
            if not isinstance(meta, dict):
                errors.append(f"contracts.{sym} must be a mapping")
                continue
            if meta.get("dollars_per_point", 0) <= 0:
                errors.append(f"contracts.{sym}.dollars_per_point must be > 0")
            if meta.get("tick_size", 0) <= 0:
                errors.append(f"contracts.{sym}.tick_size must be > 0")

    # ORB engine checks
    orb = cfg.get("orb", {})
    if orb:
        or_minutes = orb.get("open_range_minutes", 0)
        if not isinstance(or_minutes, int) or or_minutes < 1:
            errors.append("orb.open_range_minutes must be integer >= 1")
        tprs = orb.get("take_profit_r", [])
        if not isinstance(tprs, list) or any((not isinstance(x, (int, float)) or x <= 0) for x in tprs):
            errors.append("orb.take_profit_r must be a list of positive numbers")
        if orb.get("stop_policy") not in {"structure", "atr", "fixed"}:
            errors.append("orb.stop_policy must be one of: structure, atr, fixed")

    # Pullback checks
    pb = cfg.get("pullback", {})
    if pb:
        ef = pb.get("ema_fast", None)
        es = pb.get("ema_slow", None)
        if not (isinstance(ef, int) and isinstance(es, int) and ef > 0 and es > 0):
            errors.append("pullback.ema_fast and ema_slow must be positive integers")
        if ef >= es:
            # not fatal but warnable
            errors.append("pullback.ema_fast should normally be smaller than ema_slow")

    # VWAP checks
    vwap = cfg.get("vwap", {})
    if vwap:
        if vwap.get("threshold_points", -1) < 0:
            errors.append("vwap.threshold_points must be >= 0")

    return errors

def round_to_tick(price: float, tick: float, direction: str = "nearest") -> float:
    """
    Round price to contract tick size.
    direction: "nearest" | "up" | "down"
    """
    if tick <= 0:
        return price
    ticks = price / tick
    import math
    if direction == "nearest":
        rounded = round(ticks) * tick
    elif direction == "up":
        rounded = math.ceil(ticks) * tick
    elif direction == "down":
        rounded = math.floor(ticks) * tick
    else:
        rounded = round(ticks) * tick
    # avoid -0.0 float formatting
    return float(rounded)

