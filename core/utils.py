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


from pathlib import Path
from typing import Optional, Sequence
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def load_csv_parse_datetime(path: Path,
                            datetime_col_candidates: Optional[Sequence[str]] = None,
                            tz_source: Optional[str] = None,
                            tz_target: str = 'America/New_York',
                            require_ohlcv: bool = True,
                            tick_size: Optional[float] = 0.25) -> pd.DataFrame:
    """
    Load CSV with combined DateTime or separate date & time columns and return a DataFrame
    with index = timezone-aware DatetimeIndex (tz_target) and columns:
      date (YYYY-MM-DD), time (HH:MM:SS), Open, High, Low, Close, Volume

    Params:
      - path: Path to CSV
      - datetime_col_candidates: optional list of candidate datetime column names
      - tz_source: timezone of timestamps in file (e.g., 'UTC'). If provided, naive timestamps are localized to tz_source then converted to tz_target.
      - tz_target: timezone for returned index (default 'America/New_York')
      - require_ohlcv: raise if OHLCV columns are missing
      - tick_size: optional tick size (in index points) to validate price alignment (e.g. 0.25)
    Note: If tz_source is None and timestamps are naive, this function will localize naive timestamps to tz_target.
    """
    # Read + normalize headers
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    # datetime candidates
    if datetime_col_candidates is None:
        datetime_col_candidates = ["datetime", "date_time", "timestamp", "date", "time"]
    candidates_lower = [c.lower() for c in datetime_col_candidates]

    # find combined datetime column preferring combined names (not separate 'date'/'time')
    datetime_col = None
    for col in df.columns:
        lc = col.strip().lower()
        if lc in candidates_lower and lc not in ('date', 'time'):
            datetime_col = col
            break

    # build datetime Series
    if datetime_col:
        dt = pd.to_datetime(df[datetime_col], errors='raise')
    else:
        cols_lower = [c.lower() for c in df.columns]
        if 'date' in cols_lower and 'time' in cols_lower:
            date_col = next(c for c in df.columns if c.lower() == 'date')
            time_col = next(c for c in df.columns if c.lower() == 'time')
            dt = pd.to_datetime(df[date_col].astype(str).str.strip() + ' ' + df[time_col].astype(str).str.strip(),
                                errors='raise')
        else:
            # fallback: try first column
            first_col = df.columns[0]
            dt = pd.to_datetime(df[first_col], errors='raise')

    # tz handling
    if dt.dt.tz is None:
        if tz_source:
            try:
                dt = dt.dt.tz_localize(tz_source).dt.tz_convert(tz_target)
            except Exception as e:
                raise ValueError(f"Failed to localize datetimes to tz_source={tz_source}: {e}") from e
        else:
            # Explicit decision: treat naive timestamps as already in tz_target
            # (If your CSV is in UTC, pass tz_source='UTC' to convert properly.)
            dt = dt.dt.tz_localize(tz_target)
    else:
        if tz_target:
            try:
                dt = dt.dt.tz_convert(tz_target)
            except Exception as e:
                raise ValueError(f"Failed to convert datetimes to tz_target={tz_target}: {e}") from e

    # Normalize OHLCV column names (map to Title case)
    col_map = {}
    for col in df.columns:
        lc = col.strip().lower()
        if lc in ('open', 'high', 'low', 'close', 'volume'):
            col_map[col] = lc.capitalize()
    df = df.rename(columns=col_map)

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume'] if require_ohlcv else []
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}. Found columns: {df.columns.tolist()}")

    # attach datetime and set index
    df['datetime'] = dt
    df = df.set_index('datetime').sort_index()

    # create snapshot schema date/time columns
    df['date'] = df.index.strftime('%Y-%m-%d')
    df['time'] = df.index.strftime('%H:%M:%S')

    # basic validations
    dup_count = df.index.duplicated().sum()
    if dup_count:
        dup_idxs = df.index[df.index.duplicated()]
        first_dup = dup_idxs[0] if len(dup_idxs) else None
        raise ValueError(f"Duplicate timestamps found in CSV: {dup_count} duplicates (first dup at {first_dup})")

    nulls = df[required_cols].isnull().sum().to_dict() if required_cols else {}
    if any(v > 0 for v in nulls.values()):
        raise ValueError(f"Found nulls in required OHLCV columns: {nulls}")

    # optional tick alignment check (numerically stable)
    if tick_size:
        misaligned = {}
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                # convert to float and compute number of ticks
                div = df[col].astype(float) / float(tick_size)
                # mismatch if distance to nearest integer > tolerance
                small_tol = 1e-6
                mismatches = (np.abs(div - np.round(div)) > small_tol)
                misaligned_count = int(mismatches.sum())
                if misaligned_count:
                    misaligned[col] = misaligned_count
        if misaligned:
            raise ValueError(
                f"Price tick alignment issue detected (not multiples of {tick_size}): {misaligned}\n"
                f"This means your price data is not aligned to the expected tick size.\n"
                f"Solutions:\n"
                f"1. Verify the tick_size in your config.yaml for this contract matches your data\n"
                f"2. Check if your data source uses a different tick size than configured\n"
                f"3. Set tick_size=None in load_csv_parse_datetime() to disable this validation\n"
                f"4. Ensure your price data is properly rounded to the contract's tick size"
            )



    # logging summary
    try:
        logger.info("Loaded CSV '%s' rows=%d index_min=%s index_max=%s", str(path), len(df), df.index.min(), df.index.max())
    except Exception:
        # best-effort logging; don't crash on logging
        pass

    return df[['date', 'time', 'Open', 'High', 'Low', 'Close', 'Volume']]

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


def extract_contract_symbol_from_filename(filename: str) -> str:
    """
    Extract contract symbol from data filename.
    
    The naming convention expects the first 2-3 letters of the filename to be 
    the futures contract symbol (e.g., 'MES' from 'MES-0612-0806.Last_15min.csv', 
    'M2K' from 'M2K250701-0820.Last_15min.csv').
    
    Args:
        filename: Data filename (with or without path)
        
    Returns:
        Extracted contract symbol (uppercase)
    """
    import re
    
    # Get just the filename without path
    base_filename = Path(filename).name
    
    # Enhanced pattern to handle various contract symbol formats:
    # - 2-3 letters: MES, ES, NQ
    # - Letter-Digit-Letter: M2K (Micro Russell 2000)
    # - Multiple patterns at start of filename
    patterns = [
        r'^([A-Za-z]\d[A-Za-z])',      # Pattern like M2K (letter-digit-letter)
        r'^([A-Za-z]{3})',             # 3 letters like MES, MNQ  
        r'^([A-Za-z]{2})',             # 2 letters like ES, NQ
    ]
    
    for pattern in patterns:
        match = re.match(pattern, base_filename)
        if match:
            return match.group(1).upper()
    
    # Fallback: extract all letters from first word before separators
    first_part = base_filename.split('-')[0].split('_')[0].split('.')[0]
    # For mixed alphanumeric, preserve the pattern
    if any(c.isdigit() for c in first_part) and any(c.isalpha() for c in first_part):
        # Keep alphanumeric pattern for contract symbols like M2K
        alphanumeric = ''.join(c for c in first_part if c.isalnum())
        return alphanumeric.upper() if alphanumeric else ""
    else:
        # Extract only letters for pure letter symbols
        letters_only = ''.join(c for c in first_part if c.isalpha())
        return letters_only.upper() if letters_only else ""


def validate_data_file_contract(data_file_path: str, config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate that the data file's contract symbol matches configured contracts.
    
    Args:
        data_file_path: Path to the data file
        config: Configuration dictionary containing contracts
        
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if validation passes, False otherwise
        - error_message: Empty string if valid, error description if invalid
    """
    try:
        # Extract contract symbol from filename
        extracted_symbol = extract_contract_symbol_from_filename(data_file_path)
        
        if not extracted_symbol:
            return False, f"Could not extract contract symbol from filename: {Path(data_file_path).name}"
        
        # Get configured contracts
        contracts = config.get("contracts", {})
        
        if not contracts:
            return False, "No contracts configured in config.yaml"
        
        # Check if extracted symbol exists in configured contracts
        configured_symbols = list(contracts.keys())
        
        if extracted_symbol not in configured_symbols:
            return False, (
                f"Data file contract mismatch: '{extracted_symbol}' (from {Path(data_file_path).name}) "
                f"not found in configured contracts: {configured_symbols}. "
                f"Please ensure the data file matches one of the configured contract symbols."
            )
        
        return True, ""
        
    except Exception as e:
        return False, f"Error validating data file contract: {str(e)}"

