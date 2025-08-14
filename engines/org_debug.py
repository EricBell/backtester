# engines/orb_debug.py
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class OrbDebug:
    """
    Simple debug helper for ORB engine. Collects candidate rows (dicts) and writes
    one CSV per date (YYYY-MM-DD) into outdir. Disabled when enabled=False.
    """

    def __init__(self, enabled: bool = False, outdir: Optional[Path] = None):
        self.enabled = bool(enabled)
        self.outdir = Path(outdir) if outdir else Path("outputs/orb_debug")
        self._rows: List[Dict] = []

    def log(self, row: Dict):
        """
        Append one candidate row. Keys are arbitrary but should include a 'timestamp'
        or 'datetime' key (ISO or pandas Timestamp-compatible).
        """
        if not self.enabled:
            return
        # shallow copy defensive
        self._rows.append(dict(row))

def flush(self):
    """Write buffered rows to CSV files (grouped by calendar date)."""
    if not self.enabled or not self._rows:
        return
    try:
        df = pd.DataFrame(self._rows)

        # Prefer 'timestamp' then 'datetime' column for time info.
        ts_col = None
        if 'timestamp' in df.columns:
            ts_col = 'timestamp'
        elif 'datetime' in df.columns:
            ts_col = 'datetime'

        if ts_col is not None:
            # Try to parse timestamps robustly; coerce unparsable -> NaT (no exception)
            df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
            # If there are any NaT, fill with 'now' (timezone-naive)
            if df[ts_col].isna().any():
                num_bad = int(df[ts_col].isna().sum())
                logger.warning("orb_debug: %d rows had unparsable timestamps; filling with current time", num_bad)
                now_ts = pd.Timestamp.now()
                df.loc[df[ts_col].isna(), ts_col] = now_ts
            # Normalize column name to 'timestamp' for downstream convenience
            if ts_col != 'timestamp':
                df['timestamp'] = df[ts_col]
        else:
            # No timestamp-like column found; create one from 'now'
            df['timestamp'] = pd.Timestamp.now()

        # Ensure timestamp column is datetime dtype now
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        # final fallback: any remaining NaT -> now
        if df['timestamp'].isna().any():
            df.loc[df['timestamp'].isna(), 'timestamp'] = pd.Timestamp.now()

        # Group by calendar date using the timestamp (date portion)
        df['date'] = df['timestamp'].dt.date

        # Write per-date CSVs
        self.outdir.mkdir(parents=True, exist_ok=True)
        for date_val, grp in df.groupby('date'):
            fname = self.outdir / f"orb_debug_{date_val}.csv"
            grp = grp.drop(columns=['date'])
            grp.to_csv(fname, index=False)
            logger.info("Wrote ORB debug file: %s (rows=%d)", str(fname), len(grp))
    except Exception as e:
        logger.exception("orb_debug flush failed: %s", e)
    finally:
        # clear buffer in all cases
        self._rows = []        

def flush2(self):
    """Write buffered rows to CSV files (grouped by calendar date)."""
    if not self.enabled or not self._rows:
        return
    try:
        df = pd.DataFrame(self._rows)

        # Prefer 'timestamp' then 'datetime' column for time info.
        ts_col = None
        if 'timestamp' in df.columns:
            ts_col = 'timestamp'
        elif 'datetime' in df.columns:
            ts_col = 'datetime'

        if ts_col is not None:
            # Try to parse timestamps robustly; coerce unparsable -> NaT (no exception)
            df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
            # If there are any NaT, fill with 'now' (timezone-naive)
            if df[ts_col].isna().any():
                num_bad = int(df[ts_col].isna().sum())
                logger.warning("orb_debug: %d rows had unparsable timestamps; filling with current time", num_bad)
                now_ts = pd.Timestamp.now()
                df.loc[df[ts_col].isna(), ts_col] = now_ts
            # Normalize column name to 'timestamp' for downstream convenience
            if ts_col != 'timestamp':
                df['timestamp'] = df[ts_col]
        else:
            # No timestamp-like column found; create one from 'now'
            df['timestamp'] = pd.Timestamp.now()

        # Ensure timestamp column is datetime dtype now
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        # final fallback: any remaining NaT -> now
        if df['timestamp'].isna().any():
            df.loc[df['timestamp'].isna(), 'timestamp'] = pd.Timestamp.now()

        # Group by calendar date using the timestamp (date portion)
        df['date'] = df['timestamp'].dt.date

        # Write per-date CSVs
        self.outdir.mkdir(parents=True, exist_ok=True)
        for date_val, grp in df.groupby('date'):
            fname = self.outdir / f"orb_debug_{date_val}.csv"
            grp = grp.drop(columns=['date'])
            grp.to_csv(fname, index=False)
            logger.info("Wrote ORB debug file: %s (rows=%d)", str(fname), len(grp))
    except Exception as e:
        logger.exception("orb_debug flush failed: %s", e)
    finally:
        # clear buffer in all cases
        self._rows = []        

   