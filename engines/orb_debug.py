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
            # Handle timestamp column - convert parseable ones, keep others as-is
            if 'timestamp' in df.columns:
                # Try to convert timestamps, fallback to string for invalid ones
                timestamp_col = []
                for ts in df['timestamp']:
                    try:
                        timestamp_col.append(pd.to_datetime(ts))
                    except (ValueError, TypeError):
                        # For non-parseable timestamps, use current date for grouping
                        timestamp_col.append(pd.Timestamp.now().normalize())
                df['timestamp_parsed'] = timestamp_col
            elif 'datetime' in df.columns:
                # Similar logic for datetime column
                timestamp_col = []
                for ts in df['datetime']:
                    try:
                        timestamp_col.append(pd.to_datetime(ts))
                    except (ValueError, TypeError):
                        timestamp_col.append(pd.Timestamp.now().normalize())
                df['timestamp_parsed'] = timestamp_col
            else:
                # No timestamp column found, use current time
                df['timestamp_parsed'] = pd.Timestamp.now()
            
            # Group by date (using parsed timestamps for grouping)
            df['date'] = df['timestamp_parsed'].dt.date
            self.outdir.mkdir(parents=True, exist_ok=True)
            for date_val, grp in df.groupby('date'):
                fname = self.outdir / f"orb_debug_{date_val}.csv"
                # Drop the helper columns before saving
                grp = grp.drop(columns=['date', 'timestamp_parsed'])
                grp.to_csv(fname, index=False)
                logger.info("Wrote ORB debug file: %s (rows=%d)", str(fname), len(grp))
        except Exception as e:
            logger.exception("orb_debug flush failed: %s", e)
        finally:
            # clear buffer
            self._rows = []