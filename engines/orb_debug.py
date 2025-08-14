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
            # Ensure timestamp column exists and is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif 'datetime' in df.columns:
                df['timestamp'] = pd.to_datetime(df['datetime'])
            else:
                # try converting index if present; else add current time
                df['timestamp'] = pd.to_datetime(df.get('timestamp', pd.Timestamp.now()))
            # group by date (in the timestamp's timezone if tz-aware)
            df['date'] = df['timestamp'].dt.date
            self.outdir.mkdir(parents=True, exist_ok=True)
            for date_val, grp in df.groupby('date'):
                fname = self.outdir / f"orb_debug_{date_val}.csv"
                grp = grp.drop(columns=['date'])
                grp.to_csv(fname, index=False)
                logger.info("Wrote ORB debug file: %s (rows=%d)", str(fname), len(grp))
        except Exception as e:
            logger.exception("orb_debug flush failed: %s", e)
        finally:
            # clear buffer
            self._rows = []