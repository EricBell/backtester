import pytest
from pathlib import Path
import pandas as pd
from core.utils import load_csv_parse_datetime

def _write_csv(tmp_path: Path, name: str, contents: str) -> Path:
    p = tmp_path / name
    p.write_text(contents)
    return p

def test_basic_combined_datetime_naive(tmp_path):
    csv = _write_csv(tmp_path, "basic.csv",
        "DateTime,Open,High,Low,Close,Volume\n"
        "2025-06-12 08:00:00,6062.5,6064.25,6062.25,6063.75,29\n"
        "2025-06-12 08:03:00,6063.75,6065.25,6063.75,6065.25,11\n"
    )
    df = load_csv_parse_datetime(csv, tz_source=None, tz_target='America/New_York')
    assert df.shape[0] == 2
    assert 'date' in df.columns and 'time' in df.columns
    assert df.index.tz.zone == 'America/New_York'
    assert df.iloc[0]['Open'] == 6062.5

def test_utc_to_et_conversion(tmp_path):
    csv = _write_csv(tmp_path, "utc.csv",
        "DateTime,Open,High,Low,Close,Volume\n"
        "2025-06-12 12:00:00,6062.5,6064.25,6062.25,6063.75,29\n"
    )
    df = load_csv_parse_datetime(csv, tz_source='UTC', tz_target='America/New_York')
    # 12:00 UTC -> 08:00 ET in June
    assert df.index[0].hour == 8

def test_separate_date_time_columns(tmp_path):
    csv = _write_csv(tmp_path, "separate.csv",
        "date,time,Open,High,Low,Close,Volume\n"
        "2025-06-12,08:00:00,6062.5,6064.25,6062.25,6063.75,29\n"
    )
    df = load_csv_parse_datetime(csv, tz_source=None, tz_target='America/New_York')
    assert df.shape[0] == 1
    assert df.iloc[0]['Close'] == 6063.75

def test_missing_ohlcv_raises(tmp_path):
    csv = _write_csv(tmp_path, "missing.csv",
        "DateTime,Open,High,Close,Volume\n"
        "2025-06-12 08:00:00,6062.5,6064.25,6063.75,29\n"
    )
    with pytest.raises(ValueError):
        load_csv_parse_datetime(csv, tz_source=None, tz_target='America/New_York')

def test_duplicate_timestamps_raises(tmp_path):
    csv = _write_csv(tmp_path, "dups.csv",
        "DateTime,Open,High,Low,Close,Volume\n"
        "2025-06-12 08:00:00,6062.5,6064.25,6062.25,6063.75,29\n"
        "2025-06-12 08:00:00,6062.5,6064.25,6062.25,6063.75,30\n"
    )
    with pytest.raises(ValueError):
        load_csv_parse_datetime(csv, tz_source=None, tz_target='America/New_York')

def test_tick_alignment_detects(tmp_path):
    csv = _write_csv(tmp_path, "tick.csv",
        "DateTime,Open,High,Low,Close,Volume\n"
        "2025-06-12 08:00:00,6062.3,6064.25,6062.25,6063.75,29\n"  # 6062.3 not multiple of 0.25
    )
    with pytest.raises(ValueError):
        load_csv_parse_datetime(csv, tz_source=None, tz_target='America/New_York', tick_size=0.25)

def test_header_whitespace_handling(tmp_path):
    # headers with spaces should be normalized
    csv = _write_csv(tmp_path, "ws.csv",
        " DateTime , Open , High , Low , Close , Volume \n"
        "2025-06-12 08:00:00,6062.5,6064.25,6062.25,6063.75,29\n"
    )
    df = load_csv_parse_datetime(csv, tz_source=None, tz_target='America/New_York')
    assert 'Open' in df.columns
    assert df.iloc[0]['Open'] == 6062.5