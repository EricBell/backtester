#!/usr/bin/env python3
"""
Test script to verify session close functionality
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from datetime import datetime, time
from engines.pullback_engine import PullbackStrategy
from core.backtester import Backtester

# Test the session_end detection
def test_session_end_detection():
    print("Testing session end detection fix...")
    
    # Create a mock strategy with session_end = "11:30"
    strategy_params = {
        "session_start": "09:30",
        "session_end": "11:30",  # This should be used instead of global
        "ema_fast": 13,
        "ema_slow": 30,
        "atr_period": 21,
        "target_r": 2.0
    }
    strategy = PullbackStrategy(strategy_params)
    
    # Create mock config (global session_end = "12:00")
    config = {
        "session_start": "09:30", 
        "session_end": "12:00",  # Global setting (should be overridden)
        "contract_meta": {
            "dollars_per_point": 5.0,
            "tick_size": 0.25
        },
        "commission_roundtrip": 1.74
    }
    
    # Create mock data
    mock_data = pd.DataFrame({
        'open': [6000.0],
        'high': [6010.0], 
        'low': [5990.0],
        'close': [6005.0],
        'volume': [1000]
    }, index=[pd.Timestamp("2025-06-16 10:00:00-04:00")])
    
    # Create backtester
    backtester = Backtester(strategy, mock_data, config, outdir="./test_outputs")
    
    # Test the session end detection at different times
    test_times = [
        ("2025-06-16 11:00:00-04:00", False),  # Before session end
        ("2025-06-16 11:30:00-04:00", True),   # At session end (strategy setting)
        ("2025-06-16 12:00:00-04:00", True),   # Past session end
        ("2025-06-16 11:29:00-04:00", False),  # Just before
        ("2025-06-16 11:31:00-04:00", True),   # Just after
    ]
    
    print(f"Strategy session_end: {strategy.session_end}")
    print(f"Global config session_end: {config.get('session_end')}")
    print()
    
    all_passed = True
    for time_str, expected in test_times:
        test_time = pd.Timestamp(time_str)
        result = backtester._is_past_session_end(test_time)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        print(f"{time_str}: {result} (expected {expected}) {status}")
        if result != expected:
            all_passed = False
    
    print(f"\nOverall test result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    return all_passed

if __name__ == "__main__":
    test_session_end_detection()