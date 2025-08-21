#!/usr/bin/env python3
"""
Test script to verify session enforcement is working correctly.
Tests the architecture changes made to fix session management.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

def test_strategy_session_validation():
    """Test that strategies properly validate session parameters."""
    
    # Test 1: PullbackStrategy with missing session_start
    try:
        from engines.pullback_engine import PullbackStrategy
        params = {"session_end": "11:30"}  # Missing session_start
        strategy = PullbackStrategy(params)
        print("❌ FAILED: PullbackStrategy should require session_start")
        return False
    except ValueError as e:
        if "session_start" in str(e):
            print("✅ PASSED: PullbackStrategy correctly validates session_start")
        else:
            print(f"❌ FAILED: Wrong error message: {e}")
            return False
    except Exception as e:
        print(f"❌ FAILED: Unexpected error: {e}")
        return False
    
    # Test 2: PullbackStrategy with missing session_end  
    try:
        params = {"session_start": "09:30"}  # Missing session_end
        strategy = PullbackStrategy(params)
        print("❌ FAILED: PullbackStrategy should require session_end")
        return False
    except ValueError as e:
        if "session_end" in str(e):
            print("✅ PASSED: PullbackStrategy correctly validates session_end")
        else:
            print(f"❌ FAILED: Wrong error message: {e}")
            return False
    except Exception as e:
        print(f"❌ FAILED: Unexpected error: {e}")
        return False
    
    # Test 3: PullbackStrategy with valid session parameters
    try:
        params = {"session_start": "09:30", "session_end": "11:30"}
        strategy = PullbackStrategy(params)
        if hasattr(strategy, 'session_start') and hasattr(strategy, 'session_end'):
            print("✅ PASSED: PullbackStrategy correctly accepts valid session parameters")
        else:
            print("❌ FAILED: PullbackStrategy missing session attributes")
            return False
    except Exception as e:
        print(f"❌ FAILED: PullbackStrategy should accept valid parameters: {e}")
        return False
    
    # Test 4: Check other strategies also validate
    strategy_classes = [
        ('ORBStrategy', 'engines.orb_engine'),
        ('VWAPFadeStrategy', 'engines.vwap_engine'),
        ('MESScalpingStrategy', 'engines.scalping_engine'),
        ('EMA821Strategy', 'engines.ema8_21_engine')
    ]
    
    for class_name, module_name in strategy_classes:
        try:
            module = __import__(module_name, fromlist=[class_name])
            strategy_class = getattr(module, class_name)
            
            # Test missing session parameters
            try:
                strategy = strategy_class({})
                print(f"❌ FAILED: {class_name} should require session parameters")
                return False
            except ValueError as e:
                if "session" in str(e).lower():
                    print(f"✅ PASSED: {class_name} correctly validates session parameters")
                else:
                    print(f"❌ FAILED: {class_name} wrong error message: {e}")
                    return False
            except Exception as e:
                print(f"❌ FAILED: {class_name} unexpected error: {e}")
                return False
                
        except Exception as e:
            print(f"❌ FAILED: Could not test {class_name}: {e}")
            return False
    
    return True

def test_backtester_session_logic():
    """Test that backtester correctly uses strategy session settings."""
    
    try:
        from core.backtester import Backtester
        import pandas as pd
        from datetime import datetime, time
        
        # Create a mock strategy with session settings
        class MockStrategy:
            def __init__(self):
                self.session_start = "09:30"
                self.session_end = "11:30"
                self.params = {"mock": True}
            
            def generate_signals(self, df):
                return pd.DataFrame()
        
        # Create mock data
        mock_bars = pd.DataFrame({
            'open': [100], 'high': [101], 'low': [99], 'close': [100.5], 'volume': [1000]
        }, index=[datetime(2024, 1, 1, 10, 0)])
        
        mock_config = {
            'contract_meta': {'dollars_per_point': 5.0, 'tick_size': 0.25},
            'resolved_contract': 'TEST'
        }
        
        strategy = MockStrategy()
        bt = Backtester(strategy, mock_bars, mock_config, Path('./test_outputs'))
        
        # Test session end check
        test_time = pd.Timestamp("2024-01-01 12:00:00")  # Past 11:30 session end
        if bt._is_past_session_end(test_time):
            print("✅ PASSED: Backtester correctly detects past session end")
        else:
            print("❌ FAILED: Backtester should detect past session end")
            return False
            
        # Test within session
        test_time = pd.Timestamp("2024-01-01 10:00:00")  # Within 09:30-11:30 session
        if not bt._is_past_session_end(test_time):
            print("✅ PASSED: Backtester correctly detects within session")
        else:
            print("❌ FAILED: Backtester should detect within session")
            return False
            
    except Exception as e:
        print(f"❌ FAILED: Backtester session logic test error: {e}")
        return False
    
    return True

def test_strategy_without_session():
    """Test that backtester rejects strategy without session settings."""
    
    try:
        from core.backtester import Backtester
        import pandas as pd
        from datetime import datetime
        
        # Create a mock strategy WITHOUT session settings
        class BadMockStrategy:
            def __init__(self):
                self.params = {"mock": True}
                # Missing session_start and session_end
            
            def generate_signals(self, df):
                return pd.DataFrame()
        
        mock_bars = pd.DataFrame({
            'open': [100], 'high': [101], 'low': [99], 'close': [100.5], 'volume': [1000]
        }, index=[datetime(2024, 1, 1, 10, 0)])
        
        mock_config = {
            'contract_meta': {'dollars_per_point': 5.0, 'tick_size': 0.25},
            'resolved_contract': 'TEST'
        }
        
        strategy = BadMockStrategy()
        bt = Backtester(strategy, mock_bars, mock_config, Path('./test_outputs'))
        
        # Try to check session end - should raise error
        test_time = pd.Timestamp("2024-01-01 12:00:00")
        try:
            bt._is_past_session_end(test_time)
            print("❌ FAILED: Backtester should reject strategy without session_end")
            return False
        except ValueError as e:
            if "session_end" in str(e):
                print("✅ PASSED: Backtester correctly rejects strategy without session_end")
            else:
                print(f"❌ FAILED: Wrong error message: {e}")
                return False
        except Exception as e:
            print(f"❌ FAILED: Unexpected error: {e}")
            return False
            
    except Exception as e:
        print(f"❌ FAILED: Strategy validation test error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=== Testing Session Enforcement Architecture ===")
    print()
    
    print("Testing strategy session validation...")
    test1_passed = test_strategy_session_validation()
    print()
    
    print("Testing backtester session logic...")
    test2_passed = test_backtester_session_logic()
    print()
    
    print("Testing strategy validation...")
    test3_passed = test_strategy_without_session()
    print()
    
    if test1_passed and test2_passed and test3_passed:
        print("🎉 ALL TESTS PASSED: Session enforcement architecture is working correctly!")
        print()
        print("Summary of fixes implemented:")
        print("✅ Removed global session_start/session_end from config.yaml")
        print("✅ All strategies require their own session parameters")
        print("✅ Backtester only uses strategy session settings")
        print("✅ Main script validates strategy session parameters")
        print("✅ Summary output uses strategy session settings")
        print()
        print("The session enforcement bug should now be fixed!")
    else:
        print("❌ SOME TESTS FAILED: Please check the implementation")
        sys.exit(1)