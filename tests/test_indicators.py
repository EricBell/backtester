import pandas as pd
import numpy as np
from core import indicators

def test_ema_basic():
    s = pd.Series([1.0,2.0,3.0,4.0,5.0])
    e = indicators.ema(s, 3)
    assert len(e) == len(s)
    # Basic sanity checks
    assert e.iloc[-1] > 0