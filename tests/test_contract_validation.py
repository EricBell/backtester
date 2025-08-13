from core import utils
from pathlib import Path
import tempfile
import yaml

def test_validate_contract_symbol():
    cfg = {
        "contracts": {
            "MES": {"dollars_per_point": 5.0},
            "MNQ": {"dollars_per_point": 2.0}
        }
    }
    assert utils.validate_contract_symbol("MES", cfg) is True
    assert utils.validate_contract_symbol("UNKNOWN", cfg) is False