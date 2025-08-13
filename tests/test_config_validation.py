import pytest
from core.utils import load_yaml, validate_config
from pathlib import Path

def test_config_valid():
    cfg = load_yaml(Path("config.yaml"))
    errs = validate_config(cfg)
    assert errs == [], f"Config validation failed: {errs}"

def test_contract_entries_present():
    cfg = load_yaml(Path("config.yaml"))
    assert "contracts" in cfg and len(cfg["contracts"]) > 0
    for sym, meta in cfg["contracts"].items():
        assert meta.get("dollars_per_point", 0) > 0
        assert meta.get("tick_size", 0) > 0