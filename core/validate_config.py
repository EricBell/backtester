from pathlib import Path
from utils import load_yaml, validate_config_basic

cfg = load_yaml(Path("outputs/config_used.yaml"))
errs = validate_config_basic(cfg)
if errs:
    print("Config validation errors:")
    for e in errs:
        print(" -", e)
else:
    print("Config OK")