#!/usr/bin/env python3
from pathlib import Path
import engines.orb_engine as mod
from core.utils import load_csv_parse_datetime

def main():
    # Path to your CSV
    p = Path("data/threedays.csv")
    print("Loading:", p)

    # Load using your loader (returns Title-case columns by design)
    df = load_csv_parse_datetime(p, tz_source=None, tz_target='America/New_York')

    # Normalize column names to lowercase for the engine convenience
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # Strategy params: shorter volume lookback for scalping
    params = {"volume_lookback": 5}

    # Instantiate ORBStrategy with params; fallback to manual attribute set if needed
    try:
        inst = mod.ORBStrategy(params)
    except Exception:
        inst = mod.ORBStrategy({})
        inst.volume_lookback = 5

    # Run signal generation
    print("Running generate_signals with volume_lookback =", getattr(inst, "volume_lookback", "<missing>"))
    sigs = inst.generate_signals(df)

    # Print summary and first rows
    print("Signals returned:", len(sigs))
    if len(sigs):
        print(sigs.head(10).to_string())

    # Write results to outputs
    outdir = Path("outputs")
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "signals_vol5.csv"
    sigs.to_csv(out_path, index=False)
    print("Wrote:", out_path)

if __name__ == "__main__":
    main()