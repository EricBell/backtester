#!/usr/bin/env python3
"""
main.py - CLI entrypoint for MES intraday backtester (skeleton).

Usage examples (strict contract required):
  python main.py path/to/data.csv --engine orb --contract MES --outdir ./results

This script performs:
- load config (YAML)
- merge CLI overrides
- require --contract (validated against config.contracts)
- load CSV, parse datetimes, derive data_start/data_end if --start/--end omitted
- apply session filter and resample
- instantiate requested engine strategy and Backtester (skeleton)
- save merged config to outputs/config_used.yaml
- run backtester.run() (stubbed)
"""
from pathlib import Path
import sys
import yaml
import json
import typer
from typing import Optional
from core import utils
from core.backtester import Backtester
from engines import orb_engine, pullback_engine, vwap_engine, scalping_engine

app = typer.Typer(add_completion=False)


@app.command()
def main(
    data: str = typer.Argument(..., help="Path to CSV file (date,time,open,high,low,close,volume)"),
    engine: str = typer.Option(..., "--engine", "-e", help="Which engine to run: orb|pullback|vwap|scalping"),
    contract: str = typer.Option(..., "--contract", help="Contract symbol (required)"),
    config: str = typer.Option("config.yaml", "--config", help="Path to YAML config"),
    start: Optional[str] = typer.Option(None, help="YYYY-MM-DD (optional; derived from data if omitted)"),
    end: Optional[str] = typer.Option(None, help="YYYY-MM-DD (optional; derived from data if omitted)"),
    tz: Optional[str] = typer.Option(None, help="Timezone (overrides config)"),
    session_start: Optional[str] = typer.Option(None, help="Session start HH:MM (overrides config)"),
    session_end: Optional[str] = typer.Option(None, help="Session end HH:MM (overrides config)"),
    resample: Optional[int] = typer.Option(None, help="Resample minutes (overrides config)"),
    commission: Optional[float] = typer.Option(None, help="Commission round-trip override (overrides config)"),
    slippage: Optional[float] = typer.Option(None, help="Slippage points override (overrides config)"),
    dollars_per_point: Optional[float] = typer.Option(None, help="Optional override for dollars_per_point"),
    count: Optional[int] = typer.Option(None, help="Number of contracts per trade (overrides config)"),
    outdir: str = typer.Option("./outputs", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    # Load and merge config
    cfg_path = Path(config)
    if not cfg_path.exists():
        typer.echo(f"Config file not found at {cfg_path}, using built-in defaults.")
        cfg = utils.get_builtin_defaults()
    else:
        cfg = utils.load_yaml(cfg_path)

    merged = utils.merge_overrides(cfg, {
        "tz": tz,
        "session_start": session_start,
        "session_end": session_end,
        "resample_minutes": resample,
        "commission_roundtrip": commission,
        "slippage_points": slippage,
        # dollars_per_point normally from contract metadata; allow explicit override
        "dollars_per_point_override": dollars_per_point,
        "position_contracts_override": count,
        "outdir": outdir,
    })

    # Enforce contract requirement (strict)
    if not contract:
        # This should not happen because Typer requires it, but double-check
        supported = list(merged.get("contracts", {}).keys())
        msg = utils.missing_contract_error(supported)
        typer.secho(msg, fg=typer.colors.RED)
        raise typer.Abort()

    # Validate contract exists in config.contracts
    if not utils.validate_contract_symbol(contract, merged):
        supported = list(merged.get("contracts", {}).keys())
        msg = utils.invalid_contract_error(contract, supported)
        typer.secho(msg, fg=typer.colors.RED)
        raise typer.Abort()

    # Save chosen contract metadata into merged config
    merged["resolved_contract"] = contract
    merged["contract_meta"] = merged["contracts"][contract]

    # Load data and parse datetimes
    data_path = Path(data)
    if not data_path.exists():
        typer.echo(f"Data file not found: {data_path}")
        raise typer.Exit(code=2)

    df = utils.load_csv_parse_datetime(data_path, tz_target=merged.get("timezone") or merged.get("tz"))
    data_start, data_end = utils.derive_data_range(df)

    # Determine effective start/end
    eff_start, eff_end = utils.resolve_effective_date_range(data_start, data_end, start, end)
    if eff_start is None:
        typer.secho("No valid effective date range determined. Aborting.", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    # Filter to effective date range (do NOT pre-filter to session; strategy will handle session windows)
    df_filtered = utils.filter_date_range(df, eff_start, eff_end)

    # Resample full-day filtered data; strategy._day_groups will filter to session start/end internally
    resample_minutes = merged.get("resample_minutes", 3)
    df_resampled = utils.resample_to_n_minutes(df_filtered, resample_minutes)        

    # # Filter to effective date range and session times
    # df_filtered = utils.filter_date_range(df, eff_start, eff_end)
    # df_session = utils.session_filter(df_filtered, merged.get("session_start"), merged.get("session_end"))

    # # Resample if requested/ configured
    # resample_minutes = merged.get("resample_minutes", 3)
    # df_resampled = utils.resample_to_n_minutes(df_session, resample_minutes)

    # Prepare output dir
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    # Save merged config used
    merged_snapshot_path = outdir_path / "config_used.yaml"
    utils.save_yaml(merged, merged_snapshot_path)

    # Print a run header
    typer.echo("=== Backtest run summary ===")
    typer.echo(f"Data file: {data_path}")
    typer.echo(f"CSV-derived range: {data_start.date()} -> {data_end.date()}")
    typer.echo(f"Effective range: {eff_start} -> {eff_end}")
    typer.echo(f"Contract: {contract} (dollars_per_point: {merged['contract_meta']['dollars_per_point']}, tick_size: {merged['contract_meta'].get('tick_size')})")
    typer.echo(f"Engine: {engine}")
    typer.echo(f"Resample minutes: {resample_minutes}")
    typer.echo(f"Outputs: {outdir_path}")

    # Select strategy
    engine = engine.lower()
    if engine == "orb":
        strategy = orb_engine.ORBStrategy(merged.get("orb", {}))
    elif engine == "pullback":
        strategy = pullback_engine.PullbackStrategy(merged.get("pullback", {}))
    elif engine == "vwap":
        strategy = vwap_engine.VWAPFadeStrategy(merged.get("vwap", {}))
    elif engine == "scalping":
        strategy = scalping_engine.MESScalpingStrategy(merged.get("scalping", {}))
    else:
        typer.secho(f"Unknown engine: {engine}. Supported: orb, pullback, vwap, scalping", fg=typer.colors.RED)
        raise typer.Abort()

    # Initialize backtester (skeleton)
    bt = Backtester(strategy=strategy, bars_df=df_resampled, config=merged, outdir=outdir_path)
    bt.run()

    # Compute metrics and print compact run summary (trades & net P&L)
    metrics = bt.compute_metrics()
    total_trades = metrics.get("total_trades", 0)
    net_pnl = metrics.get("net_pnl", 0.0)
    gross_pnl = metrics.get("gross_pnl", 0.0)
    if total_trades == 0:
        typer.echo(f"Trades taken: 0 — no trades generated. No outputs to inspect.")
    else:
        typer.echo(f"Trades taken: {total_trades} | Net P&L: ${net_pnl:,.2f} | Gross P&L: ${gross_pnl:,.2f}")

    # Save results as before
    bt.save_results()
    typer.echo("Backtest completed. See outputs for files.")


if __name__ == "__main__":
    app()