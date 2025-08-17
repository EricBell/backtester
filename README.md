# backtester
AI helped created this backtesting platform. It's currently aimed at future contract tests but can be used for other asset classes.


MES Intraday Backtester - Skeleton (v1)
=======================================

Overview
--------
This repository contains a skeleton for an intraday futures backtester focused on MES/MNQ/ES-style contracts.
It enforces an explicit contract argument for v1 and contains optional meta-file support (disabled by default).

Install requirements (suggested)
- python >= 3.9
- pip install pandas pyyaml typer matplotlib pytest

Quick usage (strict contract required)
-------------------------------------
Example:
  python main.py path/to/data.csv --engine orb --contract MES --outdir ./results

Important: --contract is required and must match a symbol defined under `contracts:` in your config.yaml.

Error messages (exact text)
---------------------------
Missing contract:
  ERROR: --contract is required and was not provided.
  Supported contract symbols: MES, MNQ, ES
  Example: --contract MES
  Please specify the contract symbol to ensure correct P&L scaling.

Invalid contract:
  ERROR: Unknown contract 'XYZ'. Supported contract symbols: MES, MNQ, ES
  Please add the contract to your config.yaml under 'contracts' or use a supported symbol.
  Example: --contract MES

Config
------
Default config path: ./config.yaml (examples/config.yaml provided).
Key items:
- contracts: mapping of contract symbol -> metadata (dollars_per_point, tick_size, label)
- require_contract: true (v1 enforces CLI)
- meta_file_support_enabled: false (disable/enable companion meta YAML support)

Files
-----
- main.py - CLI entrypoint
- core/ - utilities, indicators, backtester skeleton
- engines/ - orb, pullback, vwap strategy stubs
- examples/config.yaml - example configuration

Next steps
----------
1. Review the skeleton files above.
2. If OK, run incremental implementation: fill in generate_signals in engines and fill/execute logic in core/backtester.
3. Use tests/ and the README as guidance.

This is a safety-first skeleton — it will not execute trades. When you want me to implement further behavior (e.g., fills, partial exits, slippage modeling), say what you want next and I will produce the code accordingly.

Tests
1. Running config validator
   1. $  python -m pytest -q tests/test_config_validation.py
   
Debugging
Here are a few ways to get DEBUG_ORB=1 into your WSL environment:

1) For your current session only  
   In your shell just run:  
     export DEBUG_ORB=1  
   You can now run whatever program needs it in that same shell.

2) Permanently, for your user only  
   a) Open your ~/.bashrc (or ~/.profile) in an editor:  
      nano ~/.bashrc  
   b) Add this line at the end:  
      export DEBUG_ORB=1  
   c) Save & exit, then reload it:  
      source ~/.bashrc  
   From now on every new shell will have DEBUG_ORB=1 set.

3) System-wide, for all users  
   Edit /etc/environment (you’ll need sudo):  
     sudo nano /etc/environment  
   Add a line (no “export”):  
     DEBUG_ORB="1"  
   Save & reboot WSL (`wsl --shutdown` from Windows Cmd/PowerShell) or just relaunch your distro.

4) One-off on a single command  
   You can also prefix any command without polluting your shell:  
     DEBUG_ORB=1 ./your-app --some-arg  

Optimizations
● Strategies to Improve Net P&L:

  1. Reduce Transaction Costs
  - Lower trade frequency: Tighter filters to trade only highest-probability
  setups
  - Reduce slippage: Use limit orders instead of market orders (reduce 0.25
  slippage)
  - Better timing: Trade during higher liquidity periods

  2. Improve Win Rate & Profit Factor
  - Tighter entry filters: Add volume confirmation, ATR filters, or time-of-day
  restrictions
  - Better exits: Trailing stops, dynamic targets based on volatility
  - Risk/reward optimization: Current [2.2, 3.2] targets might be too aggressive

  3. Test Alternative Strategies
  - VWAP strategy: Mean reversion might have higher win rate
  - Scalping strategy: Quick 1-2 tick profits with lower transaction cost ratio
  - Pullback strategy: Trend-following might capture larger moves

  4. Parameter Optimization
  - Session timing: Focus on 9:30-11:00 AM high-volume period
  - Volume filters: Only trade on above-average volume bars
  - ATR-based position sizing: Scale position size with volatility

  ● 3-Contract Tiered Exit Logic:

  Target 1 (first profit level):
  - exit_contracts = max(1, math.floor(remaining_contracts / 
  2))
  - exit_contracts = max(1, math.floor(3 / 2)) = max(1, 1) = 1
  - 1 contract exits at Target 1

  Target 2 (second profit level):
  - remaining_contracts = 3 - 1 = 2
  - 2 contracts remain for Target 2 (or stop loss)

  Current ORB targets from config:
  take_profit_r: [2.2, 3.2]  # Target 1 at 2.2R, Target 2 at 
  3.2R

  So for 3 contracts:
  - 1 contract exits at 2.2R (Target 1) - taking quick profit
  - 2 contracts hold for 3.2R (Target 2) - seeking larger
  profit
  - If Target 2 not hit, remaining 2 contracts exit at stop
  loss

  This 1:2 allocation gives you a balanced approach: secure
  some profit early while letting the majority ride for bigger
  gains.
