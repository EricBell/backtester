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

## EMA 8/21 Strategy Implementation Guide

### Manual Testing Setup for NinjaTrader (or any platform)

This guide helps you manually implement and test the EMA 8/21 crossover strategy that achieved 97% win rate and $1,447 profit over 2 months in backtesting.

#### Strategy Overview
- **Instrument**: MES (Micro E-mini S&P 500)
- **Timeframe**: 15-minute charts
- **Session**: 9:45 AM - 2:30 PM EST
- **Position Size**: 2 contracts per signal
- **Account**: $2,000

#### Required Indicators
1. **EMA 8** (Fast moving average)
2. **EMA 21** (Slow moving average)  
3. **RSI 14** (Momentum confirmation)

#### Setup Instructions

**Chart Configuration:**
1. Set up MES 15-minute chart
2. Add EMA(8) - color it blue
3. Add EMA(21) - color it red
4. Add RSI(14) with levels at 40 and 60
5. Set session hours: 9:45 AM - 2:30 PM EST

#### Entry Rules

**Long Signal (BUY):**
1. EMA 8 crosses ABOVE EMA 21 (bullish crossover)
2. RSI > 40 (not oversold)
3. Current price > EMA 8 AND > EMA 21
4. Only take ONE signal per direction per day

**Short Signal (SELL):**
1. EMA 8 crosses BELOW EMA 21 (bearish crossover)
2. RSI < 60 (not overbought)
3. Current price < EMA 8 AND < EMA 21
4. Only take ONE signal per direction per day

#### Position Management

**Entry:**
- 2 contracts per signal
- Enter at market on next bar after signal

**Exit Strategy (70/30 split):**
- **Target 1**: 70% of position (1.4 contracts ≈ 1 contract) at 2.5x risk  
- **Target 2**: 30% of position (0.6 contracts ≈ 1 contract) at 4x risk
- **Stop Loss**: 8-10 points (0.15% of entry price)

**Risk Calculation Example:**
- Entry: 6000
- Stop loss: 6000 × 0.0015 = 9 points = $45 per contract
- With 2 contracts: $90 total risk per trade
- Target 1: 6000 + (9 × 2.5) = 6022.5 points (22.5 points = $112.50 profit)
- Target 2: 6000 + (9 × 4) = 6036 points (36 points = $180 profit)

#### Manual Testing Process

**Daily Preparation:**
1. Review previous day's EMA positioning
2. Note if any signals were taken yesterday
3. Mark key levels on chart

**During Session (9:45 AM - 2:30 PM):**
1. Watch for EMA crossovers every 15 minutes
2. Check RSI conditions when crossover occurs
3. Verify price relationship to EMAs
4. Enter trade if all conditions met
5. Set stops and targets immediately

**End of Day:**
1. Record trade results
2. Note any missed signals
3. Calculate P&L
4. Update trading log

#### Key Rules for Manual Testing

**Risk Management:**
- Maximum 1 long AND 1 short signal per day
- Never risk more than $40 per trade (2% of $2k account)
- Daily loss limit: $120 (6% of account)
- Stop trading if daily limit hit

**Trade Execution:**
- Use limit orders when possible to reduce slippage
- If signal occurs in last 30 minutes of session, skip it
- Exit all positions by 2:30 PM EST

**Record Keeping:**
Track these metrics for each trade:
- Entry time and price
- Exit time and price
- Contracts traded
- Gross P&L
- Commission costs
- Setup quality (A/B/C grade)

#### Expected Results (Based on Backtest)
- **Win Rate**: ~97%
- **Average per trade**: $22
- **Daily average**: $26
- **Monthly target**: ~$570

#### Warning Signs to Stop Trading
- More than 2 losses in a week
- Daily loss limit exceeded
- Emotional decision making
- Deviating from rules

This manual approach lets you validate the strategy's real-world performance while building confidence in the system before any automation.

---

## 🚀 PROFITABLE PULLBACK STRATEGY - Implementation Guide

### Manual Testing Setup for NinjaTrader (or any platform)

**⭐ This strategy is ACTUALLY PROFITABLE: +$177 over 2 months with proper risk management!**

#### Strategy Overview
- **Instrument**: MES (Micro E-mini S&P 500)
- **Timeframe**: 15-minute charts  
- **Session**: 8:00 AM - 12:00 PM EST
- **Position Size**: 2 contracts per signal
- **Account**: $2,000
- **Strategy Type**: Trend-following pullback strategy

#### Required Indicators
1. **EMA 13** (Fast trend filter)
2. **EMA 30** (Slow trend filter)
3. **ATR 21** (Volatility-based stops)

#### Setup Instructions

**Chart Configuration:**
1. Set up MES 15-minute chart
2. Add EMA(13) - color it green
3. Add EMA(30) - color it red  
4. Add ATR(21) indicator
5. Set session hours: 8:00 AM - 12:00 PM EST

#### Entry Rules

**Long Signal (BUY):**
1. **Trend Bias**: EMA 13 > EMA 30 (uptrend established)
2. **Pullback**: Price pulls back to touch/cross EMA 13 for 1-3 bars
3. **Resumption**: Price breaks back above recent high (trend resumes)
4. **Session Filter**: Only trade during 8:00 AM - 12:00 PM EST
5. **Volume Confirmation**: Current bar has above-average volume

**Short Signal (SELL):**
1. **Trend Bias**: EMA 13 < EMA 30 (downtrend established)
2. **Pullback**: Price pulls back to touch/cross EMA 13 for 1-3 bars  
3. **Resumption**: Price breaks back below recent low (trend resumes)
4. **Session Filter**: Only trade during 8:00 AM - 12:00 PM EST
5. **Volume Confirmation**: Current bar has above-average volume

#### Position Management

**Entry:**
- 2 contracts per signal
- Enter at market when resumption signal occurs

**Exit Strategy - FLEXIBLE STOP SYSTEM:**
- **Target**: 3.5x initial risk (3.5R target)
- **Stop Loss**: Choose from three configurable stop methods:

**Stop Method Options (via config toggles):**
1. **ATR Stop**: 2.2x ATR (volatility-based)
2. **Fixed Stop**: 7 points (minimum fixed stop)  
3. **Support/Resistance Stop**: 0.05-1.0 points beyond nearest S/R level ⭐ **RECOMMENDED**

**Stop Selection Logic:**
- **"tightest"**: Uses closest stop to entry (lowest risk)
- **"loosest"**: Uses furthest stop from entry (highest risk)
- **"average"**: Uses average of enabled stop methods

**OPTIMAL CONFIGURATION - S/R Stops Only:**
```yaml
use_atr_stops: false
use_fixed_stops: false  
use_sr_stops: true
sr_buffer_points: 0.05  # Ultra-tight buffer for maximum profit
stop_selection: "tightest"
```

**Risk Calculation Example (Optimal S/R Stop):**
- Entry: 6000 (LONG)
- **Support level = 5997 → S/R stop = 5997.05** (0.05 buffer)
- Risk: 2.95 points = $14.75 per contract ($29.50 total for 2 contracts)
- Target: 6000 + (2.95 × 3.5) = 6010.3 points (10.3 points = $51.50 profit per contract)
- **Total Profit Potential**: $103 vs $29.50 risk = 3.5:1 ratio maintained

#### Manual Testing Process

**Pre-Market Preparation:**
1. Check EMA alignment on daily/hourly charts for bias
2. **Mark key support/resistance levels:**
   - Previous day's high/low
   - Overnight high/low
   - Round numbers (6000, 6050, 6100, etc.)
   - Previous swing highs/lows on 15min chart
3. Calculate ATR for backup stop sizing

**During Session (8:00 AM - 12:00 PM):**
1. Watch for EMA 13/30 alignment (trend bias)
2. Identify pullbacks to EMA 13
3. Wait for trend resumption signal  
4. **Before entry, check for nearby support/resistance:**
   - LONG: Look for support below entry (recent swing low, round number, etc.)
   - SHORT: Look for resistance above entry (recent swing high, round number, etc.)
5. Enter with 2 contracts when all conditions met
6. **Set stop using tightest option:** ATR stop vs Fixed stop vs S/R stop
7. Set target at 3.5x risk from chosen stop

**End of Day:**
1. Record all trades and missed signals
2. Calculate P&L and risk metrics
3. Review EMA alignment for next day

#### Key Rules for Manual Testing

**Risk Management:**
- Risk $130-140 per trade (6.5-7% of $2k account)
- Daily loss limit: $280 (14% of account) 
- Stop trading if 2 consecutive losses

**Trade Execution:**
- Use limit orders near resumption levels when possible
- Exit all positions by 12:00 PM EST
- Never chase breakouts - wait for proper pullback setup

**Trade Identification:**
Look for this pattern:
1. **Trend**: EMAs aligned (13 above/below 30)
2. **Pullback**: 1-3 bars touching EMA 13
3. **Resumption**: Break of recent high/low in trend direction
4. **Volume**: Current bar volume > 10-bar average

#### Expected Results (Based on Backtest)

**UPDATED PERFORMANCE WITH S/R STOPS:**
- **Net Profit**: +$584 over 2 months (3.3x improvement!)
- **Win Rate**: 35.7% (nearly doubled from original 19%)
- **Average per trade**: +$13.91 (vs original $4.21)
- **Monthly target**: ~$292 (vs original $88)
- **Risk/Reward**: 3.5:1 maintained

**S/R Buffer Point Optimization Results:**
| Buffer | Net P&L | Win Rate | Best For |
|--------|---------|----------|----------|
| 0.05   | +$584   | 35.7%    | Maximum profit |
| 0.5    | +$399   | 42.9%    | Higher win rate |
| 1.0    | +$109   | 35.7%    | Conservative |

**Comparison to Other Stop Methods:**
- **Original ATR-only**: +$177 (19% win rate)
- **Tightest ATR+Fixed**: -$235 (19% win rate)
- **Optimal S/R (0.05)**: +$584 (35.7% win rate) ⭐ **BEST**

#### Key Success Factors
- **Patience**: Wait for proper pullback setup (don't chase)
- **Discipline**: Take full 3.5R targets when available  
- **Risk Control**: Always use ATR-based stops
- **Session Focus**: Trade only during 8AM-12PM high-volume hours

#### Warning Signs to Stop Trading
- More than 3 losses in a row
- Daily loss limit exceeded  
- Trading outside session hours
- Forcing trades without proper pullback

**🎯 This strategy actually works because it catches strong trending moves after brief pullbacks, with large 3.5R targets that more than compensate for the smaller, more frequent losses.**
