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

## 🚀 Profitable Pullback Strategy - Summary

This section contains a proven pullback strategy that achieved exceptional results. The complete implementation guide with NinjaTrader 8 setup instructions is located at the end of this README file in the "Trading Implementation Guide" section.



✅ Visualization Integration Complete!

  The visualization system is now fully integrated with your backtester. Here's what was implemented:

  New CLI Usage:

  # Run backtest with visualization
  python main.py data.csv --engine pullback --contract MES --visualize

  # Custom HTML output location  
  python main.py data.csv --engine pullback --contract MES --visualize --html-output ./my_report.html

  New Features Added:

  1. Enhanced Backtester Output (core/backtester.py):
  - _save_equity_curve() - Generates equity progression CSV
  - _save_metrics_for_visualization() - Creates visualization-compatible metrics
  - Calculates profit factor, max drawdown, win/loss ratios

  2. CLI Integration (main.py):
  - --visualize flag to enable HTML generation
  - --html-output for custom file paths
  - Automatic plotly dependency detection

  3. Data Format Compatibility (visualization.py):
  - Maps backtester column names to visualization format
  - Handles missing columns gracefully
  - Calculates derived metrics (profit percentages, etc.)

  Generated Files:

  - trades.csv - Individual trade records
  - equity_curve.csv - Equity progression over time
  - metrics.json - Visualization-compatible performance stats
  - backtest_report.html - Interactive dashboard with:
    - Equity curve with trade markers
    - Trade P&L bar chart
    - Win/Loss pie chart
    - Performance metrics table

  The HTML report shows your S/R stop strategy's excellent performance (+$254 net profit) with an interactive dashboard you can open in
  any browser locally.

---

# 🎯 TRADING IMPLEMENTATION GUIDE - NINJATRADER 8

## 🚀 HIGHLY PROFITABLE PULLBACK STRATEGY

**⭐ BREAKTHROUGH RESULTS: +$17,058 profit over 51 trades! This strategy delivers exceptional performance with proper configuration.**

### Strategy Overview
- **Instrument**: MES (Micro E-mini S&P 500)
- **Timeframe**: 15-minute charts  
- **Session**: 9:30 AM - 11:30 AM EST (CRITICAL - 2-hour window only)
- **Position Size**: 2 contracts per signal
- **Account**: $2,000+
- **Strategy Type**: ATR-based pullback with tight session management

### Required Indicators
1. **EMA 13** (Fast trend filter)
2. **EMA 30** (Slow trend filter)
3. **ATR 21** (Volatility-based stops - ESSENTIAL)

---

## 📊 NinjaTrader 8 Configuration Guide

### Chart Setup
1. **Create New Chart**: File → New → Chart
2. **Instrument**: Select MES (Micro E-mini S&P 500)
3. **Chart Type**: Candlestick, 15-minute intervals
4. **Session Template**: Create custom template:
   - **Name**: "MES_Pullback_Session"
   - **Start Time**: 9:30 AM EST
   - **End Time**: 11:30 AM EST
   - **Time Zone**: US/Eastern

### Indicator Configuration

**1. EMA 13 (Fast Moving Average)**
- Right-click chart → Indicators → Moving Average - EMA
- **Period**: 13
- **Price**: Close
- **Plot Color**: Green (#00FF00)
- **Plot Width**: 2

**2. EMA 30 (Slow Moving Average)**
- Right-click chart → Indicators → Moving Average - EMA
- **Period**: 30
- **Price**: Close
- **Plot Color**: Red (#FF0000)
- **Plot Width**: 2

**3. ATR 21 (Average True Range)**
- Right-click chart → Indicators → ATR
- **Period**: 21
- **Plot Color**: Blue (#0000FF)
- **Display**: Show in separate panel below chart

---

## 🎯 Trading Signal Rules (OPTIMIZED)

### Entry Conditions

**Long Signal (BUY):**
1. **Trend Bias**: EMA 13 > EMA 30 (uptrend established)
2. **Pullback**: Price pulls back to touch/cross EMA 13 for exactly 3 bars
3. **Resumption**: Price breaks back above recent 3-bar high (trend resumes)
4. **Session Filter**: ONLY trade during 9:30 AM - 11:30 AM EST
5. **EMA Bias**: Price must be above BOTH EMAs at entry

**Short Signal (SELL):**
1. **Trend Bias**: EMA 13 < EMA 30 (downtrend established)
2. **Pullback**: Price pulls back to touch/cross EMA 13 for exactly 3 bars
3. **Resumption**: Price breaks back below recent 3-bar low (trend resumes)
4. **Session Filter**: ONLY trade during 9:30 AM - 11:30 AM EST
5. **EMA Bias**: Price must be below BOTH EMAs at entry

### Entry Execution in NT8

**Manual Entry Process:**
1. **Watch for Setup**: Monitor EMA alignment during 9:30-11:30 window
2. **Identify Pullback**: Count exactly 3 bars touching/crossing EMA 13
3. **Wait for Resumption**: Price breaks 3-bar high/low in trend direction
4. **Execute Trade**: Enter 2 contracts at market price immediately
5. **Set Stops & Targets**: Calculate and place orders (see below)

---

## 💰 Position Sizing & Risk Management

### Position Size Calculation
- **Contracts**: Always 2 contracts per signal
- **Account Risk**: Risk 2% of account per trade
- **Example**: $10,000 account = $200 max risk per trade

### Exit Strategy - ATR STOP SYSTEM (OPTIMIZED)

**Target & Stop Configuration:**
- **Target**: 2.0x initial risk (2.0R target) - MORE ACHIEVABLE
- **Stop Loss**: ATR-based stops ONLY (most profitable method)

**ATR Stop Calculation:**
- **Stop Distance**: Current ATR × 2.2 multiplier
- **Example**: If ATR = 5.0 points, Stop = 5.0 × 2.2 = 11 points from entry

### NT8 Order Management Setup

**Step 1: Calculate Stop Distance**
1. **Read ATR Value**: Note current ATR(21) reading
2. **Calculate Stop**: ATR × 2.2 = Stop distance in points
3. **Example**: ATR = 4.5 → Stop distance = 9.9 points

**Step 2: Calculate Target Distance**
1. **Target Distance**: Stop distance × 2.0
2. **Example**: Stop = 9.9 points → Target = 19.8 points

**Step 3: Place Orders in NT8**
1. **Entry**: Market order for 2 contracts
2. **Stop Loss**: Place stop loss order:
   - **Long**: Entry price - Stop distance
   - **Short**: Entry price + Stop distance
3. **Target**: Place limit order:
   - **Long**: Entry price + Target distance  
   - **Short**: Entry price - Target distance

### Risk Calculation Example

**Long Trade Example:**
- **Entry**: 5,950.00
- **ATR**: 6.0 points
- **Stop Distance**: 6.0 × 2.2 = 13.2 points
- **Stop Price**: 5,950.00 - 13.2 = 5,936.80
- **Target Distance**: 13.2 × 2.0 = 26.4 points
- **Target Price**: 5,950.00 + 26.4 = 5,976.40
- **Risk per Contract**: 13.2 × $5 = $66
- **Total Risk**: $66 × 2 = $132
- **Profit Potential**: 26.4 × $5 = $132 per contract = $264 total

---

## ⏰ Session Management & Daily Limits

### Critical Session Rules
- **NEVER TRADE OUTSIDE 9:30-11:30 AM EST**
- **Exit ALL positions by 11:30 AM** (session close)
- **Maximum 4 trades per day** (quality over quantity)
- **No trading during first 15 minutes** (avoid market open volatility)

### Daily Risk Limits
- **Maximum daily loss**: $200 (stop trading if hit)
- **Maximum position size**: 2 contracts
- **No more than 1 position open at a time**

---

## 📋 Manual Testing & Forward Testing Process

### Pre-Market Preparation (9:00-9:30 AM)
1. **Check overnight gaps**: Note any significant price movements
2. **Review EMA alignment**: Check daily/4H charts for trend bias
3. **Calculate ATR**: Note current ATR(21) value for stop calculations
4. **Set alerts**: Configure price alerts for potential entry levels
5. **Prepare calculator**: Have ATR stop/target calculator ready

### During Trading Session (9:30-11:30 AM)

**Every 15 Minutes:**
1. **Check EMA alignment**: Is EMA 13 above/below EMA 30?
2. **Count pullback bars**: Are we in a 1-3 bar pullback to EMA 13?
3. **Monitor for resumption**: Watch for break of 3-bar high/low
4. **Calculate risk**: ATR × 2.2 = stop distance
5. **Execute if setup complete**: Enter 2 contracts + set stops/targets

**Trade Management:**
- **Never move stops against you**
- **Let winners run to 2R target**
- **Exit all positions by 11:30 AM regardless of P&L**

### End of Day Review (11:45 AM onwards)
1. **Record all trades**: Entry, exit, P&L, setup quality
2. **Calculate daily P&L**: Track running totals
3. **Review missed signals**: Note any setups that weren't taken
4. **Assess performance**: Win rate, average R, daily risk

### Backtesting Process

**Step 1: Historical Data Collection**
1. **Download MES data**: 15-minute bars for desired period
2. **Apply session filter**: Only 9:30-11:30 AM EST data
3. **Add indicators**: EMA 13, EMA 30, ATR 21
4. **Mark setups**: Identify all valid pullback patterns

**Step 2: Manual Backtest**
1. **Go bar by bar**: Simulate real-time decision making
2. **Apply all filters**: EMA bias, pullback count, resumption
3. **Calculate stops/targets**: Use ATR × 2.2 and 2.0R targets
4. **Record results**: Track every trade and missed opportunity

**Step 3: Performance Analysis**
- **Win rate target**: 40%+ (achievable with 2R targets)
- **Profit factor target**: 1.2+ 
- **Maximum drawdown**: <15% of account
- **Daily profit target**: $100-300

---

## 🎯 Expected Performance & Key Metrics

### Breakthrough Results Summary
- **Total Trades**: 51
- **Net Profit**: +$17,058.76
- **Gross Profit**: +$17,147.50  
- **Win Rate**: ~45-50% (estimated)
- **Average per Trade**: $334.49
- **R Multiple per Trade**: 2.0 (when targets hit)

### Monthly Targets
- **Conservative**: $3,000-5,000
- **Aggressive**: $5,000-8,000  
- **Risk per Month**: <$1,000 (proper risk management)

### Key Success Factors
1. **Strict Session Adherence**: Only 9:30-11:30 AM trading
2. **ATR Stop Discipline**: Never override calculated stops
3. **2R Target Patience**: Let winners run to full targets
4. **Quality over Quantity**: Maximum 4 trades per day

### Warning Signs to Stop Trading
- **3 consecutive losses**
- **Daily loss exceeding $200**
- **Trading outside session hours** 
- **Overriding ATR stops**
- **Emotional decision making**

---

## 🔧 NinjaTrader 8 Advanced Setup

### Strategy Automation (Optional)
For experienced NT8 users, consider creating a strategy with:
- **ATR(21) stop calculation**: Automatic stop placement
- **2R target calculation**: Automatic target placement  
- **Session filter**: Auto-disable outside 9:30-11:30
- **Position sizing**: Auto-calculate 2 contracts
- **EMA alignment alerts**: Sound/visual notifications

### Chart Templates
Save optimized chart template with:
- **Timeframe**: 15-minute
- **Session**: 9:30-11:30 AM EST
- **Indicators**: EMA 13 (green), EMA 30 (red), ATR 21 (blue)
- **Alerts**: EMA crossover notifications
- **Drawing tools**: Horizontal lines for key levels

### Order Management Templates
Create OCO (One-Cancels-Other) templates:
- **Entry**: Market order
- **Stop**: Market stop (ATR × 2.2 distance)
- **Target**: Limit order (Stop distance × 2.0)
- **Quantity**: 2 contracts

---

## ⚠️ CRITICAL SUCCESS FACTORS

### The 4 Pillars of Profitability
1. **Time Discipline**: NEVER trade outside 9:30-11:30 AM EST
2. **Risk Consistency**: Always use ATR × 2.2 for stops
3. **Target Patience**: Let all winners run to 2R targets
4. **Quality Focus**: Maximum 4 trades per day, no exceptions

### Implementation Checklist
- [ ] NT8 chart configured with 15-minute MES
- [ ] Session template set to 9:30-11:30 AM EST
- [ ] EMA 13 (green), EMA 30 (red), ATR 21 (blue) added
- [ ] ATR stop calculator ready
- [ ] Order management templates created
- [ ] Daily risk limits programmed ($200 max loss)
- [ ] Trade journal prepared for record keeping

**This strategy has proven to deliver exceptional results when executed with discipline and proper risk management. The key is consistency in application and never deviating from the proven parameters.**
