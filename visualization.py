import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
from pathlib import Path
from core.utils import load_csv_parse_datetime
from core.indicators import ema, vwap, atr, rsi

def create_html_report(backtest_results_dir, output_file=None, bar_data_path=None):
    """
    Create an HTML report with interactive charts
    
    Args:
        backtest_results_dir: Directory containing backtest results
        output_file: Output HTML file path
        bar_data_path: Path to bar data CSV file for detailed trading chart
    """
    # Set default output file if not provided
    if output_file is None:
        output_file = os.path.join(backtest_results_dir, 'backtest_report.html')
    
    # Load data
    trades_path = os.path.join(backtest_results_dir, 'trades.csv')
    equity_path = os.path.join(backtest_results_dir, 'equity_curve.csv')
    metrics_path = os.path.join(backtest_results_dir, 'metrics.json')
    config_path = os.path.join(backtest_results_dir, 'config_used.yaml')
    
    trades_df = pd.read_csv(trades_path) if os.path.exists(trades_path) else pd.DataFrame()
    equity_df = pd.read_csv(equity_path) if os.path.exists(equity_path) else pd.DataFrame()
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f) if os.path.exists(metrics_path) else {}
    
    # Load config for strategy parameters
    config = {}
    if os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    
    # Load bar data if path provided
    bars_df = None
    if bar_data_path and os.path.exists(bar_data_path):
        try:
            bars_df = load_csv_parse_datetime(
                Path(bar_data_path),
                tz_target=config.get('timezone', 'America/New_York')
            )
        except Exception as e:
            print(f"Warning: Could not load bar data from {bar_data_path}: {e}")
    
    # Convert timestamps to datetime and map column names
    if not trades_df.empty:
        # Map our column names to expected names
        if 'entry_timestamp' in trades_df.columns:
            trades_df['entry_time'] = trades_df['entry_timestamp']
        if 'exit_timestamp' in trades_df.columns:
            trades_df['exit_time'] = trades_df['exit_timestamp']
        if 'net_pnl' in trades_df.columns:
            trades_df['profit_loss'] = trades_df['net_pnl']
        
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        
        # Calculate profit_loss_pct if not present
        if 'profit_loss_pct' not in trades_df.columns:
            trades_df['profit_loss_pct'] = (trades_df['profit_loss'] / (trades_df['entry_price'] * trades_df['contracts'] * 5.0)) * 100
        
        # Add missing columns for visualization compatibility
        if 'stop_loss' not in trades_df.columns:
            trades_df['stop_loss'] = 0.0  # Placeholder
        if 'take_profit' not in trades_df.columns:
            trades_df['take_profit'] = 0.0  # Placeholder
    
    if not equity_df.empty:
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
    
    # Create enhanced dashboard with trading chart
    fig = create_enhanced_dashboard(trades_df, equity_df, metrics, bars_df, config)
    
    # Write to HTML file
    html_string = fig.to_html(include_plotlyjs=True, full_html=True)
    with open(output_file, 'w') as f:
        f.write(html_string)
    
    print(f"Enhanced HTML report created: {output_file}")
    return output_file

def create_dashboard(trades_df, equity_df, metrics):
    """Create an interactive dashboard with Plotly"""
    # Create subplot structure
    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"colspan": 2}, None],
            [{"colspan": 2}, None],
            [{"type": "pie"}, {"type": "table"}]
        ],
        row_heights=[0.4, 0.3, 0.3],
        subplot_titles=("Equity Curve", "Trade P&L", "Win/Loss Ratio", "Trade Details")
    )
    
    # Add equity curve
    if not equity_df.empty:
        fig.add_trace(
            go.Scatter(
                x=equity_df['timestamp'],
                y=equity_df['equity'],
                mode='lines',
                name='Equity',
                line=dict(color='rgba(0, 128, 255, 0.8)', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 128, 255, 0.2)'
            ),
            row=1, col=1
        )
        
        # Add position markers
        long_entries = equity_df[equity_df['position'] > 0]
        short_entries = equity_df[equity_df['position'] < 0]
        trade_exits = equity_df[equity_df['trade_pl'] != 0]
        
        # Long entry markers
        if not long_entries.empty:
            fig.add_trace(
                go.Scatter(
                    x=long_entries['timestamp'],
                    y=long_entries['equity'],
                    mode='markers',
                    name='Long Entry',
                    marker=dict(color='green', size=10, symbol='triangle-up')
                ),
                row=1, col=1
            )
        
        # Short entry markers
        if not short_entries.empty:
            fig.add_trace(
                go.Scatter(
                    x=short_entries['timestamp'],
                    y=short_entries['equity'],
                    mode='markers',
                    name='Short Entry',
                    marker=dict(color='red', size=10, symbol='triangle-down')
                ),
                row=1, col=1
            )
        
        # Trade exit markers
        if not trade_exits.empty:
            # Color based on profit/loss
            colors = ['green' if pl > 0 else 'red' for pl in trade_exits['trade_pl']]
            fig.add_trace(
                go.Scatter(
                    x=trade_exits['timestamp'],
                    y=trade_exits['equity'],
                    mode='markers',
                    name='Trade Exit',
                    marker=dict(color=colors, size=8, symbol='circle')
                ),
                row=1, col=1
            )
    
    # Add trade P&L chart
    if not trades_df.empty:
        colors = ['green' if pl > 0 else 'red' for pl in trades_df['profit_loss']]
        fig.add_trace(
            go.Bar(
                x=trades_df['exit_time'],
                y=trades_df['profit_loss'],
                name='Trade P&L',
                marker_color=colors
            ),
            row=2, col=1
        )
    
    # Add win/loss pie chart
    if not trades_df.empty and 'winning_trades' in metrics and 'losing_trades' in metrics:
        winning = metrics['winning_trades']
        losing = metrics['losing_trades']
        
        fig.add_trace(
            go.Pie(
                labels=['Winning Trades', 'Losing Trades'],
                values=[winning, losing],
                marker_colors=['green', 'red'],
                textinfo='label+percent',
                hole=0.4
            ),
            row=3, col=1
        )
    
    # Add key metrics as a table
    metrics_table = []
    if metrics:
        metrics_table = [
            ["Total Trades", f"{metrics.get('total_trades', 0)}"],
            ["Win Rate", f"{metrics.get('win_rate', 0):.2%}"],
            ["Net Profit", f"${metrics.get('net_profit', 0):.2f}"],
            ["Net Profit %", f"{metrics.get('net_profit_pct', 0):.2f}%"],
            ["Profit Factor", f"{metrics.get('profit_factor', 0):.2f}"],
            ["Max Drawdown", f"{metrics.get('max_drawdown', 0):.2f}%"],
            ["Avg Win", f"${metrics.get('avg_win', 0):.2f}"],
            ["Avg Loss", f"${abs(metrics.get('avg_loss', 0)):.2f}"]
        ]
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Metric", "Value"],
                fill_color='rgb(30, 30, 30)',
                align='left',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=list(zip(*metrics_table)),
                fill_color='rgb(50, 50, 50)',
                align='left',
                font=dict(color='white', size=11)
            )
        ),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        title="MES Futures Trifecta Strategy Backtest Results",
        template="plotly_dark",
        height=900,
        width=1200,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_enhanced_dashboard(trades_df, equity_df, metrics, bars_df=None, config=None):
    """Create an enhanced dashboard with trading chart, equity curve, and metrics"""
    if bars_df is not None and not bars_df.empty:
        # Enhanced layout with trading chart
        fig = make_subplots(
            rows=5, cols=2,
            specs=[
                [{"colspan": 2}, None],           # Trading chart (main)
                [{"colspan": 2}, None],           # Volume chart
                [{"colspan": 2}, None],           # Indicator chart
                [{"colspan": 2}, None],           # Equity curve
                [{"type": "pie"}, {"type": "table"}]  # Pie + metrics table
            ],
            row_heights=[0.35, 0.15, 0.15, 0.25, 0.1],
            subplot_titles=("Price Chart with Trades", "Volume", "Indicators", "Equity Curve", "Win/Loss", "Metrics"),
            vertical_spacing=0.05
        )
        
        # Add detailed trading chart
        add_trading_chart(fig, bars_df, trades_df, config, row=1)
        
        # Add volume chart
        add_volume_chart(fig, bars_df, trades_df, row=2)
        
        # Add indicators chart
        add_indicators_chart(fig, bars_df, config, row=3)
        
        # Add equity curve
        add_equity_curve(fig, equity_df, row=4)
        
        # Add pie and table
        add_pie_and_metrics_table(fig, trades_df, metrics, row=5)
        
        # Enhanced layout for trading dashboard with better navigation
        fig.update_layout(
            title="Enhanced Trading Strategy Analysis",
            template="plotly_dark",
            height=1400,
            width=1400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            # Enhanced chart interactions
            dragmode='zoom',
            selectdirection='d'
        )
        
        # Add enhanced navigation and scaling for main price chart
        if bars_df is not None and not bars_df.empty:
            # Calculate smart Y-axis range for better price visibility
            price_range = bars_df['High'].max() - bars_df['Low'].min()
            padding = price_range * 0.05  # 5% padding for better visibility
            
            # Update main price chart (row 1) with enhanced navigation
            fig.update_xaxes(
                # Add range selector buttons for time navigation
                rangeselector=dict(
                    buttons=[
                        dict(count=1, label="1D", step="day", stepmode="backward"),
                        dict(count=3, label="3D", step="day", stepmode="backward"),
                        dict(count=7, label="7D", step="day", stepmode="backward"),
                        dict(step="all", label="All")
                    ],
                    bgcolor="rgba(50,50,50,0.8)",
                    font=dict(color="white")
                ),
                # Remove range slider for cleaner appearance
                rangeslider=dict(visible=False),
                # Enable zooming
                fixedrange=False,
                row=1, col=1
            )
            
            # Update Y-axis for main price chart with smart scaling
            fig.update_yaxes(
                # Set range with padding for better price action visibility
                range=[
                    bars_df['Low'].min() - padding,
                    bars_df['High'].max() + padding
                ],
                # Enable Y-axis zooming
                fixedrange=False,
                # Independent Y-axis scaling
                scaleanchor=None,
                # Add gridlines for better readability
                showgrid=True,
                gridcolor='rgba(128,128,128,0.3)',
                row=1, col=1
            )
            
            # Update other subplots for better scaling
            # Volume chart (row 2)
            fig.update_yaxes(
                fixedrange=False,
                showgrid=True,
                gridcolor='rgba(128,128,128,0.3)',
                row=2, col=1
            )
            
            # Indicators chart (row 3)  
            fig.update_yaxes(
                fixedrange=False,
                showgrid=True,
                gridcolor='rgba(128,128,128,0.3)',
                row=3, col=1
            )
            
            # Equity curve (row 4)
            fig.update_yaxes(
                fixedrange=False,
                showgrid=True,
                gridcolor='rgba(128,128,128,0.3)',
                row=4, col=1
            )
    else:
        # Fallback to original dashboard if no bars data
        fig = create_dashboard(trades_df, equity_df, metrics)
    
    return fig

def add_trading_chart(fig, bars_df, trades_df, config, row=1):
    """Add detailed OHLC chart with trade markers and indicators"""
    # Add candlestick chart with enhanced styling
    fig.add_trace(
        go.Candlestick(
            x=bars_df.index,
            open=bars_df['Open'],
            high=bars_df['High'],
            low=bars_df['Low'],
            close=bars_df['Close'],
            name='Price',
            increasing_line_color='#00ff88',                    # Brighter green
            decreasing_line_color='#ff4444',                    # Brighter red  
            increasing_fillcolor='rgba(0, 255, 136, 0.3)',     # Semi-transparent green
            decreasing_fillcolor='rgba(255, 68, 68, 0.3)',     # Semi-transparent red
            line=dict(width=1),                                 # Thinner lines for better visibility
            # Add hover info
            text=[f"Open: ${row['Open']:.2f}<br>High: ${row['High']:.2f}<br>Low: ${row['Low']:.2f}<br>Close: ${row['Close']:.2f}" 
                  for _, row in bars_df.iterrows()]
        ),
        row=row, col=1
    )
    
    # Add basic indicators (EMA if available)
    add_price_indicators(fig, bars_df, config, row)
    
    # Add trade entry/exit markers
    add_trade_markers(fig, trades_df, row)

def add_price_indicators(fig, bars_df, config, row):
    """Add price-based indicators like EMAs, VWAP to the price chart"""
    if config is None:
        return
        
    # Detect active strategy and add appropriate indicators
    if 'pullback' in config:
        # Pullback strategy: EMA 13 (green), EMA 30 (red)
        strategy_params = config['pullback']
        fast_ema = strategy_params.get('ema_fast', 13)
        slow_ema = strategy_params.get('ema_slow', 30)
        
        fast_ema_data = ema(bars_df['Close'], fast_ema)
        slow_ema_data = ema(bars_df['Close'], slow_ema)
        
        fig.add_trace(
            go.Scatter(
                x=bars_df.index,
                y=fast_ema_data,
                mode='lines',
                name=f"EMA {fast_ema}",
                line=dict(color='green', width=2)  # Green as per README
            ),
            row=row, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=bars_df.index,
                y=slow_ema_data,
                mode='lines',
                name=f"EMA {slow_ema}",
                line=dict(color='red', width=2)  # Red as per README
            ),
            row=row, col=1
        )
        
    elif 'ema8_21' in config:
        # EMA8/21 strategy: EMA 8 (blue), EMA 21 (red)
        strategy_params = config['ema8_21']
        fast_ema = strategy_params.get('fast_ema_period', 8)
        slow_ema = strategy_params.get('slow_ema_period', 21)
        
        fast_ema_data = ema(bars_df['Close'], fast_ema)
        slow_ema_data = ema(bars_df['Close'], slow_ema)
        
        fig.add_trace(
            go.Scatter(
                x=bars_df.index,
                y=fast_ema_data,
                mode='lines',
                name=f"EMA {fast_ema}",
                line=dict(color='blue', width=2)  # Blue as per README
            ),
            row=row, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=bars_df.index,
                y=slow_ema_data,
                mode='lines',
                name=f"EMA {slow_ema}",
                line=dict(color='red', width=2)  # Red as per README
            ),
            row=row, col=1
        )
        
    elif 'scalping' in config:
        # Scalping strategy: Multiple EMAs
        strategy_params = config['scalping']
        fast_ema = strategy_params.get('fast_ema', 8)
        medium_ema = strategy_params.get('medium_ema', 25)
        slow_ema = strategy_params.get('slow_ema', 50)
        
        fast_ema_data = ema(bars_df['Close'], fast_ema)
        medium_ema_data = ema(bars_df['Close'], medium_ema)
        slow_ema_data = ema(bars_df['Close'], slow_ema)
        
        fig.add_trace(
            go.Scatter(
                x=bars_df.index,
                y=fast_ema_data,
                mode='lines',
                name=f"EMA {fast_ema}",
                line=dict(color='blue', width=1)
            ),
            row=row, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=bars_df.index,
                y=medium_ema_data,
                mode='lines',
                name=f"EMA {medium_ema}",
                line=dict(color='orange', width=1)
            ),
            row=row, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=bars_df.index,
                y=slow_ema_data,
                mode='lines',
                name=f"EMA {slow_ema}",
                line=dict(color='purple', width=1)
            ),
            row=row, col=1
        )
        
    elif 'vwap' in config:
        # VWAP strategy: Show VWAP line only
        if 'Volume' in bars_df.columns and not bars_df['Volume'].isna().all():
            try:
                vwap_data = vwap(bars_df['High'], bars_df['Low'], bars_df['Close'], bars_df['Volume'])
                fig.add_trace(
                    go.Scatter(
                        x=bars_df.index,
                        y=vwap_data,
                        mode='lines',
                        name='VWAP',
                        line=dict(color='yellow', width=2, dash='dash')
                    ),
                    row=row, col=1
                )
            except Exception:
                pass  # Skip VWAP if calculation fails
    
    # ORB strategy doesn't use EMAs or VWAP - it uses breakout levels

def add_trade_markers(fig, trades_df, row):
    """Add trade entry and exit markers to the price chart"""
    if trades_df.empty:
        return
    
    # Entry markers
    long_entries = trades_df[trades_df['side'] == 'LONG']
    short_entries = trades_df[trades_df['side'] == 'SHORT']
    
    if not long_entries.empty:
        fig.add_trace(
            go.Scatter(
                x=long_entries['entry_time'],
                y=long_entries['entry_price'],
                mode='markers',
                name='Long Entry',
                marker=dict(
                    color='lime',
                    size=12,
                    symbol='triangle-up',
                    line=dict(width=2, color='darkgreen')
                ),
                text=[f"L {row['trade_id']}: ${row['entry_price']:.2f}" for _, row in long_entries.iterrows()],
                textposition="top center"
            ),
            row=row, col=1
        )
    
    if not short_entries.empty:
        fig.add_trace(
            go.Scatter(
                x=short_entries['entry_time'],
                y=short_entries['entry_price'],
                mode='markers',
                name='Short Entry',
                marker=dict(
                    color='red',
                    size=12,
                    symbol='triangle-down',
                    line=dict(width=2, color='darkred')
                ),
                text=[f"S {row['trade_id']}: ${row['entry_price']:.2f}" for _, row in short_entries.iterrows()],
                textposition="bottom center"
            ),
            row=row, col=1
        )
    
    # Exit markers with P&L color coding
    winning_trades = trades_df[trades_df['profit_loss'] > 0]
    losing_trades = trades_df[trades_df['profit_loss'] <= 0]
    
    if not winning_trades.empty:
        fig.add_trace(
            go.Scatter(
                x=winning_trades['exit_time'],
                y=winning_trades['exit_price'],
                mode='markers',
                name='Winning Exit',
                marker=dict(
                    color='green',
                    size=10,
                    symbol='square',
                    line=dict(width=2, color='darkgreen')
                ),
                text=[f"Exit {row['trade_id']}: +${row['profit_loss']:.2f}" for _, row in winning_trades.iterrows()],
                textposition="top center"
            ),
            row=row, col=1
        )
    
    if not losing_trades.empty:
        fig.add_trace(
            go.Scatter(
                x=losing_trades['exit_time'],
                y=losing_trades['exit_price'],
                mode='markers',
                name='Losing Exit',
                marker=dict(
                    color='red',
                    size=10,
                    symbol='square',
                    line=dict(width=2, color='darkred')
                ),
                text=[f"Exit {row['trade_id']}: ${row['profit_loss']:.2f}" for _, row in losing_trades.iterrows()],
                textposition="bottom center"
            ),
            row=row, col=1
        )

def add_volume_chart(fig, bars_df, trades_df, row=2):
    """Add volume chart with trade highlighting"""
    if 'Volume' not in bars_df.columns:
        return
    
    # Volume bars
    colors = ['green' if close > open else 'red' 
              for close, open in zip(bars_df['Close'], bars_df['Open'])]
    
    fig.add_trace(
        go.Bar(
            x=bars_df.index,
            y=bars_df['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ),
        row=row, col=1
    )
    
    # Highlight volume at trade entry times
    if not trades_df.empty:
        # Find volume at entry times by merging with bars data
        entry_volumes = []
        for _, trade in trades_df.iterrows():
            entry_time = trade['entry_time']
            # Find closest bar
            closest_bar = bars_df.index[bars_df.index <= entry_time]
            if len(closest_bar) > 0:
                volume = bars_df.loc[closest_bar[-1], 'Volume']
                entry_volumes.append((entry_time, volume))
        
        if entry_volumes:
            entry_times, volumes = zip(*entry_volumes)
            fig.add_trace(
                go.Scatter(
                    x=list(entry_times),
                    y=list(volumes),
                    mode='markers',
                    name='Trade Volume',
                    marker=dict(color='yellow', size=8, symbol='diamond')
                ),
                row=row, col=1
            )

def add_indicators_chart(fig, bars_df, config, row=3):
    """Add technical indicators based on strategy"""
    if config is None:
        return
    
    # Strategy-specific indicators
    if 'pullback' in config:
        # Pullback strategy: Show ATR for volatility context
        strategy_params = config['pullback']
        atr_period = strategy_params.get('atr_period', 21)
        
        try:
            # Calculate ATR
            high_low = bars_df['High'] - bars_df['Low']
            high_close = abs(bars_df['High'] - bars_df['Close'].shift())
            low_close = abs(bars_df['Low'] - bars_df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr_data = true_range.rolling(atr_period).mean()
            
            fig.add_trace(
                go.Scatter(
                    x=bars_df.index,
                    y=atr_data,
                    mode='lines',
                    name=f'ATR({atr_period})',
                    line=dict(color='orange', width=2)
                ),
                row=row, col=1
            )
        except Exception:
            pass
            
    elif 'ema8_21' in config:
        # EMA8/21 strategy: Show RSI
        strategy_params = config['ema8_21']
        rsi_period = strategy_params.get('rsi_period', 14)
        
        try:
            rsi_data = rsi(bars_df['Close'], rsi_period)
            fig.add_trace(
                go.Scatter(
                    x=bars_df.index,
                    y=rsi_data,
                    mode='lines',
                    name=f'RSI({rsi_period})',
                    line=dict(color='cyan', width=2)
                ),
                row=row, col=1
            )
            
            # Add RSI overbought/oversold lines
            overbought = strategy_params.get('rsi_overbought', 70)
            oversold = strategy_params.get('rsi_oversold', 30)
            
            fig.add_hline(
                y=overbought,
                line_dash="dash",
                line_color="red",
                row=row, col=1
            )
            fig.add_hline(
                y=oversold,
                line_dash="dash",
                line_color="green",
                row=row, col=1
            )
            fig.add_hline(
                y=50,
                line_dash="dot",
                line_color="gray",
                row=row, col=1
            )
        except Exception:
            pass
            
    elif 'scalping' in config:
        # Scalping strategy: Show RSI
        strategy_params = config['scalping']
        rsi_period = strategy_params.get('rsi_period', 14)
        
        try:
            rsi_data = rsi(bars_df['Close'], rsi_period)
            fig.add_trace(
                go.Scatter(
                    x=bars_df.index,
                    y=rsi_data,
                    mode='lines',
                    name=f'RSI({rsi_period})',
                    line=dict(color='cyan', width=2)
                ),
                row=row, col=1
            )
            
            # Add RSI overbought/oversold lines
            overbought = strategy_params.get('rsi_overbought', 75)
            oversold = strategy_params.get('rsi_oversold', 25)
            
            fig.add_hline(
                y=overbought,
                line_dash="dash",
                line_color="red",
                row=row, col=1
            )
            fig.add_hline(
                y=oversold,
                line_dash="dash",
                line_color="green",
                row=row, col=1
            )
        except Exception:
            pass
    
    # ORB and VWAP strategies don't use additional indicators in separate panel

def add_equity_curve(fig, equity_df, row=4):
    """Add equity curve to the dashboard"""
    if equity_df.empty:
        return
    
    fig.add_trace(
        go.Scatter(
            x=equity_df['timestamp'],
            y=equity_df['equity'],
            mode='lines',
            name='Equity',
            line=dict(color='rgba(0, 128, 255, 0.8)', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 128, 255, 0.2)'
        ),
        row=row, col=1
    )

def add_pie_and_metrics_table(fig, trades_df, metrics, row=5):
    """Add pie chart and metrics table"""
    # Win/loss pie chart
    if not trades_df.empty:
        wins = len(trades_df[trades_df['profit_loss'] > 0])
        losses = len(trades_df[trades_df['profit_loss'] <= 0])
        
        fig.add_trace(
            go.Pie(
                labels=['Wins', 'Losses'],
                values=[wins, losses],
                marker_colors=['green', 'red'],
                textinfo='label+percent',
                hole=0.4,
                name='Win/Loss'
            ),
            row=row, col=1
        )
    
    # Metrics table
    metrics_table = []
    if metrics:
        perf = metrics.get('performance', metrics)  # Handle both flat and nested structures
        
        # Handle different key names for net profit
        net_pnl = perf.get('net_pnl', perf.get('net_profit', 0))
        gross_pnl = perf.get('gross_pnl', perf.get('gross_profit', 0))
        wins = perf.get('wins', perf.get('winning_trades', 0))
        losses = perf.get('losses', perf.get('losing_trades', 0))
        
        metrics_table = [
            ["Total Trades", f"{perf.get('total_trades', 0)}"],
            ["Win Rate", f"{perf.get('win_rate', 0):.2%}"],
            ["Net P&L", f"${net_pnl:.2f}"],
            ["Gross P&L", f"${gross_pnl:.2f}"],
            ["Wins", f"{wins}"],
            ["Losses", f"{losses}"]
        ]
    
    if metrics_table:
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Metric", "Value"],
                    fill_color='rgb(30, 30, 30)',
                    align='left',
                    font=dict(color='white', size=12)
                ),
                cells=dict(
                    values=list(zip(*metrics_table)),
                    fill_color='rgb(50, 50, 50)',
                    align='left',
                    font=dict(color='white', size=11)
                )
            ),
            row=row, col=2
        )

def generate_trade_table_html(trades_df):
    """Generate HTML table of trades"""
    if trades_df.empty:
        return "<p>No trades to display</p>"
        
    # Format DataFrame for display
    display_df = trades_df.copy()
    display_df['entry_time'] = display_df['entry_time'].dt.strftime('%Y-%m-%d %H:%M')
    display_df['exit_time'] = display_df['exit_time'].dt.strftime('%Y-%m-%d %H:%M')
    display_df['profit_loss'] = display_df['profit_loss'].map('${:.2f}'.format)
    display_df['profit_loss_pct'] = display_df['profit_loss_pct'].map('{:.2f}%'.format)
    display_df['entry_price'] = display_df['entry_price'].map('{:.2f}'.format)
    display_df['exit_price'] = display_df['exit_price'].map('{:.2f}'.format)
    display_df['stop_loss'] = display_df['stop_loss'].map('{:.2f}'.format)
    display_df['take_profit'] = display_df['take_profit'].map('{:.2f}'.format)
    
    # Generate HTML table
    trade_table_html = display_df.to_html(index=False, classes='table table-striped table-bordered table-hover')
    return trade_table_html