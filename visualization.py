import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json

def create_html_report(backtest_results_dir, output_file=None):
    """
    Create an HTML report with interactive charts
    
    Args:
        backtest_results_dir: Directory containing backtest results
        output_file: Output HTML file path
    """
    # Set default output file if not provided
    if output_file is None:
        output_file = os.path.join(backtest_results_dir, 'backtest_report.html')
    
    # Load data
    trades_path = os.path.join(backtest_results_dir, 'trades.csv')
    equity_path = os.path.join(backtest_results_dir, 'equity_curve.csv')
    metrics_path = os.path.join(backtest_results_dir, 'metrics.json')
    
    trades_df = pd.read_csv(trades_path) if os.path.exists(trades_path) else pd.DataFrame()
    equity_df = pd.read_csv(equity_path) if os.path.exists(equity_path) else pd.DataFrame()
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f) if os.path.exists(metrics_path) else {}
    
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
    
    # Create figure with equity curve and trades
    fig = create_dashboard(trades_df, equity_df, metrics)
    
    # Write to HTML file
    html_string = fig.to_html(include_plotlyjs=True, full_html=True)
    with open(output_file, 'w') as f:
        f.write(html_string)
    
    print(f"HTML report created: {output_file}")
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