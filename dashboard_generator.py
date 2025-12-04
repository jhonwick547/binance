import pandas as pd
import plotly.graph_objects as go
import os

def generate_dashboard():
    has_trades = os.path.exists("trades.csv") and os.path.getsize("trades.csv") > 0

    if has_trades:
        df = pd.read_csv("trades.csv")
        if df.empty:
            has_trades = False

    if has_trades:
        # Basic PnL approximation based on SL/TP vs entry
        df["pnl"] = 0.0
        df.loc[df["side"] == "buy",  "pnl"] = df["tp"] - df["entry"]
        df.loc[df["side"] == "sell", "pnl"] = df["entry"] - df["tp"]

        df["equity"] = df["pnl"].cumsum()

        total_pnl = df["pnl"].sum()
        wins = (df["pnl"] > 0).sum()
        losses = (df["pnl"] <= 0).sum()
        total_trades = len(df)
        winrate = round(wins / total_trades * 100, 2) if total_trades > 0 else 0.0

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["ts"],
            y=df["equity"],
            mode="lines",
            name="Equity"
        ))
        fig.update_layout(
            title="Equity Curve",
            xaxis_title="Time",
            yaxis_title="PnL (USDT)",
            template="plotly_dark",
            height=400
        )

        equity_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
        recent_html = df.tail(20).to_html(index=False)
    else:
        # No trades yet -> placeholder content
        total_pnl = 0.0
        wins = losses = total_trades = 0
        winrate = 0.0
        equity_html = "<p>No trades yet. Come back after the bot runs and logs some trades.</p>"
        recent_html = "<p>No trades to display.</p>"

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Mega Bot Dashboard</title>
    <meta charset="utf-8" />
    <style>
        body {{
            background-color: #0d0d0d;
            color: #e0e0e0;
            font-family: Arial, sans-serif;
            padding: 20px;
        }}
        .card {{
            background: #181818;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
        }}
        h1, h2 {{
            margin-top: 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            border: 1px solid #333;
            padding: 6px;
            text-align: center;
        }}
        th {{
            background: #222;
        }}
        a {{
            color: #4fa3ff;
        }}
    </style>
</head>
<body>

    <h1>Mega Bot Dashboard</h1>

    <div class="card">
        <h2>Performance Summary</h2>
        <p><b>Total PnL:</b> {total_pnl:.4f} USDT</p>
        <p><b>Win Rate:</b> {winrate:.2f}%</p>
        <p><b>Total Trades:</b> {total_trades}</p>
        <p><b>Wins:</b> {wins} | <b>Losses:</b> {losses}</p>
    </div>

    <div class="card">
        <h2>Equity Curve</h2>
        {equity_html}
    </div>

    <div class="card">
        <h2>Recent Trades</h2>
        {recent_html}
    </div>

</body>
</html>
"""
    with open("dashboard.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("Dashboard updated → dashboard.html")


if __name__ == "__main__":
    generate_dashboard()
