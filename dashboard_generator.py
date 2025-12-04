import pandas as pd
import plotly.graph_objects as go
import os

def generate_dashboard():

    if not os.path.exists("trades.csv"):
        print("No trades.csv found — skipping dashboard update")
        return

    df = pd.read_csv("trades.csv")

    # =========================
    # METRICS
    # =========================
    df["pnl"] = 0.0
    df.loc[df["side"]=="buy",  "pnl"] = (df["tp"] - df["entry"])
    df.loc[df["side"]=="sell", "pnl"] = (df["entry"] - df["tp"])

    total_pnl = df["pnl"].sum()
    wins = (df["pnl"] > 0).sum()
    losses = (df["pnl"] <= 0).sum()
    winrate = round((wins / len(df)) * 100, 2) if len(df) > 0 else 0

    # =========================
    # EQUITY CURVE
    # =========================
    df["equity"] = df["pnl"].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["ts"],
        y=df["equity"],
        mode="lines",
        name="Equity Curve"
    ))

    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Time",
        yaxis_title="PnL (USDT)",
        template="plotly_dark"
    )

    equity_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    # =========================
    # HTML OUTPUT
    # =========================

    html = f"""
    <html>
    <head>
        <title>Mega Bot Dashboard</title>
        <style>
            body {{
                background-color: #0e0e0e;
                color: white;
                font-family: Arial;
                padding: 20px;
            }}
            .card {{
                background: #1a1a1a;
                padding: 20px;
                border-radius: 12px;
                margin-bottom: 20px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                border: 1px solid #333;
                padding: 8px;
                text-align: center;
            }}
            th {{
                background: #222;
            }}
        </style>
    </head>
    <body>

        <h1>Mega Bot Dashboard</h1>

        <div class="card">
            <h2>Performance Summary</h2>
            <p><b>Total PnL:</b> {total_pnl:.4f} USDT</p>
            <p><b>Win Rate:</b> {winrate}%</p>
            <p><b>Total Trades:</b> {len(df)}</p>
            <p><b>Winning Trades:</b> {wins}</p>
            <p><b>Losing Trades:</b> {losses}</p>
        </div>

        <div class="card">
            <h2>Equity Curve</h2>
            {equity_html}
        </div>

        <div class="card">
            <h2>Recent Trades</h2>
            {df.tail(15).to_html(index=False)}
        </div>

    </body>
    </html>
    """

    with open("dashboard.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("Dashboard updated → dashboard.html")

if __name__ == "__main__":
    generate_dashboard()
