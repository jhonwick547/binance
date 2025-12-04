import os
import time
import csv
import ccxt
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
import ta
from datetime import datetime

# =============================
# Simple logger
# =============================
def log(*args):
    print("[MEGA]", *args)

# =============================
# Trade logging -> CSV
# =============================
def log_trade(symbol, side, qty, entry, sl, tp):
    file = "trades.csv"
    new_file = not os.path.exists(file)
    with open(file, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["ts", "symbol", "side", "qty", "entry", "sl", "tp"])
        w.writerow([datetime.utcnow().isoformat(), symbol, side, qty, entry, sl, tp])

# =============================
# OHLCV loader
# =============================
def load_ohlcv(symbol, timeframe, limit=500):
    ex = ccxt.binance()
    data = ex.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df

# =============================
# Indicator builder (ta)
# =============================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ema20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema200"] = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()

    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macds"] = macd.macd_signal()
    df["macdh"] = macd.macd_diff()

    df["atr"] = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=14
    ).average_true_range()

    df["vol_sma"] = df["volume"].rolling(20).mean()

    df.dropna(inplace=True)
    return df

# =============================
# LSTM model
# =============================
SEQ = 20

class LSTM(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.lstm = nn.LSTM(n_features, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1]
        out = self.fc(out)
        return self.sig(out)

# =============================
# MegaBot
# =============================
class MegaBot:
    def __init__(self, api_key: str, api_secret: str):
        self.ex = ccxt.binance({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })

        # Try to enable demo trading; ignore if unsupported
        try:
            try:
                self.ex.enable_demo_trading(True)
            except TypeError:
                self.ex.enable_demo_trading()
            log("Demo trading enabled.")
        except Exception as e:
            log("Demo trading not explicitly enabled:", e)

        self.symbols = ["ETHUSDT", "XRPUSDT", "1000PEPEUSDT"]

        # Risk / MM
        self.cooldown_sec = 15 * 60
        self.daily_limit = 3
        self.last_trade = {s: 0 for s in self.symbols}
        self.trade_count = {s: 0 for s in self.symbols}
        self.current_day = time.strftime("%Y-%m-%d")

        # Load ML models if they exist; otherwise bot falls back to rule-only
        self.rf = None
        self.lstm = None
        self.Q = None
        self.A = ["hold", "long", "short"]

        # RandomForest
        if os.path.exists("models/rf.pkl"):
            try:
                self.rf = joblib.load("models/rf.pkl")
                log("RandomForest model loaded.")
            except Exception as e:
                log("Failed to load rf.pkl:", e)

        # RL
        if os.path.exists("models/rl.pkl"):
            try:
                self.Q, self.A = joblib.load("models/rl.pkl")
                log("RL policy loaded.")
            except Exception as e:
                log("Failed to load rl.pkl:", e)

        # LSTM
        if os.path.exists("models/lstm.pt"):
            try:
                df_tmp = add_indicators(load_ohlcv("ETHUSDT", "5m", SEQ + 5))
                n_features = df_tmp.drop(["ts"], axis=1).shape[1]
                self.lstm = LSTM(n_features)
                self.lstm.load_state_dict(torch.load("models/lstm.pt"))
                self.lstm.eval()
                log("LSTM model loaded.")
            except Exception as e:
                log("Failed to load lstm.pt:", e)

    # -------------------------
    # Trusted rule-based filter
    # -------------------------
    def trusted_filter(self, symbol):
        try:
            df5 = add_indicators(load_ohlcv(symbol, "5m"))
            df15 = add_indicators(load_ohlcv(symbol, "15m"))
            df1h = add_indicators(load_ohlcv(symbol, "1h"))
        except Exception as e:
            log("trusted_filter data error", symbol, e)
            return None, None

        f = df5.iloc[-1]
        fp = df5.iloc[-2]
        m = df15.iloc[-1]
        s = df1h.iloc[-1]

        price = f["close"]

        # Filters: volatility & volume
        if f["atr"] / price < 0.002:
            return None, None
        if f["volume"] < f["vol_sma"] * 1.2:
            return None, None

        uptrend = s["ema50"] > s["ema200"]
        downtrend = s["ema50"] < s["ema200"]

        mid_bull = m["ema20"] > m["ema50"]
        mid_bear = m["ema20"] < m["ema50"]

        rsi_up = fp["rsi"] < 30 <= f["rsi"]
        rsi_down = fp["rsi"] > 70 >= f["rsi"]

        macd_up = f["macd"] > f["macds"] and f["macdh"] > 0
        macd_down = f["macd"] < f["macds"] and f["macdh"] < 0

        if uptrend and mid_bull and rsi_up and macd_up:
            return "long", f["atr"]

        if downtrend and mid_bear and rsi_down and macd_down:
            return "short", f["atr"]

        return None, None

    # -------------------------
    # ML predictions
    # -------------------------
    def ml_predict(self, symbol):
        if self.rf is None:
            return None
        df = add_indicators(load_ohlcv(symbol, "5m", 100))
        X = df.drop(["ts"], axis=1).iloc[-1:].values
        try:
            return int(self.rf.predict(X)[0])  # 1=long, 0=short
        except Exception as e:
            log("RF prediction error", symbol, e)
            return None

    def lstm_predict(self, symbol):
        if self.lstm is None:
            return None
        df = add_indicators(load_ohlcv(symbol, "5m", SEQ + 5))
        X = df.drop(["ts"], axis=1).values[-SEQ:]
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
        try:
            prob = float(self.lstm(X))
            return prob  # >0.5 long, <0.5 short
        except Exception as e:
            log("LSTM prediction error", symbol, e)
            return None

    def rl_predict(self, symbol):
        if self.Q is None:
            return None
        df = add_indicators(load_ohlcv(symbol, "5m", 50))
        rsi = df["rsi"].iloc[-1]
        s = np.digitize(rsi, [30, 50, 70])
        try:
            a = np.argmax(self.Q[s])
            return self.A[a]  # "hold","long","short"
        except Exception as e:
            log("RL prediction error", symbol, e)
            return None

    # -------------------------
    # Position sizing
    # -------------------------
    def position_size(self, atr, price):
        try:
            bal = self.ex.fetch_balance()["total"].get("USDT", 0)
        except Exception as e:
            log("fetch_balance error", e)
            return 0

        if bal <= 0:
            return 0

        risk = bal * 0.01  # 1% risk
        sl_dist = atr * 2
        if sl_dist <= 0:
            return 0

        qty = risk / sl_dist
        return round(qty, 3)

    # -------------------------
    # Execute one trade for one symbol
    # -------------------------
    def trade_symbol(self, symbol):
        # Reset day counters
        today = time.strftime("%Y-%m-%d")
        if today != self.current_day:
            self.current_day = today
            self.trade_count = {s: 0 for s in self.symbols}

        # Cooldown
        if time.time() - self.last_trade[symbol] < self.cooldown_sec:
            return

        # Daily limit
        if self.trade_count[symbol] >= self.daily_limit:
            return

        # 1) Rule-based filter
        rule_dir, atr = self.trusted_filter(symbol)
        if not rule_dir or not atr:
            return

        # 2) ML / LSTM / RL signals (may be None)
        ml = self.ml_predict(symbol)
        lstm_prob = self.lstm_predict(symbol)
        rl_action = self.rl_predict(symbol)

        # Build ensemble with graceful fallbacks
        long_ok = (rule_dir == "long")
        short_ok = (rule_dir == "short")

        if ml is not None:
            long_ok = long_ok and (ml == 1)
            short_ok = short_ok and (ml == 0)

        if lstm_prob is not None:
            long_ok = long_ok and (lstm_prob > 0.5)
            short_ok = short_ok and (lstm_prob < 0.5)

        if rl_action is not None:
            long_ok = long_ok and (rl_action != "short")
            short_ok = short_ok and (rl_action != "long")

        if not long_ok and not short_ok:
            return

        try:
            ticker = self.ex.fetch_ticker(symbol)
            price = ticker["last"]
        except Exception as e:
            log("fetch_ticker error", symbol, e)
            return

        qty = self.position_size(atr, price)
        if qty <= 0:
            return

        side = "buy" if long_ok else "sell"
        exit_side = "sell" if long_ok else "buy"

        sl = price - 2 * atr if long_ok else price + 2 * atr
        tp = price + 3 * atr if long_ok else price - 3 * atr

        try:
            # Entry
            self.ex.create_order(symbol, "MARKET", side, qty)

            # Stop-loss
            self.ex.create_order(
                symbol,
                "STOP_MARKET",
                exit_side,
                qty,
                params={"stopPrice": float(sl)},
            )

            # Take-profit
            self.ex.create_order(
                symbol,
                "LIMIT",
                exit_side,
                qty,
                price=float(tp),
            )

            log("TRADE", symbol, side, "qty", qty, "SL", sl, "TP", tp)
            log_trade(symbol, side, qty, price, sl, tp)

            self.last_trade[symbol] = time.time()
            self.trade_count[symbol] += 1

        except Exception as e:
            log("order error", symbol, e)

    # -------------------------
    # One GitHub-safe cycle
    # -------------------------
    def run_once(self):
        for s in self.symbols:
            try:
                self.trade_symbol(s)
            except Exception as e:
                log("loop error", s, e)


if __name__ == "__main__":
    api = os.getenv("BINANCE_KEY")
    sec = os.getenv("BINANCE_SECRET")

    if not api or not sec:
        raise SystemExit("Missing BINANCE_KEY or BINANCE_SECRET")

    bot = MegaBot(api, sec)
    bot.run_once()
