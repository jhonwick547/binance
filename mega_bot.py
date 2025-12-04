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

# =========================================================
# LOGGING (simple print so GitHub Actions shows everything)
# =========================================================
def log(*args):
    print("[MEGA]", *args)

# =========================================================
# UTIL: Save trades to CSV
# =========================================================
def log_trade(symbol, side, qty, entry, sl, tp):
    file = "trades.csv"
    new = not os.path.exists(file)
    with open(file, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts","symbol","side","qty","entry","sl","tp"])
        w.writerow([datetime.utcnow().isoformat(), symbol, side, qty, entry, sl, tp])

# =========================================================
# UTIL: Load OHLCV
# =========================================================
def load_ohlcv(symbol, timeframe, limit=500):
    ex = ccxt.binance()
    o = ex.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(o, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df

# =========================================================
# INDICATORS
# =========================================================
def add_indicators(df):
    df = df.copy()

    df["ema20"] = ta.trend.EMAIndicator(df["close"], 20).ema_indicator()
    df["ema50"] = ta.trend.EMAIndicator(df["close"], 50).ema_indicator()
    df["ema200"] = ta.trend.EMAIndicator(df["close"], 200).ema_indicator()

    df["rsi"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()

    macd = ta.trend.MACD(df["close"])
    df["macd"]  = macd.macd()
    df["macds"] = macd.macd_signal()
    df["macdh"] = macd.macd_diff()

    df["atr"] = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], 14
    ).average_true_range()

    df["vol_sma"] = df["volume"].rolling(20).mean()

    df.dropna(inplace=True)
    return df

# =========================================================
# LSTM MODEL
# =========================================================
SEQ = 20

class LSTM(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.lstm = nn.LSTM(n, 64, batch_first=True)
        self.fc   = nn.Linear(64, 1)
        self.sig  = nn.Sigmoid()

    def forward(self, x):
        out,_ = self.lstm(x)
        out = out[:, -1]
        out = self.fc(out)
        return self.sig(out)

# =========================================================
# MEGA BOT
# =========================================================
class MegaBot:
    def __init__(self, key, sec):
        self.ex = ccxt.binance({
            "apiKey": key,
            "secret": sec,
            "enableRateLimit": True,
            "options": {"defaultType": "future"}
        })

        # Demo trading safety
        try:
            self.ex.enable_demo_trading(True)
        except:
            pass

        self.symbols = ["ETHUSDT", "XRPUSDT", "1000PEPEUSDT"]
        self.cooldown = 900  # 15 minutes
        self.daily_limit = 3
        self.last_trade = {s: 0 for s in self.symbols}
        self.trade_count = {s: 0 for s in self.symbols}
        self.day = time.strftime("%Y-%m-%d")

        # Load ML model
        if os.path.exists("models/rf.pkl"):
            self.rf = joblib.load("models/rf.pkl")
        else:
            self.rf = None

        # Load RL policy
        if os.path.exists("models/rl.pkl"):
            self.Q, self.A = joblib.load("models/rl.pkl")
        else:
            self.Q, self.A = None, ["hold","long","short"]

        # Load LSTM
        if os.path.exists("models/lstm.pt"):
            # create dummy to load into
            df = add_indicators(load_ohlcv("ETHUSDT","5m",SEQ+5))
            n = df.drop(["ts"],axis=1).shape[1]
            self.lstm = LSTM(n)
            self.lstm.load_state_dict(torch.load("models/lstm.pt"))
            self.lstm.eval()
        else:
            self.lstm = None

    # =====================================================
    # Trusted rule-based filter
    # =====================================================
    def trusted_filter(self, symbol):
        try:
            df5  = add_indicators(load_ohlcv(symbol,"5m"))
            df15 = add_indicators(load_ohlcv(symbol,"15m"))
            df1h = add_indicators(load_ohlcv(symbol,"1h"))
        except:
            return None, None

        f = df5.iloc[-1]
        fp = df5.iloc[-2]
        m = df15.iloc[-1]
        s = df1h.iloc[-1]

        price = f["close"]

        if f["atr"]/price < 0.002: return None, None
        if f["volume"] < f["vol_sma"]*1.2: return None, None

        uptrend   = s["ema50"] > s["ema200"]
        downtrend = s["ema50"] < s["ema200"]

        mid_bull = m["ema20"] > m["ema50"]
        mid_bear = m["ema20"] < m["ema50"]

        rsi_up   = fp["rsi"] < 30 <= f["rsi"]
        rsi_down = fp["rsi"] > 70 >= f["rsi"]

        macd_up   = f["macd"] > f["macds"] and f["macdh"] > 0
        macd_down = f["macd"] < f["macds"] and f["macdh"] < 0

        if uptrend and mid_bull and rsi_up and macd_up:
            return "long", f["atr"]

        if downtrend and mid_bear and rsi_down and macd_down:
            return "short", f["atr"]

        return None, None

    # =====================================================
    # ML prediction (RF)
    # =====================================================
    def ml_predict(self, symbol):
        if self.rf is None:
            return None

        df = add_indicators(load_ohlcv(symbol,"5m",100))
        X = df.drop(["ts"],axis=1).iloc[-1:].values
        return int(self.rf.predict(X)[0])  # 1 long / 0 short

    # =====================================================
    # LSTM prediction
    # =====================================================
    def lstm_predict(self, symbol):
        if self.lstm is None:
            return None

        df = add_indicators(load_ohlcv(symbol,"5m",SEQ+5))
        X = df.drop(["ts"],axis=1).values[-SEQ:]
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
        return float(self.lstm(X))  # >0.5 long, <0.5 short

    # =====================================================
    # RL action
    # =====================================================
    def rl_predict(self, symbol):
        if self.Q is None:
            return "hold"

        df = add_indicators(load_ohlcv(symbol,"5m",50))
        rsi = df["rsi"].iloc[-1]
        s = np.digitize(rsi,[30,50,70])
        a = np.argmax(self.Q[s])
        return self.A[a]

    # =====================================================
    # Position size
    # =====================================================
    def size(self, atr, price):
        bal = self.ex.fetch_balance()["total"].get("USDT", 0)
        if bal <= 0:
            return 0
        risk = bal * 0.01
        sl_dist = atr * 2
        qty = risk / sl_dist
        return round(qty, 3)

    # =====================================================
    # Execute trade
    # =====================================================
    def trade(self, symbol):
        # Reset daily counter
        today = time.strftime("%Y-%m-%d")
        if today != self.day:
            self.day = today
            self.trade_count = {s: 0 for s in self.symbols}

        # Cooldown
        if time.time() - self.last_trade[symbol] < self.cooldown:
            return None

        # Daily limit
        if self.trade_count[symbol] >= self.daily_limit:
            return None

        # === All Models ===
        rule, atr = self.trusted_filter(symbol)
        if not rule:
            return None

        ml = self.ml_predict(symbol)
        lstm = self.lstm_predict(symbol)
        rl = self.rl_predict(symbol)

        long_signal = (rule=="long" and ml==1 and lstm>0.5 and rl!="short")
        short_signal= (rule=="short" and ml==0 and lstm<0.5 and rl!="long")

        if not long_signal and not short_signal:
            return None

        price = self.ex.fetch_ticker(symbol)["last"]
        qty = self.size(atr, price)
        if qty <= 0:
            return None

        side = "buy" if long_signal else "sell"
        exit_side = "sell" if long_signal else "buy"

        sl = price - atr*2 if long_signal else price + atr*2
        tp = price + atr*3 if long_signal else price - atr*3

        # Execute
        self.ex.create_order(symbol, "MARKET", side, qty)
        self.ex.create_order(symbol, "STOP_MARKET", exit_side, qty, params={"stopPrice": float(sl)})
        self.ex.create_order(symbol, "LIMIT", exit_side, qty, price=float(tp))

        log("TRADE:", symbol, side, qty, "SL", sl, "TP", tp)
        log_trade(symbol, side, qty, price, sl, tp)

        self.last_trade[symbol] = time.time()
        self.trade_count[symbol] += 1

        return side, price

    # =====================================================
    # One cycle (GitHub safe)
    # =====================================================
    def run_once(self):
        for sym in self.symbols:
            try:
                self.trade(sym)
            except Exception as e:
                log("ERR", sym, e)


if __name__ == "__main__":
    api = os.getenv("BINANCE_KEY")
    sec = os.getenv("BINANCE_SECRET")

    if not api or not sec:
        raise SystemExit("Missing keys")

    bot = MegaBot(api, sec)
    bot.run_once()
