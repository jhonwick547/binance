import os
import time
import logging

import ccxt
import pandas as pd
import ta

# =========================
# LOGGING
# =========================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

fh = logging.FileHandler("trusted_bot.log")
fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(ch)


# =========================
# BOT CLASS
# =========================
class TrustedSignalBot:
    """
    High-filter futures bot using ta-lib library:
      - Multi-timeframe (5m / 15m / 1h)
      - Very strict signal conditions
      - ATR-based SL/TP
      - Cooldown + daily trade cap
    """

    def __init__(self, api_key: str, api_secret: str):
        # Base exchange config
        self.exchange = ccxt.binance({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })

        # Demo trading compatibility
        try:
            try:
                self.exchange.enable_demo_trading(True)
            except TypeError:
                self.exchange.enable_demo_trading()
            logger.info("Demo trading enabled.")
        except Exception as e:
            logger.warning(f"Demo trading method unavailable: {e}")

        # Symbols to trade
        self.symbols = ["ETHUSDT", "XRPUSDT", "1000PEPEUSDT"]

        # Timeframes
        self.tf_fast = "5m"
        self.tf_mid = "15m"
        self.tf_slow = "1h"

        # Risk
        self.risk_per_trade = 0.01
        self.atr_mult_sl = 2.0
        self.atr_mult_tp = 3.0

        # Anti-overtrade
        self.cooldown_sec = 15 * 60
        self.max_trades_per_day = 5

        # Trackers
        self.last_trade_time = {s: 0 for s in self.symbols}
        self.trades_today = {s: 0 for s in self.symbols}
        self.current_day = time.strftime("%Y-%m-%d")

    # ======================================================
    # Indicators using ta library
    # ======================================================
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
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

    # ======================================================
    # Data
    # ======================================================
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit=300):
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=["ts", "open", "high", "low", "close", "volume"],
            )
            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch {symbol} {timeframe}: {e}")
            return None

    def get_mtf(self, symbol):
        f = self.fetch_ohlcv(symbol, self.tf_fast)
        m = self.fetch_ohlcv(symbol, self.tf_mid)
        s = self.fetch_ohlcv(symbol, self.tf_slow)

        if f is None or m is None or s is None:
            return None, None, None

        f = self.add_indicators(f)
        m = self.add_indicators(m)
        s = self.add_indicators(s)

        if len(f) < 5 or len(m) < 5 or len(s) < 5:
            return None, None, None

        return f, m, s

    # ======================================================
    # SIGNAL LOGIC
    # ======================================================
    def trusted_signal(self, symbol):
        f, m, s = self.get_mtf(symbol)
        if f is None:
            return None, None, None

        last = f.iloc[-1]
        prev = f.iloc[-2]
        mid = m.iloc[-1]
        slow = s.iloc[-1]

        price = last["close"]

        # Filters
        if last["atr"] / price < 0.002:
            return None, None, None
        if last["volume"] < last["vol_sma"] * 1.2:
            return None, None, None

        # Trend
        uptrend = slow["ema50"] > slow["ema200"]
        downtrend = slow["ema50"] < slow["ema200"]

        # Mid structure
        mid_bull = mid["close"] > mid["ema50"] and mid["ema20"] > mid["ema50"]
        mid_bear = mid["close"] < mid["ema50"] and mid["ema20"] < mid["ema50"]

        # Momentum
        rsi_up = prev["rsi"] < 30 <= last["rsi"]
        rsi_down = prev["rsi"] > 70 >= last["rsi"]

        macd_up = last["macd"] > last["macds"] and last["macdh"] > 0
        macd_down = last["macd"] < last["macds"] and last["macdh"] < 0

        # Long
        if uptrend and mid_bull and rsi_up and macd_up:
            return "long", last["atr"], price

        # Short
        if downtrend and mid_bear and rsi_down and macd_down:
            return "short", last["atr"], price

        return None, None, None

    # ======================================================
    # Position sizing
    # ======================================================
    def size(self, atr, price):
        bal = self.exchange.fetch_balance()["total"].get("USDT", 0)
        risk = bal * self.risk_per_trade
        sl_distance = self.atr_mult_sl * atr
        if sl_distance <= 0:
            return 0
        qty = risk / sl_distance
        return round(qty, 3)

    # ======================================================
    # Execution
    # ======================================================
    def execute(self, symbol, direction, price, atr):
        now = time.time()

        # Reset daily count
        today = time.strftime("%Y-%m-%d")
        if today != self.current_day:
            self.current_day = today
            self.trades_today = {s: 0 for s in self.symbols}

        # Cooldown
        if now - self.last_trade_time[symbol] < self.cooldown_sec:
            return

        # Daily cap
        if self.trades_today[symbol] >= self.max_trades_per_day:
            return

        qty = self.size(atr, price)
        if qty <= 0:
            return

        if direction == "long":
            side = "buy"
            exit_side = "sell"
            sl = price - self.atr_mult_sl * atr
            tp = price + self.atr_mult_tp * atr
        else:
            side = "sell"
            exit_side = "buy"
            sl = price + self.atr_mult_sl * atr
            tp = price - self.atr_mult_tp * atr

        try:
            self.exchange.create_order(symbol, "MARKET", side, qty)

            # SL
            self.exchange.create_order(
                symbol, "STOP_MARKET", exit_side, qty, params={"stopPrice": float(sl)}
            )

            # TP
            self.exchange.create_order(
                symbol, "LIMIT", exit_side, qty, price=float(tp)
            )

            self.last_trade_time[symbol] = now
            self.trades_today[symbol] += 1

            logger.info(f"{symbol}: {direction.upper()} qty={qty} SL={sl} TP={tp}")

        except Exception as e:
            logger.error(f"{symbol}: execution error {e}")

    # ======================================================
    # ONE-CYCLE MODE FOR GITHUB ACTIONS
    # ======================================================
    def run_once(self):
        for symbol in self.symbols:
            direction, atr, price = self.trusted_signal(symbol)
            if direction:
                self.execute(symbol, direction, price, atr)
            else:
                logger.info(f"{symbol}: No trade this cycle")


# ======================================================
# ENTRY POINT
# ======================================================
if __name__ == "__main__":
    api_key = os.environ.get("BINANCE_KEY")
    api_secret = os.environ.get("BINANCE_SECRET")

    if not api_key or not api_secret:
        raise SystemExit("Missing BINANCE_KEY or BINANCE_SECRET")

    bot = TrustedSignalBot(api_key, api_secret)

    # GitHub Actions mode: only run once
    if "--once" in os.sys.argv:
        bot.run_once()
    else:
        while True:
            bot.run_once()
            time.sleep(300)
