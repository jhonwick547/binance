import os
import time
import logging

import ccxt
import pandas as pd
import pandas_ta as ta

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
    High-filter futures bot:
      - Multi-timeframe (5m / 15m / 1h)
      - Trades only when ALL conditions line up
      - ATR-based SL/TP
      - Cooldown + daily trade cap
    """

    def __init__(self, api_key: str, api_secret: str):
        self.exchange = ccxt.binance({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })

        # IMPORTANT: use demo trading, not sandbox
        # Requires recent ccxt version
        self.exchange.enable_demo_trading()

        # Symbols to trade
        self.symbols = ["ETHUSDT", "XRPUSDT", "1000PEPEUSDT"]

        # Timeframes
        self.tf_fast = "5m"
        self.tf_mid = "15m"
        self.tf_slow = "1h"

        # Risk / MM
        self.risk_per_trade = 0.01       # 1% of equity
        self.atr_mult_sl = 2.0
        self.atr_mult_tp = 3.0

        # Overtrading protection
        self.cooldown_sec = 15 * 60      # 15 minutes cooldown per symbol
        self.max_trades_per_day = 5      # per symbol

        # Tracking
        self.last_trade_time = {s: 0 for s in self.symbols}
        self.trades_today = {s: 0 for s in self.symbols}
        self.current_day = time.strftime("%Y-%m-%d")

    # -----------------------------
    # Data + indicators
    # -----------------------------
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
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

    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Core indicators for each timeframe."""
        df = df.copy()
        df["ema20"] = ta.ema(df["close"], 20)
        df["ema50"] = ta.ema(df["close"], 50)
        df["ema200"] = ta.ema(df["close"], 200)

        df["rsi"] = ta.rsi(df["close"], 14)

        macd = ta.macd(df["close"])
        df["macd"] = macd["MACD_12_26_9"]
        df["macds"] = macd["MACDs_12_26_9"]
        df["macdh"] = macd["MACDh_12_26_9"]

        df["adx"] = ta.adx(df["high"], df["low"], df["close"], 14)["ADX_14"]
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], 14)
        df["vol_sma"] = ta.sma(df["volume"], 20)

        df = df.dropna().reset_index(drop=True)
        return df

    def get_mtf_frames(self, symbol: str):
        df_fast = self.fetch_ohlcv(symbol, self.tf_fast)
        df_mid = self.fetch_ohlcv(symbol, self.tf_mid)
        df_slow = self.fetch_ohlcv(symbol, self.tf_slow)

        if df_fast is None or df_mid is None or df_slow is None:
            return None, None, None

        df_fast = self.add_indicators(df_fast)
        df_mid = self.add_indicators(df_mid)
        df_slow = self.add_indicators(df_slow)

        if len(df_fast) < 5 or len(df_mid) < 5 or len(df_slow) < 5:
            return None, None, None

        return df_fast, df_mid, df_slow

    # -----------------------------
    # Signal logic
    # -----------------------------
    def _trusted_signal(self, symbol: str):
        """
        Returns: 'long', 'short', or None.
        Conditions are intentionally strict.
        """
        frames = self.get_mtf_frames(symbol)
        if frames[0] is None:
            logger.info(f"{symbol}: not enough data / failed fetch")
            return None, None, None

        df_fast, df_mid, df_slow = frames

        # Last rows
        f = df_fast.iloc[-1]
        f_prev = df_fast.iloc[-2]
        m = df_mid.iloc[-1]
        s = df_slow.iloc[-1]

        price = f["close"]

        # Volatility + volume filters
        if f["atr"] / price < 0.002:  # < 0.2% ATR → too dead
            logger.info(f"{symbol}: volatility too low, skipping")
            return None, None, None

        if f["volume"] < f["vol_sma"] * 1.2:  # need 20% above avg volume
            logger.info(f"{symbol}: volume weak, skipping")
            return None, None, None

        # Higher timeframe trend
        slow_uptrend = s["ema50"] > s["ema200"]
        slow_downtrend = s["ema50"] < s["ema200"]

        # Mid timeframe structure
        mid_bull = m["close"] > m["ema50"] and m["ema20"] > m["ema50"]
        mid_bear = m["close"] < m["ema50"] and m["ema20"] < m["ema50"]

        # Fast timeframe momentum + candle pattern
        rsi_now = f["rsi"]
        rsi_prev = f_prev["rsi"]
        macd_now = f["macd"]
        macds_now = f["macds"]
        macdh_now = f["macdh"]

        # --- LONG SETUP ---
        long_conditions = [
            slow_uptrend,                    # higher timeframe uptrend
            mid_bull,                        # mid timeframe bullish
            rsi_prev < 30 <= rsi_now,        # RSI crosses up from oversold
            macd_now > macds_now,            # MACD line above signal
            macdh_now > 0,                   # MACD histogram positive
        ]

        # --- SHORT SETUP ---
        short_conditions = [
            slow_downtrend,
            mid_bear,
            rsi_prev > 70 >= rsi_now,        # RSI crosses down from overbought
            macd_now < macds_now,
            macdh_now < 0,
        ]

        if all(long_conditions):
            logger.info(f"{symbol}: TRUSTED LONG signal")
            return "long", f["atr"], price

        if all(short_conditions):
            logger.info(f"{symbol}: TRUSTED SHORT signal")
            return "short", f["atr"], price

        logger.info(f"{symbol}: no trusted setup")
        return None, None, None

    # -----------------------------
    # Position sizing
    # -----------------------------
    def _position_size(self, symbol: str, price: float, atr: float) -> float:
        balance = self.exchange.fetch_balance()
        equity = balance["total"]["USDT"]
        risk_capital = equity * self.risk_per_trade

        stop_distance = self.atr_mult_sl * atr
        if stop_distance <= 0:
            return 0.0

        qty = risk_capital / stop_distance
        # Round to 3 decimals, safe enough for USDT-margined futures
        return round(max(qty, 0), 3)

    # -----------------------------
    # Order execution
    # -----------------------------
    def _execute_trade(self, symbol: str, direction: str, price: float, atr: float):
        now = time.time()

        # Reset daily counters if new day
        today = time.strftime("%Y-%m-%d")
        if today != self.current_day:
            self.current_day = today
            self.trades_today = {s: 0 for s in self.symbols}

        # Cooldown
        if now - self.last_trade_time[symbol] < self.cooldown_sec:
            logger.info(f"{symbol}: in cooldown, skipping trade")
            return

        # Daily trade cap
        if self.trades_today[symbol] >= self.max_trades_per_day:
            logger.info(f"{symbol}: reached daily trade limit, skipping")
            return

        qty = self._position_size(symbol, price, atr)
        if qty <= 0:
            logger.info(f"{symbol}: position size <= 0, skipping")
            return

        # ATR-based SL/TP
        if direction == "long":
            sl = price - self.atr_mult_sl * atr
            tp = price + self.atr_mult_tp * atr
            side = "buy"
            sl_side = "sell"
        else:
            sl = price + self.atr_mult_sl * atr
            tp = price - self.atr_mult_tp * atr
            side = "sell"
            sl_side = "buy"

        try:
            logger.info(f"{symbol}: placing {direction.upper()} qty={qty} @ ~{price}")
            order = self.exchange.create_order(
                symbol=symbol,
                type="MARKET",
                side=side,
                amount=qty,
            )
            logger.info(f"{symbol}: entry order: {order}")

            # Stop-loss (STOP_MARKET)
            self.exchange.create_order(
                symbol=symbol,
                type="STOP_MARKET",
                side=sl_side,
                amount=qty,
                params={"stopPrice": float(f"{sl:.4f}")},
            )

            # Take-profit (LIMIT)
            self.exchange.create_order(
                symbol=symbol,
                type="LIMIT",
                side=sl_side,
                amount=qty,
                price=float(f"{tp:.4f}"),
            )

            logger.info(
                f"{symbol}: SL={sl:.4f}, TP={tp:.4f}, qty={qty}"
            )

            self.last_trade_time[symbol] = now
            self.trades_today[symbol] += 1

        except Exception as e:
            logger.error(f"{symbol}: trade failed: {e}")

    # -----------------------------
    # Main loop
    # -----------------------------
    def run(self):
        logger.info("TrustedSignalBot started.")
        while True:
            for symbol in self.symbols:
                try:
                    direction, atr, price = self._trusted_signal(symbol)
                    if direction is None:
                        continue
                    self._execute_trade(symbol, direction, price, atr)
                except Exception as e:
                    logger.error(f"Error in loop for {symbol}: {e}")
                    continue

            # Check every 5 minutes (aligned with 5m TF)
            logger.info("Cycle completed, sleeping 300s...")
            time.sleep(300)


if __name__ == "__main__":
    api_key = os.environ.get("BINANCE_KEY", "")
    api_secret = os.environ.get("BINANCE_SECRET", "")

    if not api_key or not api_secret:
        raise SystemExit("Set BINANCE_KEY and BINANCE_SECRET env vars.")

    bot = TrustedSignalBot(api_key, api_secret)
    bot.run()
