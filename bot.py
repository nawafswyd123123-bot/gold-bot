import os
import time
import math
import traceback
from datetime import datetime

import requests
import pandas as pd
import yfinance as yf

# =========================
# ENV
# =========================
BOT_TOKEN = (os.getenv("BOT_TOKEN") or "").strip()
CHAT_ID = (os.getenv("CHAT_ID") or "").strip()

SYMBOL = os.getenv("SYMBOL", "GC=F").strip()
INTERVAL = os.getenv("INTERVAL", "15m").strip()
PERIOD = os.getenv("PERIOD", "5d").strip()

SLEEP_SECONDS = int(os.getenv("SLEEP_SECONDS", "600"))  # safe default: 10 min
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "1800"))  # default: 30 min between alerts

# Indicators
RSI_LEN = int(os.getenv("RSI_LEN", "14"))
RSI_BUY = float(os.getenv("RSI_BUY", "35"))   # less strict than 30
RSI_SELL = float(os.getenv("RSI_SELL", "65")) # less strict than 70

SMA_FAST = int(os.getenv("SMA_FAST", "20"))
SMA_SLOW = int(os.getenv("SMA_SLOW", "50"))

# =========================
# Telegram
# =========================
def send_telegram(text: str) -> None:
    if not BOT_TOKEN or not CHAT_ID:
        print("❌ Missing BOT_TOKEN or CHAT_ID.")
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}

    try:
        r = requests.post(url, data=payload, timeout=20)
        if r.status_code != 200:
            print("❌ Telegram error:", r.status_code, r.text[:200])
    except Exception as e:
        print("❌ Telegram exception:", e)

# =========================
# Indicators
# =========================
def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean()

    rs = avg_gain / avg_loss.replace(0, math.nan)
    return 100 - (100 / (1 + rs))

def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()

# =========================
# Data
# =========================
def fetch_data() -> pd.DataFrame:
    try:
        df = yf.download(
            SYMBOL,
            interval=INTERVAL,
            period=PERIOD,
            progress=False,
            threads=False,
        )
    except Exception as e:
        print("❌ yfinance exception:", e)
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # fix multiindex columns
    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(-1)

    needed = {"Open", "High", "Low", "Close"}
    if not needed.issubset(set(df.columns)):
        return pd.DataFrame()

    df = df.dropna()
    return df

# =========================
# Strategy (Upgraded)
# =========================
def make_signal(df: pd.DataFrame):
    close = df["Close"].astype(float)

    r = rsi(close, RSI_LEN)
    fast = sma(close, SMA_FAST)
    slow = sma(close, SMA_SLOW)

    if r.isna().all() or fast.isna().all() or slow.isna().all():
        return None

    last_close = float(close.iloc[-1])
    last_rsi = float(r.iloc[-1])
    last_fast = float(fast.iloc[-1])
    last_slow = float(slow.iloc[-1])

    # Trend filter (stronger):
    # - Only BUY if price above slow MA
    # - Only SELL if price below slow MA
    in_uptrend = last_close > last_slow
    in_downtrend = last_close < last_slow

    # Cross filter:
    # - BUY stronger when SMA20 > SMA50
    # - SELL stronger when SMA20 < SMA50
    cross_up = last_fast > last_slow
    cross_down = last_fast < last_slow

    # BUY conditions
    if in_uptrend and cross_up and last_rsi <= RSI_BUY:
        conf = "STRONG"
        return ("BUY", conf, last_close, last_rsi, last_fast, last_slow)

    # SELL conditions
    if in_downtrend and cross_down and last_rsi >= RSI_SELL:
        conf = "STRONG"
        return ("SELL", conf, last_close, last_rsi, last_fast, last_slow)

    # Optional weaker signals (if you want later): comment out for now
    return None

# =========================
# Main loop
# =========================
last_candle_ts = None
last_signal = None
last_sent_at = 0

def main():
    global last_candle_ts, last_signal, last_sent_at

    send_telegram("✅ Gold Signal Bot started successfully")
    print("✅ Started. Symbol:", SYMBOL, "Interval:", INTERVAL, "Sleep:", SLEEP_SECONDS)

    while True:
        try:
            df = fetch_data()
            if df.empty:
                print("⚠️ No data, sleeping", SLEEP_SECONDS)
                time.sleep(SLEEP_SECONDS)
                continue

            current_ts = str(df.index[-1])

            # only act on new candle
            if current_ts != last_candle_ts:
                sig = make_signal(df)

                if sig:
                    side, conf, price, last_rsi, sma20, sma50 = sig

                    now = time.time()
                    cooldown_ok = (now - last_sent_at) >= COOLDOWN_SECONDS
                    not_repeat = (side != last_signal)

                    if cooldown_ok and not_repeat:
                        msg = (
                            f"📌 GOLD SIGNAL ({INTERVAL})\n"
                            f"Symbol: {SYMBOL}\n"
                            f"Signal: {side} ({conf})\n"
                            f"Price: {price:.2f}\n"
                            f"SMA{SMA_FAST}: {sma20:.2f}\n"
                            f"SMA{SMA_SLOW}: {sma50:.2f}\n"
                            f"RSI({RSI_LEN}): {last_rsi:.1f}\n"
                            f"Time: {current_ts}"
                        )
                        send_telegram(msg)
                        last_signal = side
                        last_sent_at = now
                        print("✅ Sent:", side, "at", current_ts)
                    else:
                        print("ℹ️ Signal ignored (cooldown/repeat):", side, current_ts)

                last_candle_ts = current_ts

            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            print("❌ Loop error:", e)
            traceback.print_exc()
            time.sleep(max(180, SLEEP_SECONDS))

if __name__ == "__main__":
    main()