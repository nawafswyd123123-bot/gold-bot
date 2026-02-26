import os
import time
import math
import traceback
from datetime import datetime, timezone

import requests
import pandas as pd
import yfinance as yf

# =========================
# ENV (Render Environment)
# =========================
BOT_TOKEN = (os.getenv("BOT_TOKEN") or "").strip()
CHAT_ID = (os.getenv("CHAT_ID") or "").strip()

SYMBOL = os.getenv("SYMBOL", "GC=F").strip()       # Gold futures on Yahoo
INTERVAL = os.getenv("INTERVAL", "15m").strip()    # 1m, 5m, 15m, 1h, ...
PERIOD = os.getenv("PERIOD", "5d").strip()         # 1d, 5d, 1mo, ...

SLEEP_SECONDS = int(os.getenv("SLEEP_SECONDS", "600"))  # default 10 minutes (safe)

# RSI thresholds (you can change)
RSI_LEN = int(os.getenv("RSI_LEN", "14"))
RSI_BUY = float(os.getenv("RSI_BUY", "30"))    # Buy if RSI <= 30
RSI_SELL = float(os.getenv("RSI_SELL", "70"))  # Sell if RSI >= 70

# =========================
# Helpers
# =========================
def send_telegram(text: str) -> None:
    if not BOT_TOKEN or not CHAT_ID:
        print("Missing BOT_TOKEN or CHAT_ID in environment variables.")
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}

    try:
        r = requests.post(url, data=payload, timeout=20)
        if r.status_code != 200:
            print("Telegram error:", r.status_code, r.text[:200])
    except Exception as e:
        print("Telegram exception:", e)


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean()

    rs = avg_gain / avg_loss.replace(0, math.nan)
    return 100 - (100 / (1 + rs))


def fetch_data() -> pd.DataFrame:
    """
    Download candles from yfinance. Returns empty DF if failed.
    """
    try:
        df = yf.download(
            SYMBOL,
            interval=INTERVAL,
            period=PERIOD,
            progress=False,
            threads=False,
        )
    except Exception as e:
        print("yfinance download exception:", e)
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # Fix MultiIndex columns sometimes returned by yfinance
    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(-1)

    needed = {"Open", "High", "Low", "Close"}
    if not needed.issubset(set(df.columns)):
        return pd.DataFrame()

    df = df.dropna()
    return df


def make_signal(df: pd.DataFrame):
    """
    Very simple strategy:
    - BUY when RSI <= RSI_BUY
    - SELL when RSI >= RSI_SELL
    Only if RSI is valid.
    """
    close = df["Close"].astype(float)
    r = rsi(close, RSI_LEN)

    if r.isna().all():
        return None

    last_rsi = float(r.iloc[-1])
    last_close = float(close.iloc[-1])

    if last_rsi <= RSI_BUY:
        return ("BUY", last_close, last_rsi)
    if last_rsi >= RSI_SELL:
        return ("SELL", last_close, last_rsi)

    return None


# =========================
# Main loop
# =========================
last_candle_ts = None
last_signal_sent = None

def main():
    global last_candle_ts, last_signal_sent

    send_telegram("✅ Bot started successfully")

    while True:
        try:
            df = fetch_data()
            if df.empty:
                print("No data. Sleeping...")
                time.sleep(SLEEP_SECONDS)
                continue

            # candle timestamp (as string, stable)
            current_ts = str(df.index[-1])

            # Only react on NEW candle
            if current_ts != last_candle_ts:
                sig = make_signal(df)
                if sig:
                    side, price, last_rsi = sig

                    # Anti-repeat: don't spam same side repeatedly
                    if side != last_signal_sent:
                        msg = (
                            f"📌 {SYMBOL} ({INTERVAL})\n"
                            f"Signal: {side}\n"
                            f"Price: {price:.2f}\n"
                            f"RSI({RSI_LEN}): {last_rsi:.1f}\n"
                            f"Time: {current_ts}"
                        )
                        send_telegram(msg)
                        last_signal_sent = side

                last_candle_ts = current_ts

            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            print("Loop error:", e)
            traceback.print_exc()
            time.sleep(max(180, SLEEP_SECONDS))


if __name__ == "__main__":
    main()