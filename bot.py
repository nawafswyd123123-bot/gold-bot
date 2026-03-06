import os
import time
import traceback
from datetime import datetime

import requests
import pandas as pd
import yfinance as yf


BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("CHAT_ID", "").strip()

SYMBOLS = ["GC=F", "XAUUSD=X"]   # نجرب الذهب futures أولاً ثم spot
INTERVAL = "15m"
PERIOD = "5d"


def send_telegram(message: str):
    if not BOT_TOKEN or not CHAT_ID:
        print("Missing BOT_TOKEN or CHAT_ID")
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
    }

    try:
        r = requests.post(url, data=payload, timeout=20)
        print("Telegram status:", r.status_code, r.text)
    except Exception as e:
        print("Telegram send error:", e)


def download_gold_data():
    last_error = None

    for symbol in SYMBOLS:
        for attempt in range(3):
            try:
                print(f"Trying {symbol} | attempt {attempt + 1}")
                df = yf.download(
                    tickers=symbol,
                    period=PERIOD,
                    interval=INTERVAL,
                    progress=False,
                    auto_adjust=False,
                    threads=False,
                )

                if df is None or df.empty:
                    raise ValueError(f"No data returned for {symbol}")

                # أحياناً يرجع MultiIndex
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] for col in df.columns]

                needed = ["Open", "High", "Low", "Close"]
                for col in needed:
                    if col not in df.columns:
                        raise ValueError(f"Missing column {col} for {symbol}")

                df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()

                if len(df) < 50:
                    raise ValueError(f"Not enough candles for {symbol}. Got {len(df)}")

                print(f"Loaded data from {symbol}, rows={len(df)}")
                return df, symbol

            except Exception as e:
                last_error = f"{symbol} attempt {attempt + 1}: {e}"
                print(last_error)
                time.sleep(2)

    raise RuntimeError(f"All symbols failed. Last error: {last_error}")


def add_indicators(df: pd.DataFrame):
    df = df.copy()

    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)

    return df


def build_signal(df: pd.DataFrame):
    last = df.iloc[-1]
    prev = df.iloc[-2]

    buy = (
        last["Close"] > last["EMA20"] > last["EMA50"]
        and prev["Close"] <= prev["EMA20"]
        and last["RSI"] > 52
    )

    sell = (
        last["Close"] < last["EMA20"] < last["EMA50"]
        and prev["Close"] >= prev["EMA20"]
        and last["RSI"] < 48
    )

    if buy:
        return "BUY"
    if sell:
        return "SELL"
    return None


def main():
    try:
        df, used_symbol = download_gold_data()
        df = add_indicators(df)

        signal = build_signal(df)
        last = df.iloc[-1]

        now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        price = float(last["Close"])
        ema20 = float(last["EMA20"])
        ema50 = float(last["EMA50"])
        rsi = float(last["RSI"])

        if signal:
            message = (
                f"🔥 GOLD SIGNAL ({INTERVAL})\n"
                f"Type: {signal}\n"
                f"Price: {price:.2f}\n"
                f"EMA20: {ema20:.2f}\n"
                f"EMA50: {ema50:.2f}\n"
                f"RSI: {rsi:.2f}\n"
                f"Source: {used_symbol}\n"
                f"Time: {now_str}"
            )
            send_telegram(message)
            print("Signal sent:", signal)
        else:
            print("No signal now.")
            send_telegram(
                f"ℹ️ No signal now\nPrice: {price:.2f}\nRSI: {rsi:.2f}\nSource: {used_symbol}\nTime: {now_str}"
            )

    except Exception as e:
        err = f"Bot error: {e}"
        print(err)
        print(traceback.format_exc())
        send_telegram(f"❌ {err}")


if __name__ == "__main__":
    main()