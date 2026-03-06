import os
import time
from datetime import datetime

import pandas as pd
import requests
import yfinance as yf


BOT_TOKEN = os.getenv("BOT_TOKEN", "PUT_YOUR_BOT_TOKEN_HERE")
CHAT_ID = os.getenv("CHAT_ID", "PUT_YOUR_CHAT_ID_HERE")

SYMBOLS = ["GC=F", "XAUUSD=X"]
PERIOD = "2d"
INTERVAL = "15m"
CHECK_EVERY_SECONDS = 900

LAST_SIGNAL = None
LAST_SENT_CANDLE = None


def send_telegram_message(message: str) -> None:
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
    }

    response = requests.post(url, data=payload, timeout=20)
    response.raise_for_status()


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
                    raise ValueError("No data returned")

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] for col in df.columns]

                needed = ["Open", "High", "Low", "Close", "Volume"]
                for col in needed:
                    if col not in df.columns:
                        raise ValueError(f"Missing column: {col}")

                df = df.dropna(subset=["Open", "High", "Low", "Close"])

                if len(df) < 50:
                    raise ValueError("Not enough candles")

                print(f"Loaded data from {symbol}")
                return df, symbol

            except Exception as e:
                last_error = f"{symbol} attempt {attempt + 1} failed: {e}"
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

    buy = last["RSI"] < 45
    sell = last["RSI"] > 55

    if buy:
        return "BUY"
    if sell:
        return "SELL"
    return None


def main():
    global LAST_SIGNAL, LAST_SENT_CANDLE

    try:
        df, used_symbol = download_gold_data()
        df = add_indicators(df)

        signal = build_signal(df)
        last = df.iloc[-1]
        candle_time = str(df.index[-1])

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        price = float(last["Close"])
        ema20 = float(last["EMA20"])
        ema50 = float(last["EMA50"])
        rsi = float(last["RSI"])

        if signal and (signal != LAST_SIGNAL or candle_time != LAST_SENT_CANDLE):
            LAST_SIGNAL = signal
            LAST_SENT_CANDLE = candle_time

            message = (
                f"🔥 GOLD SIGNAL (15m)\n"
                f"Type: {signal}\n"
                f"Price: {price:.2f}\n"
                f"EMA20: {ema20:.2f}\n"
                f"EMA50: {ema50:.2f}\n"
                f"RSI: {rsi:.2f}\n"
                f"Source: {used_symbol}\n"
                f"Time: {now_str}"
            )

            print("Sending new signal to Telegram...")
            send_telegram_message(message)
        else:
            print("No new signal to send.")

    except Exception as e:
        print(f"Main loop error: {e}")


if __name__ == "__main__":
    print("Gold Signal Bot started...")
    while True:
        main()
        print(f"Sleeping for {CHECK_EVERY_SECONDS} seconds")
        time.sleep(CHECK_EVERY_SECONDS)