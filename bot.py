import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests
import yfinance as yf

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")

SYMBOLS = ["GC=F", "XAUUSD=X"]
PERIOD = "5d"
INTERVAL = "15m"
CHECK_EVERY_SECONDS = 900

LAST_SIGNAL = None
LAST_SENT_CANDLE = None


def send_telegram_message(message: str) -> None:
    if not BOT_TOKEN or not CHAT_ID:
        print("BOT_TOKEN or CHAT_ID missing")
        return

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
                df = yf.download(
                    tickers=symbol,
                    period=PERIOD,
                    interval=INTERVAL,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )

                if df is not None and not df.empty:
                    df = df.dropna().copy()
                    if len(df) >= 60:
                        print(f"Downloaded data from {symbol}, rows={len(df)}")
                        return df, symbol

            except Exception as e:
                last_error = e
                print(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                time.sleep(5)

    print(f"Download failed. Last error: {last_error}")
    return None, None


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    close_col = "Close"
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

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

    df["ATR"] = (
        pd.concat(
            [
                (df["High"] - df["Low"]).abs(),
                (df["High"] - df["Close"].shift()).abs(),
                (df["Low"] - df["Close"].shift()).abs(),
            ],
            axis=1,
        )
        .max(axis=1)
        .rolling(14)
        .mean()
    )

    return df


def build_signal(df: pd.DataFrame):
    global LAST_SIGNAL, LAST_SENT_CANDLE

    if len(df) < 60:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]
    candle_time = str(df.index[-1])

    price = float(last["Close"])
    ema20 = float(last["EMA20"])
    ema50 = float(last["EMA50"])
    rsi = float(last["RSI"])
    atr = float(last["ATR"]) if pd.notna(last["ATR"]) else 0.0

    # فلتر اتجاه عام
    trend_up = ema20 > ema50
    trend_down = ema20 < ema50

    # فلتر حركة السعر
    bullish_price = price > ema20 and prev["Close"] > prev["EMA20"]
    bearish_price = price < ema20 and prev["Close"] < prev["EMA20"]

    signal = None
    strength = 0
    reason = []

    # BUY قوي
    if trend_up:
        strength += 1
        reason.append("EMA20 > EMA50")

        if bullish_price:
            strength += 1
            reason.append("price above EMA20")

        if 52 <= rsi <= 68:
            strength += 1
            reason.append("RSI supportive")

        if strength >= 3:
            signal = "BUY"

    # SELL قوي
    if trend_down:
        strength_sell = 1
        reason_sell = ["EMA20 < EMA50"]

        if bearish_price:
            strength_sell += 1
            reason_sell.append("price below EMA20")

        if 32 <= rsi <= 48:
            strength_sell += 1
            reason_sell.append("RSI supportive")

        if strength_sell >= 3:
            signal = "SELL"
            strength = strength_sell
            reason = reason_sell

    if signal is None:
        return None

    # منع تكرار نفس الإشارة على نفس الشمعة
    if LAST_SIGNAL == signal and LAST_SENT_CANDLE == candle_time:
        return None

    LAST_SIGNAL = signal
    LAST_SENT_CANDLE = candle_time

    # مستويات تقريبية
    if atr <= 0:
        atr = max(price * 0.002, 1.0)

    if signal == "BUY":
        sl = price - (atr * 1.2)
        tp1 = price + (atr * 1.2)
        tp2 = price + (atr * 2.0)
    else:
        sl = price + (atr * 1.2)
        tp1 = price - (atr * 1.2)
        tp2 = price - (atr * 2.0)

    score_percent = int((strength / 3) * 100)

    msg = (
        f"🔥 GOLD SIGNAL ({INTERVAL})\n"
        f"Type: {signal}\n"
        f"Price: {price:.2f}\n"
        f"EMA20: {ema20:.2f}\n"
        f"EMA50: {ema50:.2f}\n"
        f"RSI: {rsi:.2f}\n"
        f"Strength: {score_percent}%\n"
        f"SL: {sl:.2f}\n"
        f"TP1: {tp1:.2f}\n"
        f"TP2: {tp2:.2f}\n"
        f"Reason: {', '.join(reason)}\n"
        f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )

    return msg


def main():
    print("Bot started...")

    while True:
        try:
            df, source = download_gold_data()

            if df is None or source is None:
                print("No data received")
                time.sleep(CHECK_EVERY_SECONDS)
                continue

            df = calculate_indicators(df)
            signal_message = build_signal(df)

            if signal_message:
                signal_message += f"\nSource: {source}"
                send_telegram_message(signal_message)
                print("Signal sent successfully")
            else:
                print("No strong signal")

        except Exception as e:
            print(f"Main loop error: {e}")

        time.sleep(CHECK_EVERY_SECONDS)


if __name__ == "__main__":
    main()