import os
import time
import requests
import yfinance as yf
import pandas as pd

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message
    }
    requests.post(url, data=payload)

def fetch_gold():
    df = yf.download("GC=F", interval="15m", period="2d")

    # حل مشكلة MultiIndex
    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(0)

    return df

def make_signal(df):
    close = df["Close"]

    # تأكد انو Series مش DataFrame
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    price = round(close.iloc[-1], 2)
    last_sma20 = round(sma20.iloc[-1], 2)
    last_sma50 = round(sma50.iloc[-1], 2)
    last_rsi = round(rsi.iloc[-1], 2)

    if last_sma20 > last_sma50 and last_rsi > 50:
        signal = "BUY"
    elif last_sma20 < last_sma50 and last_rsi < 50:
        signal = "SELL"
    else:
        signal = "WAIT"

    message = f"""
📊 GOLD SCALP (15m)

💰 Price: {price}
📈 SMA20: {last_sma20}
📉 SMA50: {last_sma50}
📊 RSI: {last_rsi}

🚦 Signal: {signal}
"""

    return message


last_candle = None

while True:
    try:
        df = fetch_gold()
        current_candle = str(df.index[-1])

        if current_candle != last_candle:
            msg = make_signal(df)
            send_telegram(msg)
            last_candle = current_candle

        time.sleep(60)

    except Exception as e:
        print("Error:", e)
        time.sleep(60)