import requests
import yfinance as yf
import pandas as pd
import time

TOKEN = "8772073953:AAFSpv_JfoO50m-BMQu0o-xo0knwXGKBXA8"
CHAT_ID = "6150648369"

def send(msg):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": msg})

def fetch_gold_15m():
    df = yf.download("GC=F", interval="15m", period="2d")
    df.dropna(inplace=True)
    return df

def make_signal(df):
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    last = df.iloc[-1]

    price = round(last["Close"], 2)
    sma20 = round(last["SMA20"], 2)
    sma50 = round(last["SMA50"], 2)
    rsi = round(last["RSI"], 1)

    if sma20 > sma50 and rsi > 50:
        signal = "🟢 BUY"
    elif sma20 < sma50 and rsi < 50:
        signal = "🔴 SELL"
    else:
        signal = "⏳ WAIT"

    msg = f"""📊 GOLD SCALP (15m)

💰 Price: {price}
📈 SMA20: {sma20}
📉 SMA50: {sma50}
📊 RSI: {rsi}

{signal}
"""
    return msg

last_sent_key = None

while True:
    try:
        df = fetch_gold_15m()
        key = str(df.index[-1])

        if key != last_sent_key:
            msg = make_signal(df)
            send(msg)
            last_sent_key = key

        time.sleep(900)  # كل 15 دقيقة

    except Exception as e:
        send(f"⚠ Error: {e}")
        time.sleep(60)
