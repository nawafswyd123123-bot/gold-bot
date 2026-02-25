import requests
import yfinance as yf
import pandas as pd
import time

TOKEN = "8772073953:AAGpdi9Q3AykDDa4L0pOKHcgJlsXMkOKplE"
CHAT_ID = "6150648369"

def send(msg):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": msg})

def fetch_gold():
    df = yf.download("GC=F", interval="15m", period="2d")
    df.dropna(inplace=True)
    return df

def make_signal(df):
    close = df["Close"]

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

    if last_sma20 > last_sma50 and last_rsi > 55:
        signal = "BUY"
    elif last_sma20 < last_sma50 and last_rsi < 45:
        signal = "SELL"
    else:
        signal = "WAIT"

    msg = f"""
GOLD SCALP (15m)

Price: {price}
SMA20: {last_sma20}
SMA50: {last_sma50}
RSI: {last_rsi}

Signal: {signal}
"""

    return msg

last_key = None

while True:
    try:
        df = fetch_gold()
        key = str(df.index[-1])

        if key != last_key:
            msg = make_signal(df)
            send(msg)
            last_key =
