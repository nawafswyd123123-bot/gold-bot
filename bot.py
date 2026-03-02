import time
import requests
import yfinance as yf
import pandas as pd

TOKEN ="8772073953:AAGpdi9Q3AykDDa4L0pOKHcgJlsXMkOKplE"
CHAT_ID = "6150648369"

last_signal = None

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message}
    requests.post(url, data=data)

def get_signal():
    df = yf.download("XAUUSD=X", interval="15m", period="1d")

    df["EMA50"] = df["Close"].ewm(span=50).mean()
    df["EMA200"] = df["Close"].ewm(span=200).mean()

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    last = df.iloc[-1]

    # 🔥 إشارة قوية فقط
    if last["EMA50"] > last["EMA200"] and last["RSI"] < 30:
        return "BUY"
    elif last["EMA50"] < last["EMA200"] and last["RSI"] > 70:
        return "SELL"
    else:
        return None

while True:
    try:
        signal = get_signal()

        if signal is not None and signal != last_signal:
            message = f"🔥 GOLD SIGNAL (M15)\nSignal: {signal}"
            send_telegram(message)
            last_signal = signal

        time.sleep(300)  # يفحص كل 5 دقائق

    except Exception as e:
        print("Error:", e)
        time.sleep(60)