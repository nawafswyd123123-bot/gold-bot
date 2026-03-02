import time
import requests
import yfinance as yf

TOKEN = "PUT_YOUR_TOKEN"
CHAT_ID = "PUT_YOUR_CHAT_ID"

SYMBOL = "GC=F"   # Gold Futures (stable)
INTERVAL = "15m"
PERIOD = "5d"

last_signal = None

def send(msg):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": msg})

def get_signal():
    try:
        df = yf.download(SYMBOL, interval=INTERVAL, period=PERIOD, progress=False)

        if df.empty:
            print("No market data")
            return None

        df["EMA50"] = df["Close"].ewm(span=50).mean()
        df["EMA200"] = df["Close"].ewm(span=200).mean()

        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        last = df.iloc[-1]

        if last["EMA50"] > last["EMA200"] and last["RSI"] < 30:
            return "BUY"
        elif last["EMA50"] < last["EMA200"] and last["RSI"] > 70:
            return "SELL"
        else:
            return None

    except Exception as e:
        print("Error:", e)
        return None

send("✅ Gold Bot Started")

while True:
    signal = get_signal()

    if signal and signal != last_signal:
        send(f"🔥 GOLD M15 SIGNAL: {signal}")
        last_signal = signal

    time.sleep(900)  # 15 minutes