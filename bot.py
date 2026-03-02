import time
import requests
import yfinance as yf

TOKEN = "PUT_YOUR_TOKEN"
CHAT_ID = "PUT_YOUR_CHAT_ID"

SYMBOL = "GC=F"
INTERVAL = "15m"
PERIOD = "3d"

last_signal = None


def send(msg):
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": msg}, timeout=20)
    except:
        pass


def get_signal():
    try:
        df = yf.download(SYMBOL, interval=INTERVAL, period=PERIOD, progress=False)

        if df is None or df.empty:
            print("No market data")
            return None

        df["EMA50"] = df["Close"].ewm(span=50).mean()
        df["EMA200"] = df["Close"].ewm(span=200).mean()

        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # 🔥 ناخد آخر قيمة فقط كرقم
        ema50 = float(df["EMA50"].iloc[-1])
        ema200 = float(df["EMA200"].iloc[-1])
        rsi = float(df["RSI"].iloc[-1])

        if ema50 > ema200 and rsi < 30:
            return "BUY"
        elif ema50 < ema200 and rsi > 70:
            return "SELL"
        else:
            return None

    except Exception as e:
        print("Error:", e)
        return None


send("✅ Gold Bot Final Stable Version Started")

while True:
    signal = get_signal()

    if signal and signal != last_signal:
        send(f"🔥 GOLD M15 SIGNAL: {signal}")
        last_signal = signal

    time.sleep(900)  # 15 minutes