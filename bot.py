import time
import requests
import yfinance as yf

TOKEN = "PUT_YOUR_TOKEN"
CHAT_ID = "PUT_YOUR_CHAT_ID"

SYMBOL = "GC=F"
INTERVAL = "15m"
PERIOD = "3d"

CHECK_INTERVAL = 900  # 15 minutes
last_signal = None


def send(msg):
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": msg}, timeout=20)
    except:
        pass


def get_data():
    for attempt in range(3):  # retry 3 times
        try:
            df = yf.download(
                SYMBOL,
                interval=INTERVAL,
                period=PERIOD,
                progress=False,
                threads=False
            )

            if df is not None and not df.empty:
                return df

            time.sleep(5)

        except Exception as e:
            print("Retrying data fetch...")
            time.sleep(10)

    return None


def get_signal():
    df = get_data()

    if df is None:
        print("No data (rate limit or market closed)")
        return None

    df["EMA50"] = df["Close"].ewm(span=50).mean()
    df["EMA200"] = df["Close"].ewm(span=200).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    last = df.iloc[-1]

    ema50 = last["EMA50"]
    ema200 = last["EMA200"]
    rsi = last["RSI"]

    if ema50 > ema200 and rsi < 30:
        return "BUY"
    elif ema50 < ema200 and rsi > 70:
        return "SELL"
    else:
        return None


send("✅ Gold Bot Stable Version Started")

while True:
    signal = get_signal()

    if signal and signal != last_signal:
        send(f"🔥 GOLD M15 SIGNAL: {signal}")
        last_signal = signal

    time.sleep(CHECK_INTERVAL)