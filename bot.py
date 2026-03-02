import time
import requests
import yfinance as yf

TOKEN = "PUT_YOUR_TOKEN"
CHAT_ID = "PUT_YOUR_CHAT_ID"

SYMBOL = "GC=F"
INTERVAL = "15m"
PERIOD = "3d"
SLEEP_SEC = 900  # 15 minutes

last_signal = None


def send(msg: str):
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": msg}, timeout=20)
    except Exception as e:
        print("Telegram error:", e)


def fetch_df():
    try:
        df = yf.download(SYMBOL, interval=INTERVAL, period=PERIOD, progress=False, threads=False)
        if df is None or df.empty:
            return None
        return df
    except Exception as e:
        print("Yahoo error:", e)
        return None


def get_signal():
    df = fetch_df()
    if df is None:
        print("No market data")
        return None

    # Indicators
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # ✅ IMPORTANT: last value as FLOAT (not Series)
    ema50 = float(df["EMA50"].iloc[-1])
    ema200 = float(df["EMA200"].iloc[-1])
    rsi = float(df["RSI"].iloc[-1])

    if ema50 > ema200 and rsi < 30:
        return "BUY"
    if ema50 < ema200 and rsi > 70:
        return "SELL"
    return None


send("✅ Gold Bot started (M15)")

while True:
    sig = get_signal()

    if sig and sig != last_signal:
        send(f"🔥 GOLD M15 SIGNAL: {sig}\nSymbol: {SYMBOL}")
        last_signal = sig

    time.sleep(SLEEP_SEC)