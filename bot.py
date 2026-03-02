import time
import requests
import MetaTrader5 as mt5
import pandas as pd

TOKEN = "PUT_YOUR_TOKEN"
CHAT_ID = "PUT_YOUR_CHAT_ID"

SYMBOL = "XAUUSD"
TF = mt5.TIMEFRAME_M15
BARS = 400
CHECK_SEC = 60
COOLDOWN_SEC = 15 * 60

last_signal = None
last_sent = 0

def send(msg):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": msg},
            timeout=20
        )
    except:
        pass

def rsi(close, n=14):
    d = close.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    rs = g / l
    return 100 - (100 / (1 + rs))

def get_signal():
    rates = mt5.copy_rates_from_pos(SYMBOL, TF, 0, BARS)
    if rates is None or len(rates) < 250:
        return None, None

    df = pd.DataFrame(rates)
    df["EMA50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["close"].ewm(span=200, adjust=False).mean()
    df["RSI"] = rsi(df["close"], 14)

    ema50 = float(df["EMA50"].iloc[-1])
    ema200 = float(df["EMA200"].iloc[-1])
    rr = float(df["RSI"].iloc[-1])
    price = float(df["close"].iloc[-1])

    if ema50 > ema200 and rr <= 30:
        return "BUY", price
    if ema50 < ema200 and rr >= 70:
        return "SELL", price

    return None, price

# start
if not mt5.initialize():
    send("❌ MT5 init failed")
    raise SystemExit

mt5.symbol_select(SYMBOL, True)
send("✅ Gold Bot started (MT5)")

while True:
    sig, price = get_signal()
    now = int(time.time())

    if sig and sig != last_signal and (now - last_sent) >= COOLDOWN_SEC:
        send(f"🔥 XAUUSD M15 SIGNAL: {sig}\nPrice: {price:.2f}")
        last_signal = sig
        last_sent = now

    time.sleep(CHECK_SEC)