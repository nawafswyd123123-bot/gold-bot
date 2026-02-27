import os
import time
import json
import requests
import pandas as pd
from typing import Dict, Any

# =====================
# CONFIG
# =====================

SYMBOL = "GC=F"   # Gold Futures
INTERVAL = "30m"
RANGE = "5d"

CYCLE_SECONDS = 600
BACKOFF_SECONDS = 300

STATE_FILE = "state.json"

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# =====================
# TELEGRAM
# =====================

def send_telegram(text: str):
    if not TOKEN or not CHAT_ID:
        print("⚠ Telegram env vars missing")
        print(text)
        return

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text
    }

    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            print("Telegram error:", r.text)
    except Exception as e:
        print("Telegram exception:", e)

# =====================
# STATE
# =====================

def load_state() -> Dict[str, Any]:
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                return json.load(f)
    except:
        pass
    return {}

def save_state(state: Dict[str, Any]):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

# =====================
# DIRECT YAHOO FETCH
# =====================

def fetch_data():
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{SYMBOL}"
        params = {
            "interval": INTERVAL,
            "range": RANGE
        }

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        r = requests.get(url, params=params, headers=headers, timeout=10)
        data = r.json()

        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        quote = result["indicators"]["quote"][0]

        df = pd.DataFrame({
            "Open": quote["open"],
            "High": quote["high"],
            "Low": quote["low"],
            "Close": quote["close"]
        })

        df["Datetime"] = pd.to_datetime(timestamps, unit="s")
        df.set_index("Datetime", inplace=True)

        df = df.dropna()

        print(f"📥 DIRECT fetch rows={len(df)}")

        return df

    except Exception as e:
        print("❌ fetch error:", e)
        return pd.DataFrame()

# =====================
# SIMPLE EMA STRATEGY
# =====================

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def get_signal(df):
    if len(df) < 50:
        return "HOLD", df["Close"].iloc[-1]

    close = df["Close"]
    fast = ema(close, 12)
    slow = ema(close, 26)

    if fast.iloc[-2] <= slow.iloc[-2] and fast.iloc[-1] > slow.iloc[-1]:
        return "BUY", close.iloc[-1]

    if fast.iloc[-2] >= slow.iloc[-2] and fast.iloc[-1] < slow.iloc[-1]:
        return "SELL", close.iloc[-1]

    return "HOLD", close.iloc[-1]

# =====================
# MAIN LOOP
# =====================

def main():
    state = load_state()
    last_signal = state.get("last_signal")

    send_telegram("✅ Gold bot started")

    while True:
        df = fetch_data()

        if df.empty:
            print("⚠ df empty — sleeping")
            time.sleep(BACKOFF_SECONDS)
            continue

        signal, price = get_signal(df)

        print("Signal:", signal, "Price:", price)

        if signal in ["BUY", "SELL"] and signal != last_signal:
            message = (
                f"📌 Gold Signal\n"
                f"Signal: {signal}\n"
                f"Price: {round(price,2)}\n"
                f"Timeframe: {INTERVAL}"
            )

            send_telegram(message)

            state["last_signal"] = signal
            save_state(state)
            last_signal = signal

        time.sleep(CYCLE_SECONDS)

if __name__ == "__main__":
    main()