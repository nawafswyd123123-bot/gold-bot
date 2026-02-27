import os
import time
import json
from typing import Dict, Any, Tuple

import requests
import pandas as pd


# =========================
# CONFIG (تقدر تغيّرهم من Environment إذا بدك)
# =========================
SYMBOL = os.getenv("SYMBOL", "XAU/USD")          # Gold spot
INTERVAL = os.getenv("INTERVAL", "30min")        # 1min, 5min, 15min, 30min, 45min, 1h...
OUTPUTSIZE = int(os.getenv("OUTPUTSIZE", "240")) # عدد الشموع
CYCLE_SECONDS = int(os.getenv("CYCLE_SECONDS", "600"))   # كل قديش يفحص (ثواني)
BACKOFF_SECONDS = int(os.getenv("BACKOFF_SECONDS", "120"))

STATE_FILE = os.getenv("STATE_FILE", "state.json")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

TWELVE_API_KEY = os.getenv("TWELVE_API_KEY", "").strip()


# =========================
# TELEGRAM
# =========================
def send_telegram(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram env vars missing (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID).")
        print("MSG:", text)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "disable_web_page_preview": True}

    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code != 200:
            print("⚠️ Telegram send failed:", r.status_code, r.text[:300])
    except Exception as e:
        print("⚠️ Telegram exception:", e)


# =========================
# STATE (anti-spam)
# =========================
def load_state() -> Dict[str, Any]:
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print("⚠️ state load error:", e)
    return {}

def save_state(state: Dict[str, Any]) -> None:
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("⚠️ state save error:", e)


# =========================
# DATA FETCH (TwelveData)
# =========================
def fetch_data() -> pd.DataFrame:
    if not TWELVE_API_KEY:
        print("❌ Missing TWELVE_API_KEY in environment variables")
        return pd.DataFrame()

    try:
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "outputsize": OUTPUTSIZE,
            "apikey": TWELVE_API_KEY,
            "format": "JSON"
        }

        r = requests.get(url, params=params, timeout=15)
        data = r.json()

        # إذا في خطأ بالمفتاح أو limit
        if "status" in data and data["status"] == "error":
            print("❌ TwelveData error:", data.get("message", data))
            return pd.DataFrame()

        values = data.get("values")
        if not values:
            print("❌ TwelveData empty values:", str(data)[:200])
            return pd.DataFrame()

        df = pd.DataFrame(values)

        # TwelveData بيرجع strings
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()

        df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
        df = df[["Open", "High", "Low", "Close"]].astype(float)
        df = df.dropna()

        print(f"📥 TWELVE fetch rows={len(df)} last={df.index[-1]}")
        return df

    except Exception as e:
        print("❌ Twelve fetch exception:", e)
        return pd.DataFrame()


# =========================
# STRATEGY (EMA Cross)
# =========================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def compute_signal(df: pd.DataFrame) -> Tuple[str, float, str]:
    if len(df) < 60:
        return "HOLD", float(df["Close"].iloc[-1]), "Not enough candles"

    close = df["Close"]
    fast = ema(close, 12)
    slow = ema(close, 26)

    # cross detection
    if fast.iloc[-2] <= slow.iloc[-2] and fast.iloc[-1] > slow.iloc[-1]:
        return "BUY", float(close.iloc[-1]), "EMA12 crossed above EMA26"
    if fast.iloc[-2] >= slow.iloc[-2] and fast.iloc[-1] < slow.iloc[-1]:
        return "SELL", float(close.iloc[-1]), "EMA12 crossed below EMA26"

    return "HOLD", float(close.iloc[-1]), "No cross"


# =========================
# MAIN LOOP
# =========================
def main() -> None:
    state = load_state()
    last_signal = state.get("last_signal", "NONE")
    last_sent_at = float(state.get("last_sent_at", 0))

    cooldown = int(os.getenv("SIGNAL_COOLDOWN_SECONDS", "1800"))  # 30 min

    send_telegram("✅ Gold bot started (TwelveData).")

    while True:
        df = fetch_data()

        if df.empty:
            print("⚠️ df empty — backoff")
            time.sleep(BACKOFF_SECONDS)
            continue

        signal, price, reason = compute_signal(df)
        now = time.time()

        print(f"📊 signal={signal} price={price:.2f} reason={reason}")

        # anti-spam:
        # - only BUY/SELL
        # - only if changed
        # - cooldown
        if signal in ("BUY", "SELL"):
            changed = (signal != last_signal)
            cooled = (now - last_sent_at) >= cooldown

            if changed and cooled:
                msg = (
                    f"📌 Gold Signal ({SYMBOL})\n"
                    f"Signal: {signal}\n"
                    f"Price: {price:.2f}\n"
                    f"TF: {INTERVAL}\n"
                    f"Reason: {reason}"
                )
                send_telegram(msg)

                last_signal = signal
                last_sent_at = now
                state["last_signal"] = last_signal
                state["last_sent_at"] = last_sent_at
                save_state(state)

        time.sleep(CYCLE_SECONDS)


if __name__ == "__main__":
    main()