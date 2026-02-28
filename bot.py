import os
import time
import json
import math
from datetime import datetime, timezone

import requests

# =========================
# CONFIG (via ENV on Render)
# =========================
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY", "").strip()
SYMBOL = os.getenv("SYMBOL", "XAU/USD").strip()          # Examples: "XAU/USD" or "XAUUSD"
INTERVAL = os.getenv("INTERVAL", "15min").strip()         # 1min, 5min, 15min, 1h, 4h, 1day ...
ROWS = int(os.getenv("ROWS", "240"))                      # candles
SLEEP_SECONDS = int(os.getenv("SLEEP_SECONDS", "600"))    # check every 10 minutes by default

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# Strategy params
FAST_EMA = int(os.getenv("FAST_EMA", "20"))
SLOW_EMA = int(os.getenv("SLOW_EMA", "50"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))

RSI_BUY_MAX = float(os.getenv("RSI_BUY_MAX", "70"))   # buy only if RSI <= 70
RSI_SELL_MIN = float(os.getenv("RSI_SELL_MIN", "30")) # sell only if RSI >= 30

# Local persistence (prevents duplicate alerts)
LAST_SIGNAL_FILE = os.getenv("LAST_SIGNAL_FILE", "last_signal.json")


# =========================
# Helpers
# =========================
def log(msg: str):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{now}] {msg}", flush=True)


def require_env():
    missing = []
    if not TWELVE_API_KEY:
        missing.append("TWELVE_API_KEY")
    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not TELEGRAM_CHAT_ID:
        missing.append("TELEGRAM_CHAT_ID")

    if missing:
        raise RuntimeError(f"Missing ENV vars: {', '.join(missing)}")


def read_last_signal():
    try:
        with open(LAST_SIGNAL_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("signal", ""), data.get("ts", "")
    except Exception:
        return "", ""


def write_last_signal(signal: str):
    try:
        with open(LAST_SIGNAL_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {"signal": signal, "ts": datetime.now(timezone.utc).isoformat()},
                f,
                ensure_ascii=False,
            )
    except Exception as e:
        log(f"WARN: could not write last signal file: {e}")


def send_telegram(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "disable_web_page_preview": True}
    r = requests.post(url, json=payload, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"Telegram error {r.status_code}: {r.text}")


# =========================
# Indicators
# =========================
def ema(values, period):
    """Exponential moving average"""
    if len(values) < period:
        return []
    k = 2 / (period + 1)
    out = [None] * len(values)
    # seed with SMA
    sma = sum(values[:period]) / period
    out[period - 1] = sma
    prev = sma
    for i in range(period, len(values)):
        prev = values[i] * k + prev * (1 - k)
        out[i] = prev
    return out


def rsi(values, period=14):
    if len(values) < period + 1:
        return []
    out = [None] * len(values)
    gains = []
    losses = []
    for i in range(1, period + 1):
        change = values[i] - values[i - 1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    rs = (avg_gain / avg_loss) if avg_loss != 0 else math.inf
    out[period] = 100 - (100 / (1 + rs))

    for i in range(period + 1, len(values)):
        change = values[i] - values[i - 1]
        gain = max(change, 0)
        loss = max(-change, 0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        rs = (avg_gain / avg_loss) if avg_loss != 0 else math.inf
        out[i] = 100 - (100 / (1 + rs))
    return out


# =========================
# Data fetch (Twelve Data)
# =========================
def fetch_candles():
    """
    Twelve Data time_series docs: returns list with datetime, open, high, low, close, volume
    """
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "outputsize": ROWS,
        "apikey": TWELVE_API_KEY,
        "format": "JSON",
    }
    r = requests.get(url, params=params, timeout=30)
    data = r.json()

    if "status" in data and data["status"] == "error":
        raise RuntimeError(f"TwelveData error: {data.get('message')}")

    values = data.get("values", [])
    if not values:
        raise RuntimeError(f"No candles returned. Response: {data}")

    # TwelveData returns newest first -> reverse to oldest first
    values = list(reversed(values))

    closes = []
    times = []
    for row in values:
        try:
            closes.append(float(row["close"]))
            times.append(row["datetime"])
        except Exception:
            continue

    if len(closes) < max(SLOW_EMA, RSI_PERIOD) + 5:
        raise RuntimeError(f"Not enough data: got {len(closes)} closes")

    return times, closes


# =========================
# Strategy
# =========================
def compute_signal(times, closes):
    e_fast = ema(closes, FAST_EMA)
    e_slow = ema(closes, SLOW_EMA)
    r = rsi(closes, RSI_PERIOD)

    i = len(closes) - 1
    price = closes[i]
    fast = e_fast[i]
    slow = e_slow[i]
    rsi_now = r[i]

    if fast is None or slow is None or rsi_now is None:
        return "HOLD", price, "Indicators not ready"

    # previous values for crossover detection
    prev_i = i - 1
    prev_fast = e_fast[prev_i]
    prev_slow = e_slow[prev_i]
    if prev_fast is None or prev_slow is None:
        return "HOLD", price, "Prev indicators not ready"

    # Bullish crossover: fast crosses above slow
    if prev_fast <= prev_slow and fast > slow:
        if rsi_now <= RSI_BUY_MAX:
            return "BUY", price, f"EMA{FAST_EMA}>{SLOW_EMA} cross up | RSI={rsi_now:.1f}"
        else:
            return "HOLD", price, f"Cross up but RSI high ({rsi_now:.1f})"

    # Bearish crossover: fast crosses below slow
    if prev_fast >= prev_slow and fast < slow:
        if rsi_now >= RSI_SELL_MIN:
            return "SELL", price, f"EMA{FAST_EMA}<{SLOW_EMA} cross down | RSI={rsi_now:.1f}"
        else:
            return "HOLD", price, f"Cross down but RSI low ({rsi_now:.1f})"

    return "HOLD", price, "No cross"


def format_alert(signal, price, reason):
    return (
        f"🐋 GOLD SIGNAL: {signal}\n"
        f"Symbol: {SYMBOL}\n"
        f"TF: {INTERVAL}\n"
        f"Price: {price:.2f}\n"
        f"Reason: {reason}\n"
        f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )


# =========================
# Main loop
# =========================
def main():
    require_env()
    log(f"Starting bot | symbol={SYMBOL} interval={INTERVAL} rows={ROWS} sleep={SLEEP_SECONDS}s")

    while True:
        try:
            times, closes = fetch_candles()
            signal, price, reason = compute_signal(times, closes)

            log(f"signal={signal} price={price:.2f} reason={reason}")

            if signal in ("BUY", "SELL"):
                last_sig, last_ts = read_last_signal()
                if last_sig != signal:
                    send_telegram(format_alert(signal, price, reason))
                    write_last_signal(signal)
                    log(f"Telegram sent ✅ ({signal})")
                else:
                    log(f"Skip duplicate signal ({signal}) last_ts={last_ts}")

        except Exception as e:
            log(f"ERROR: {e}")

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()