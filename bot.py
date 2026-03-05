import os
import time
import json
import math
import requests
from datetime import datetime, timezone

# ====== Telegram settings (put in Render ENV vars) ======
TG_TOKEN = os.getenv("TG_TOKEN", "")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "")

SYMBOL = os.getenv("SYMBOL", "XAUUSD=X")   # Yahoo symbol for XAUUSD spot
INTERVAL = os.getenv("INTERVAL", "15m")    # 15m
RANGE = os.getenv("RANGE", "5d")           # enough candles for EMAs

FAST_EMA = int(os.getenv("FAST_EMA", "20"))
SLOW_EMA = int(os.getenv("SLOW_EMA", "50"))

SLEEP_SECONDS = int(os.getenv("SLEEP_SECONDS", "60"))  # check every minute; send only on new candle

STATE_FILE = "state.json"


def tg_send(message: str) -> bool:
    if not TG_TOKEN or not TG_CHAT_ID:
        print("Telegram env vars missing: TG_TOKEN / TG_CHAT_ID")
        return False
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    r = requests.post(url, data={"chat_id": TG_CHAT_ID, "text": message}, timeout=20)
    ok = (r.status_code == 200)
    if not ok:
        print("Telegram error:", r.status_code, r.text[:300])
    return ok


def ema(values, period: int):
    """Simple EMA. Returns list with same length; leading values may be None until enough data."""
    if period <= 1:
        return values[:]
    k = 2 / (period + 1)
    out = [None] * len(values)
    ema_prev = None
    for i, v in enumerate(values):
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            out[i] = None
            continue
        if ema_prev is None:
            # seed when we first have enough values
            if i + 1 >= period:
                window = [x for x in values[i + 1 - period:i + 1] if x is not None]
                if len(window) == period:
                    ema_prev = sum(window) / period
                    out[i] = ema_prev
                else:
                    out[i] = None
            else:
                out[i] = None
        else:
            ema_prev = (v - ema_prev) * k + ema_prev
            out[i] = ema_prev
    return out


def fetch_yahoo_candles(symbol: str, interval="15m", rng="5d"):
    """
    Fetch candles from Yahoo Finance chart endpoint.
    Returns: timestamps (seconds), closes (float)
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"interval": interval, "range": rng, "includePrePost": "false"}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    result = data.get("chart", {}).get("result", [])
    if not result:
        raise RuntimeError("No Yahoo chart result")

    res = result[0]
    ts = res.get("timestamp", [])
    indicators = res.get("indicators", {}).get("quote", [])
    if not indicators:
        raise RuntimeError("No indicators in Yahoo response")
    closes = indicators[0].get("close", [])
    # filter out None
    clean_ts = []
    clean_cl = []
    for t, c in zip(ts, closes):
        if c is None:
            continue
        clean_ts.append(int(t))
        clean_cl.append(float(c))
    return clean_ts, clean_cl


def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"last_signal": None, "last_candle_ts": None}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f)


def make_signal(ts_list, close_list):
    """
    Strategy: EMA(FAST) vs EMA(SLOW)
    - BUY when fast > slow
    - SELL when fast < slow
    Return: (signal, last_candle_ts, last_price, fast, slow)
    """
    if len(close_list) < max(FAST_EMA, SLOW_EMA) + 5:
        return None, None, None, None, None

    fast = ema(close_list, FAST_EMA)
    slow = ema(close_list, SLOW_EMA)

    # take last index where both exist
    idx = None
    for i in range(len(close_list) - 1, -1, -1):
        if fast[i] is not None and slow[i] is not None:
            idx = i
            break
    if idx is None:
        return None, None, None, None, None

    last_price = close_list[idx]
    last_ts = ts_list[idx]

    if fast[idx] > slow[idx]:
        sig = "BUY"
    elif fast[idx] < slow[idx]:
        sig = "SELL"
    else:
        sig = None

    return sig, last_ts, last_price, fast[idx], slow[idx]


def fmt_time(ts: int):
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def main():
    state = load_state()
    tg_send("✅ Bot started: XAUUSD signals (EMA crossover)")

    while True:
        try:
            ts, cl = fetch_yahoo_candles(SYMBOL, INTERVAL, RANGE)
            sig, candle_ts, price, f, s = make_signal(ts, cl)

            if sig is None or candle_ts is None:
                print("No signal yet / not enough data")
                time.sleep(SLEEP_SECONDS)
                continue

            # Only act on NEW candle timestamp
            if state.get("last_candle_ts") != candle_ts:
                # Send only if signal changed (BUY <-> SELL)
                if state.get("last_signal") != sig:
                    msg = (
                        f"📌 XAUUSD SIGNAL: {sig}\n"
                        f"Price: {price:.2f}\n"
                        f"Time: {fmt_time(candle_ts)}\n"
                        f"EMA{FAST_EMA}: {f:.2f} | EMA{SLOW_EMA}: {s:.2f}"
                    )
                    tg_send(msg)
                    state["last_signal"] = sig

                state["last_candle_ts"] = candle_ts
                save_state(state)

            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            print("Error:", repr(e))
            time.sleep(20)


if __name__ == "__main__":
    main()