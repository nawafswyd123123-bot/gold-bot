import os
import time
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf


# =========================
# CONFIG
# =========================
SYMBOL = os.getenv("SYMBOL", "GC=F")  # Gold Futures on Yahoo Finance
INTERVAL_PRIMARY = os.getenv("INTERVAL", "15m")
PERIOD_PRIMARY = os.getenv("PERIOD", "5d")

# Fallbacks if Yahoo returns empty
FALLBACKS = [
    ("15m", "1d"),
    ("30m", "5d"),
    ("30m", "1mo"),
    ("60m", "3mo"),
]

CYCLE_SECONDS = int(os.getenv("CYCLE_SECONDS", "600"))   # 10 minutes
EMPTY_BACKOFF_SECONDS = int(os.getenv("EMPTY_BACKOFF_SECONDS", "300"))  # 5 minutes (بدل 1800)
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "15"))

STATE_FILE = os.getenv("STATE_FILE", "state.json")

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()


# =========================
# TELEGRAM
# =========================
def tg_send(text: str) -> None:
    if not TOKEN or not CHAT_ID:
        print("⚠️ Telegram env vars missing (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID). Message not sent.")
        print("MSG:", text)
        return

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            print("⚠️ Telegram send failed:", r.status_code, r.text[:500])
    except Exception as e:
        print("⚠️ Telegram exception:", e)


# =========================
# STATE (anti-spam + survive restarts)
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
# CACHE
# =========================
@dataclass
class CacheItem:
    df: pd.DataFrame
    fetched_at: float

_cache: Dict[str, CacheItem] = {}
MAX_CACHE_AGE = 60  # seconds

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Fix MultiIndex columns + ensure OHLC exists + drop NaNs."""
    if df is None or len(df) == 0:
        return pd.DataFrame()

    # Some yfinance returns multiindex columns
    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(-1)

    needed = {"Open", "High", "Low", "Close"}
    if not needed.issubset(set(df.columns)):
        return pd.DataFrame()

    df = df.copy()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df

def _yfinance_download(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Robust download: retries + normalization + clear logs."""
    for attempt in range(1, 4):
        try:
            df = yf.download(
                symbol,
                interval=interval,
                period=period,
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            df = _normalize_df(df)
            print(f"📥 fetch {symbol} interval={interval} period={period} attempt={attempt} rows={len(df)}")
            if not df.empty:
                return df
        except Exception as e:
            print(f"⚠️ yfinance error attempt={attempt}:", e)
        time.sleep(1.2 * attempt)

    return pd.DataFrame()

def fetch_cached(symbol: str, interval: str, period: str) -> pd.DataFrame:
    key = f"{symbol}|{interval}|{period}"
    now = time.time()
    item = _cache.get(key)
    if item and (now - item.fetched_at) <= MAX_CACHE_AGE:
        return item.df

    df = _yfinance_download(symbol, interval, period)
    if not df.empty:
        _cache[key] = CacheItem(df=df, fetched_at=now)
    return df

def fetch_market_data(symbol: str) -> Tuple[pd.DataFrame, str]:
    """
    Try primary, then fallbacks until we get rows.
    Returns (df, info_string).
    """
    # 1) Primary
    df = fetch_cached(symbol, INTERVAL_PRIMARY, PERIOD_PRIMARY)
    if not df.empty:
        return df, f"{INTERVAL_PRIMARY}/{PERIOD_PRIMARY}"

    # 2) Fallbacks
    for interval, period in FALLBACKS:
        df2 = fetch_cached(symbol, interval, period)
        if not df2.empty:
            return df2, f"{interval}/{period}"

    return pd.DataFrame(), "EMPTY"


# =========================
# SIMPLE TREND SIGNAL
# (EMA cross + basic filters)
# =========================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def compute_signal(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns dict:
    - signal: "BUY" | "SELL" | "HOLD"
    - price
    - reason
    """
    close = df["Close"]
    if len(close) < 60:
        return {"signal": "HOLD", "price": float(close.iloc[-1]), "reason": "Not enough candles"}

    fast = ema(close, 12)
    slow = ema(close, 26)

    price = float(close.iloc[-1])
    prev_fast, prev_slow = float(fast.iloc[-2]), float(slow.iloc[-2])
    curr_fast, curr_slow = float(fast.iloc[-1]), float(slow.iloc[-1])

    # Cross logic
    if prev_fast <= prev_slow and curr_fast > curr_slow:
        return {"signal": "BUY", "price": price, "reason": "EMA12 crossed above EMA26"}
    if prev_fast >= prev_slow and curr_fast < curr_slow:
        return {"signal": "SELL", "price": price, "reason": "EMA12 crossed below EMA26"}

    return {"signal": "HOLD", "price": price, "reason": "No cross"}


# =========================
# MAIN LOOP
# =========================
def main():
    state = load_state()
    last_sent = state.get("last_sent", {})  # e.g. {"BUY": timestamp, "SELL": timestamp}
    last_signal = state.get("last_signal", "NONE")

    tg_send("✅ Bot started.")

    while True:
        try:
            df, used = fetch_market_data(SYMBOL)

            if df.empty:
                print("⚠️ Trend df empty; skipping cycle.")
                print(f"⏳ Backoff sleeping {EMPTY_BACKOFF_SECONDS}s")
                time.sleep(EMPTY_BACKOFF_SECONDS)
                continue

            sig = compute_signal(df)
            signal = sig["signal"]
            price = sig["price"]
            reason = sig["reason"]

            now = time.time()

            # Anti-spam:
            # - send only when BUY/SELL
            # - and only if changed from last_signal
            # - and cooldown per direction (default 30 min)
            cooldown = int(os.getenv("SIGNAL_COOLDOWN_SECONDS", "1800"))

            should_send = signal in ("BUY", "SELL") and signal != last_signal
            last_time_for_dir = float(last_sent.get(signal, 0))
            if should_send and (now - last_time_for_dir) < cooldown:
                should_send = False
                print(f"🛑 Cooldown active for {signal}. remaining={int(cooldown - (now - last_time_for_dir))}s")

            print(f"📊 signal={signal} price={price} used={used} reason={reason}")

            if should_send:
                msg = (
                    f"📌 Gold Signal ({SYMBOL})\n"
                    f"Signal: {signal}\n"
                    f"Price: {price:.2f}\n"
                    f"Data: {used}\n"
                    f"Reason: {reason}"
                )
                tg_send(msg)

                last_signal = signal
                last_sent[signal] = now
                state["last_signal"] = last_signal
                state["last_sent"] = last_sent
                save_state(state)

            time.sleep(CYCLE_SECONDS)

        except Exception as e:
            print("🔥 Loop error:", e)
            # short backoff
            time.sleep(10)


if __name__ == "__main__":
    main()