import os
import time
import json
import math
import traceback
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import requests
import pandas as pd
import yfinance as yf


# ============================================================
# ENV CONFIG (Render Environment Variables)
# ============================================================
BOT_TOKEN = (os.getenv("BOT_TOKEN") or "").strip()
CHAT_ID = (os.getenv("CHAT_ID") or "").strip()

SYMBOL = os.getenv("SYMBOL", "GC=F").strip()

ENTRY_INTERVAL = os.getenv("ENTRY_INTERVAL", "15m").strip()   # entry timeframe
ENTRY_PERIOD = os.getenv("ENTRY_PERIOD", "5d").strip()

TREND_INTERVAL = os.getenv("TREND_INTERVAL", "1h").strip()    # trend timeframe
TREND_PERIOD = os.getenv("TREND_PERIOD", "30d").strip()

SLEEP_SECONDS = int(os.getenv("SLEEP_SECONDS", "600"))        # main loop sleep
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "1800")) # min time between alerts

# Signal strength tuning
RSI_LEN = int(os.getenv("RSI_LEN", "14"))
RSI_BUY = float(os.getenv("RSI_BUY", "35"))    # buy if RSI <= this (when trend up)
RSI_SELL = float(os.getenv("RSI_SELL", "65"))  # sell if RSI >= this (when trend down)

FAST_MA_LEN = int(os.getenv("FAST_MA_LEN", "20"))
SLOW_MA_LEN = int(os.getenv("SLOW_MA_LEN", "50"))

# ATR / risk suggestions
ATR_LEN = int(os.getenv("ATR_LEN", "14"))
MIN_ATR = float(os.getenv("MIN_ATR", "0"))           # set later if you want stricter (e.g. 5 or 10 depending on broker units)
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "1.5"))
TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "2.0"))

# Caching to avoid Yahoo rate limit
ENTRY_REFRESH_SECONDS = int(os.getenv("ENTRY_REFRESH_SECONDS", "900"))  # refresh entry data at most every 15 min
TREND_REFRESH_SECONDS = int(os.getenv("TREND_REFRESH_SECONDS", "1800")) # refresh trend data at most every 30 min

# Startup ping
SEND_STARTUP_MESSAGE = (os.getenv("SEND_STARTUP_MESSAGE", "1").strip() == "1")

# State file (Render disk is ephemeral across rebuilds, but good for runtime restarts)
STATE_FILE = os.getenv("STATE_FILE", "state.json").strip()


# ============================================================
# TELEGRAM
# ============================================================
def send_telegram(text: str) -> None:
    if not BOT_TOKEN or not CHAT_ID:
        print("❌ Missing BOT_TOKEN or CHAT_ID in env.")
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}

    try:
        r = requests.post(url, data=payload, timeout=20)
        if r.status_code != 200:
            print("❌ Telegram error:", r.status_code, r.text[:200])
    except Exception as e:
        print("❌ Telegram exception:", e)


# ============================================================
# INDICATORS
# ============================================================
def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean()

    rs = avg_gain / avg_loss.replace(0, math.nan)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    return tr.rolling(length).mean()


# ============================================================
# DATA FETCH (with caching + rate-limit protection)
# ============================================================
@dataclass
class CacheItem:
    df: pd.DataFrame
    fetched_at: float

_cache: Dict[str, CacheItem] = {}

def _yfinance_download(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """
    Wrapper around yf.download with safe defaults.
    """
    try:
        df = yf.download(
            symbol,
            interval=interval,
            period=period,
            progress=False,
            threads=False,
        )
    except Exception as e:
        print(f"❌ yfinance exception ({symbol} {interval}):", e)
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # Fix MultiIndex columns if needed
    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(-1)

    needed = {"Open", "High", "Low", "Close"}
    if not needed.issubset(set(df.columns)):
        return pd.DataFrame()

    return df.dropna()

def fetch_cached(symbol: str, interval: str, period: str, max_age_seconds: int) -> pd.DataFrame:
    key = f"{symbol}|{interval}|{period}"
    now = time.time()
    item = _cache.get(key)

    if item and (now - item.fetched_at) <= max_age_seconds and not item.df.empty:
        return item.df

    df = _yfinance_download(symbol, interval, period)
    if not df.empty:
        _cache[key] = CacheItem(df=df, fetched_at=now)
    return df


# ============================================================
# STATE (anti-spam + survive restarts)
# ============================================================
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


# ============================================================
# STRATEGY (Pro Mode)
# ============================================================
def compute_trend(trend_df: pd.DataFrame) -> Optional[str]:
    """
    Determine macro trend from 1H:
    - UP if EMA(20) > EMA(50) and last close above EMA(50)
    - DOWN if EMA(20) < EMA(50) and last close below EMA(50)
    else: None (no-trade)
    """
    close = trend_df["Close"].astype(float)
    e20 = ema(close, FAST_MA_LEN)
    e50 = ema(close, SLOW_MA_LEN)

    if e50.isna().all():
        return None

    last_close = float(close.iloc[-1])
    last_e20 = float(e20.iloc[-1])
    last_e50 = float(e50.iloc[-1])

    if last_e20 > last_e50 and last_close >= last_e50:
        return "UP"
    if last_e20 < last_e50 and last_close <= last_e50:
        return "DOWN"
    return None

def make_entry_signal(entry_df: pd.DataFrame, trend: str) -> Optional[Dict[str, Any]]:
    """
    Entry on 15m only if macro trend agrees.
    Conditions (strong & stable):
    - Trend UP: price above SMA50 AND SMA20 > SMA50 AND RSI <= RSI_BUY
    - Trend DOWN: price below SMA50 AND SMA20 < SMA50 AND RSI >= RSI_SELL
    Plus ATR filter (optional via MIN_ATR)
    """
    close = entry_df["Close"].astype(float)
    s20 = sma(close, FAST_MA_LEN)
    s50 = sma(close, SLOW_MA_LEN)
    r = rsi(close, RSI_LEN)
    a = atr(entry_df, ATR_LEN)

    # need enough history
    if s50.isna().all() or r.isna().all() or a.isna().all():
        return None

    last_close = float(close.iloc[-1])
    last_s20 = float(s20.iloc[-1])
    last_s50 = float(s50.iloc[-1])
    last_rsi = float(r.iloc[-1])
    last_atr = float(a.iloc[-1])

    if last_atr < MIN_ATR:
        return None

    # Proposed risk levels
    def build(side: str) -> Dict[str, Any]:
        sl = last_close - (SL_ATR_MULT * last_atr) if side == "BUY" else last_close + (SL_ATR_MULT * last_atr)
        tp = last_close + (TP_ATR_MULT * last_atr) if side == "BUY" else last_close - (TP_ATR_MULT * last_atr)

        return {
            "side": side,
            "price": last_close,
            "rsi": last_rsi,
            "atr": last_atr,
            "sma20": last_s20,
            "sma50": last_s50,
            "tp": tp,
            "sl": sl,
        }

    if trend == "UP":
        if last_close > last_s50 and last_s20 > last_s50 and last_rsi <= RSI_BUY:
            return build("BUY")

    if trend == "DOWN":
        if last_close < last_s50 and last_s20 < last_s50 and last_rsi >= RSI_SELL:
            return build("SELL")

    return None


# ============================================================
# MAIN LOOP
# ============================================================
def main():
    state = load_state()
    last_candle_ts = state.get("last_candle_ts")
    last_signal = state.get("last_signal")            # "BUY"/"SELL"
    last_sent_at = float(state.get("last_sent_at", 0))

    # backoff for rate limit / errors
    error_backoff = 0  # seconds

    if SEND_STARTUP_MESSAGE:
        send_telegram("✅ Gold Signal Bot started successfully (PRO MODE)")

    print("✅ Running PRO MODE")
    print("Symbol:", SYMBOL, "| Entry:", ENTRY_INTERVAL, "| Trend:", TREND_INTERVAL)
    print("Sleep:", SLEEP_SECONDS, "| Cooldown:", COOLDOWN_SECONDS)

    while True:
        try:
            # If we recently had errors/rate limit, wait a bit more
            if error_backoff > 0:
                print(f"⏳ Backoff sleeping {error_backoff}s")
                time.sleep(error_backoff)

            # Fetch trend data (cached)
            trend_df = fetch_cached(SYMBOL, TREND_INTERVAL, TREND_PERIOD, TREND_REFRESH_SECONDS)
            if trend_df.empty:
                print("⚠️ Trend df empty; skipping cycle.")
                error_backoff = min(1800, max(120, error_backoff * 2) if error_backoff else 120)
                time.sleep(SLEEP_SECONDS)
                continue

            trend = compute_trend(trend_df)
            if trend is None:
                print("ℹ️ No clear trend (NO-TRADE).")
                error_backoff = 0
                time.sleep(SLEEP_SECONDS)
                continue

            # Fetch entry data (cached)
            entry_df = fetch_cached(SYMBOL, ENTRY_INTERVAL, ENTRY_PERIOD, ENTRY_REFRESH_SECONDS)
            if entry_df.empty:
                print("⚠️ Entry df empty; skipping cycle.")
                error_backoff = min(1800, max(120, error_backoff * 2) if error_backoff else 120)
                time.sleep(SLEEP_SECONDS)
                continue

            current_ts = str(entry_df.index[-1])

            # Only act on new candle
            if current_ts != last_candle_ts:
                sig = make_entry_signal(entry_df, trend)

                if sig:
                    now = time.time()
                    cooldown_ok = (now - last_sent_at) >= COOLDOWN_SECONDS
                    not_repeat = (sig["side"] != last_signal)

                    if cooldown_ok and not_repeat:
                        msg = (
                            f"📌 GOLD PRO SIGNAL ({ENTRY_INTERVAL})\n"
                            f"Symbol: {SYMBOL}\n"
                            f"Trend({TREND_INTERVAL}): {trend}\n"
                            f"Signal: {sig['side']} (STRONG)\n"
                            f"Price: {sig['price']:.2f}\n"
                            f"SMA{FAST_MA_LEN}: {sig['sma20']:.2f} | SMA{SLOW_MA_LEN}: {sig['sma50']:.2f}\n"
                            f"RSI({RSI_LEN}): {sig['rsi']:.1f} | ATR({ATR_LEN}): {sig['atr']:.2f}\n"
                            f"TP: {sig['tp']:.2f}\n"
                            f"SL: {sig['sl']:.2f}\n"
                            f"Time: {current_ts}"
                        )
                        send_telegram(msg)

                        last_signal = sig["side"]
                        last_sent_at = now
                        print("✅ Sent signal:", last_signal, "at", current_ts)
                    else:
                        print("ℹ️ Signal ignored (cooldown/repeat):", sig["side"], current_ts)

                last_candle_ts = current_ts

                # persist state
                state = {
                    "last_candle_ts": last_candle_ts,
                    "last_signal": last_signal,
                    "last_sent_at": last_sent_at,
                }
                save_state(state)

            error_backoff = 0
            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            # Generic catch: increase backoff to protect Yahoo and stabilize
            print("❌ Loop error:", e)
            traceback.print_exc()

            # Heuristic: if it looks like a rate limit, back off stronger
            msg = str(e).lower()
            if "rate" in msg or "too many" in msg or "ratelimit" in msg:
                error_backoff = min(3600, max(600, error_backoff * 2) if error_backoff else 600)  # start at 10 min
            else:
                error_backoff = min(1800, max(180, error_backoff * 2) if error_backoff else 180)

            time.sleep(max(300, SLEEP_SECONDS))


if __name__ == "__main__":
    main()