import os
import json
import time
from datetime import datetime, timezone

import requests
import pandas as pd

# يفضّل تثبيت yfinance
import yfinance as yf


# =========================
# CONFIG (ENV VARS)
# =========================
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# Yahoo tickers (جرّب بالترتيب)
# XAUUSD=X عادة يعطي Spot Gold بالدولار
# GC=F عقود ذهب (قد يختلف قليل)
TICKERS = [
    os.getenv("SYMBOL", "").strip() or "XAUUSD=X",
    "GC=F",
]

INTERVAL = os.getenv("INTERVAL", "15m").strip()  # لازم 15m
LOOKBACK = int(os.getenv("LOOKBACK_BARS", "400"))  # عدد شموع لجلب البيانات

EMA_FAST = int(os.getenv("EMA_FAST", "20"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "50"))
EMA_TREND = int(os.getenv("EMA_TREND", "200"))

RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
RSI_BUY = float(os.getenv("RSI_BUY", "55"))
RSI_SELL = float(os.getenv("RSI_SELL", "45"))

# إذا بدك يبعث HOLD كل ربع ساعة فعّلها:
SEND_HOLD = os.getenv("SEND_HOLD", "false").lower() in ("1", "true", "yes", "y")

# منع التكرار: يبعت مرة واحدة لكل شمعة + لكل إشارة
STATE_FILE = os.getenv("STATE_FILE", "state.json")


# =========================
# INDICATORS
# =========================
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / (avg_loss.replace(0, 1e-12))
    return 100 - (100 / (1 + rs))


# =========================
# SIGNAL LOGIC (STRONG)
# =========================
def strong_signal(close, ema_fast, ema_slow, ema_trend, rsi_val, prev_ema_fast, prev_ema_slow):
    cross_up = (prev_ema_fast <= prev_ema_slow) and (ema_fast > ema_slow)
    cross_down = (prev_ema_fast >= prev_ema_slow) and (ema_fast < ema_slow)

    # BUY قوي
    if close > ema_trend and cross_up and rsi_val > RSI_BUY and close > ema_slow:
        return "BUY", "Strong: trend(>EMA200)+cross(20>50)+RSI>55+close>EMA50"

    # SELL قوي
    if close < ema_trend and cross_down and rsi_val < RSI_SELL and close < ema_slow:
        return "SELL", "Strong: trend(<EMA200)+cross(20<50)+RSI<45+close<EMA50"

    return "HOLD", "Filtered: not strong"


# =========================
# TELEGRAM
# =========================
def send_telegram(text: str):
    if not TOKEN or not CHAT_ID:
        print("ERROR: Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
        return False

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "disable_web_page_preview": True
    }
    try:
        r = requests.post(url, json=payload, timeout=20)
        ok = (r.status_code == 200)
        if not ok:
            print("Telegram error:", r.status_code, r.text[:300])
        return ok
    except Exception as e:
        print("Telegram exception:", str(e))
        return False


# =========================
# STATE (ANTI-DUPLICATE)
# =========================
def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(state: dict):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("State save error:", str(e))


# =========================
# DATA FETCH
# =========================
def fetch_data():
    last_err = None
    for t in TICKERS:
        if not t:
            continue
        try:
            df = yf.download(
                t,
                interval=INTERVAL,
                period="60d",     # كفاية لـ EMA200 على 15m
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            if df is None or df.empty:
                last_err = f"Empty data for {t}"
                continue

            # توحيد الأعمدة
            df = df.rename(columns=str.title)
            needed = {"Open", "High", "Low", "Close"}
            if not needed.issubset(set(df.columns)):
                last_err = f"Missing columns for {t}: {df.columns}"
                continue

            df = df.dropna().tail(LOOKBACK)
            df.index = pd.to_datetime(df.index)

            return t, df
        except Exception as e:
            last_err = str(e)
            continue

    raise RuntimeError(f"Failed to fetch data. Last error: {last_err}")


# =========================
# MAIN
# =========================
def main():
    state = load_state()

    symbol, df = fetch_data()

    close = df["Close"].copy()
    df["EMA_FAST"] = ema(close, EMA_FAST)
    df["EMA_SLOW"] = ema(close, EMA_SLOW)
    df["EMA_TREND"] = ema(close, EMA_TREND)
    df["RSI"] = rsi(close, RSI_PERIOD)

    # نستخدم آخر شمعتين (آخر إغلاق + اللي قبلها)
    last = df.iloc[-1]
    prev = df.iloc[-2]

    candle_time = df.index[-1]
    # صيغة ثابتة
    candle_time_iso = candle_time.tz_localize(timezone.utc, ambiguous="NaT", nonexistent="NaT").isoformat() \
        if candle_time.tzinfo is None else candle_time.astimezone(timezone.utc).isoformat()

    sig, reason = strong_signal(
        close=float(last["Close"]),
        ema_fast=float(last["EMA_FAST"]),
        ema_slow=float(last["EMA_SLOW"]),
        ema_trend=float(last["EMA_TREND"]),
        rsi_val=float(last["RSI"]),
        prev_ema_fast=float(prev["EMA_FAST"]),
        prev_ema_slow=float(prev["EMA_SLOW"]),
    )

    price = float(last["Close"])
    rsi_now = float(last["RSI"])
    ema200 = float(last["EMA_TREND"])
    ema50 = float(last["EMA_SLOW"])

    # منع التكرار: نفس الشمعة ونفس الإشارة -> لا تبعث
    last_sent_candle = state.get("last_candle_time")
    last_sent_signal = state.get("last_signal")

    should_send = False
    if sig in ("BUY", "SELL"):
        if (candle_time_iso != last_sent_candle) or (sig != last_sent_signal):
            should_send = True
    else:
        # HOLD فقط إذا المستخدم فعّل SEND_HOLD
        if SEND_HOLD and (candle_time_iso != last_sent_candle):
            should_send = True

    # رسالة
    msg = (
        f"XAU 15m | {symbol}\n"
        f"time(UTC): {candle_time_iso}\n"
        f"signal: {sig}\n"
        f"price: {price:.2f}\n"
        f"RSI({RSI_PERIOD}): {rsi_now:.1f}\n"
        f"EMA{EMA_TREND}: {ema200:.2f}\n"
        f"EMA{EMA_SLOW}: {ema50:.2f}\n"
        f"reason: {reason}"
    )

    # طباعة للـ logs
    print(msg.replace("\n", " | "))

    if should_send:
        ok = send_telegram(msg)
        if ok:
            state["last_candle_time"] = candle_time_iso
            state["last_signal"] = sig
            state["last_sent_at_utc"] = datetime.now(timezone.utc).isoformat()
            save_state(state)


if __name__ == "__main__":
    main()