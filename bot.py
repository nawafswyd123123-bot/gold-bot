import os
import json
from datetime import datetime, timezone

import requests
import pandas as pd
import yfinance as yf


# =========================
# ENV CONFIG
# =========================
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# جرّب Spot أولاً، إذا فشل جرّب Futures
TICKERS = ["XAUUSD=X", "GC=F"]

INTERVAL = os.getenv("INTERVAL", "15m").strip()      # 15m
PERIOD = os.getenv("PERIOD", "60d").strip()          # 60d كافي لـ EMA200
LOOKBACK_BARS = int(os.getenv("LOOKBACK_BARS", "500"))

EMA_FAST = int(os.getenv("EMA_FAST", "20"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "50"))
EMA_TREND = int(os.getenv("EMA_TREND", "200"))

RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
RSI_BUY = float(os.getenv("RSI_BUY", "55"))
RSI_SELL = float(os.getenv("RSI_SELL", "45"))

# إذا بدك يبعث كل 15 دقيقة حتى HOLD (للتأكد من التلغرام):
SEND_HEARTBEAT = os.getenv("SEND_HEARTBEAT", "false").lower() in ("1", "true", "yes", "y")

STATE_FILE = os.getenv("STATE_FILE", "state.json")


# =========================
# UTIL
# =========================
def now_utc_iso():
    return datetime.now(timezone.utc).isoformat()

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
# TELEGRAM
# =========================
def send_telegram(text: str) -> bool:
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
        if r.status_code != 200:
            print("Telegram error:", r.status_code, r.text[:300])
            return False
        return True
    except Exception as e:
        print("Telegram exception:", str(e))
        return False


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
# DATA FETCH (ROBUST)
# =========================
def normalize_yf_df(df: pd.DataFrame) -> pd.DataFrame:
    # حل مشكلة MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # توحيد أسماء الأعمدة
    df = df.rename(columns=str.title)

    # بعض الأحيان يطلع "Adj Close" بدل "Close"
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(set(df.columns)):
        raise RuntimeError(f"Missing required columns: {list(df.columns)}")

    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    return df

def fetch_data() -> tuple[str, pd.DataFrame]:
    last_err = None

    for ticker in TICKERS:
        try:
            df = yf.download(
                ticker,
                interval=INTERVAL,
                period=PERIOD,
                progress=False,
                auto_adjust=False,
                threads=False
            )
            if df is None or df.empty:
                last_err = f"Empty data for {ticker}"
                continue

            df = normalize_yf_df(df)
            df = df.tail(LOOKBACK_BARS)

            # تأكد في بيانات كفاية لـ EMA200
            if len(df) < (EMA_TREND + 5):
                last_err = f"Not enough bars for {ticker}: got {len(df)}"
                continue

            return ticker, df

        except Exception as e:
            last_err = f"{ticker}: {str(e)}"
            continue

    raise RuntimeError(f"Failed to fetch data. Last error: {last_err}")


# =========================
# STRONG SIGNAL LOGIC
# =========================
def strong_signal(last_row, prev_row):
    close = float(last_row["Close"])
    ema_fast_now = float(last_row["EMA_FAST"])
    ema_slow_now = float(last_row["EMA_SLOW"])
    ema_trend_now = float(last_row["EMA_TREND"])
    rsi_now = float(last_row["RSI"])

    prev_fast = float(prev_row["EMA_FAST"])
    prev_slow = float(prev_row["EMA_SLOW"])

    cross_up = (prev_fast <= prev_slow) and (ema_fast_now > ema_slow_now)
    cross_down = (prev_fast >= prev_slow) and (ema_fast_now < ema_slow_now)

    # BUY قوي
    if close > ema_trend_now and cross_up and rsi_now > RSI_BUY and close > ema_slow_now:
        return "BUY", "Strong: >EMA200 + EMA20 cross up EMA50 + RSI>55 + close>EMA50"

    # SELL قوي
    if close < ema_trend_now and cross_down and rsi_now < RSI_SELL and close < ema_slow_now:
        return "SELL", "Strong: <EMA200 + EMA20 cross down EMA50 + RSI<45 + close<EMA50"

    return "HOLD", "Filtered: not strong"


# =========================
# MAIN
# =========================
def main():
    state = load_state()

    ticker, df = fetch_data()

    # indicators
    close = df["Close"].copy()
    df["EMA_FAST"] = ema(close, EMA_FAST)
    df["EMA_SLOW"] = ema(close, EMA_SLOW)
    df["EMA_TREND"] = ema(close, EMA_TREND)
    df["RSI"] = rsi(close, RSI_PERIOD)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    candle_time = df.index[-1]
    candle_utc = candle_time
    if candle_utc.tzinfo is None:
        candle_utc = candle_utc.tz_localize(timezone.utc, ambiguous="NaT", nonexistent="NaT")
    else:
        candle_utc = candle_utc.astimezone(timezone.utc)

    candle_iso = candle_utc.isoformat()

    sig, reason = strong_signal(last, prev)

    price = float(last["Close"])
    rsi_now = float(last["RSI"])
    ema200 = float(last["EMA_TREND"])
    ema50 = float(last["EMA_SLOW"])

    msg = (
        f"XAU | 15m | {ticker}\n"
        f"time(UTC): {candle_iso}\n"
        f"signal: {sig}\n"
        f"price: {price:.2f}\n"
        f"RSI({RSI_PERIOD}): {rsi_now:.1f}\n"
        f"EMA{EMA_TREND}: {ema200:.2f}\n"
        f"EMA{EMA_SLOW}: {ema50:.2f}\n"
        f"reason: {reason}"
    )

    # logs
    print(msg.replace("\n", " | "))

    # anti-duplicate
    last_candle_sent = state.get("last_candle_time")
    last_signal_sent = state.get("last_signal")

    should_send = False

    if sig in ("BUY", "SELL"):
        # ابعت إذا هاي شمعة جديدة أو إشارة جديدة
        if candle_iso != last_candle_sent or sig != last_signal_sent:
            should_send = True
    else:
        # HOLD فقط إذا بدك Heartbeat
        if SEND_HEARTBEAT and candle_iso != last_candle_sent:
            should_send = True

    if should_send:
        ok = send_telegram(msg)
        if ok:
            state["last_candle_time"] = candle_iso
            state["last_signal"] = sig
            state["last_sent_at_utc"] = now_utc_iso()
            state["last_ticker"] = ticker
            save_state(state)


if __name__ == "__main__":
    main()