import os
import json
from datetime import datetime, timezone

import requests
import pandas as pd
import yfinance as yf


# =========================
# SETTINGS (ENV)
# =========================
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

INTERVAL = "15m"
PERIOD = "60d"
STATE_FILE = "state.json"

# Tickers fallback
TICKERS = ["XAUUSD=X", "GC=F"]

# Strong signal parameters
EMA_FAST = 20
EMA_SLOW = 50
EMA_TREND = 200
RSI_PERIOD = 14
RSI_BUY = 55
RSI_SELL = 45

# If true: send message every candle (even HOLD) to prove Telegram works
SEND_HEARTBEAT = os.getenv("SEND_HEARTBEAT", "false").lower() in ("1", "true", "yes", "y")


# =========================
# STATE
# =========================
def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(state):
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
    payload = {"chat_id": CHAT_ID, "text": text, "disable_web_page_preview": True}

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
# DATA FETCH (VERY ROBUST)
# =========================
def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    # If MultiIndex like ('Close','GC=F') -> keep first level
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Sometimes it is tuples even without pd.MultiIndex
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.columns = [str(c).strip() for c in df.columns]
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
                threads=False,
                group_by="column",
            )

            if df is None or df.empty:
                last_err = f"Empty data for {ticker}"
                continue

            df = _flatten_columns(df)

            # Map typical columns
            if "Close" not in df.columns and "Adj Close" in df.columns:
                df["Close"] = df["Adj Close"]

            required = {"Open", "High", "Low", "Close"}
            if not required.issubset(set(df.columns)):
                last_err = f"Missing columns for {ticker}: {list(df.columns)}"
                continue

            df = df[["Open", "High", "Low", "Close"]].dropna()
            df.index = pd.to_datetime(df.index)

            # Need enough bars for EMA200
            if len(df) < (EMA_TREND + 10):
                last_err = f"Not enough bars for {ticker}: {len(df)}"
                continue

            return ticker, df

        except Exception as e:
            last_err = f"{ticker}: {str(e)}"
            continue

    raise RuntimeError(f"Failed to fetch data. Last error: {last_err}")


# =========================
# STRONG SIGNAL
# =========================
def compute_signal(df: pd.DataFrame):
    close = df["Close"].copy()

    df["EMA_FAST"] = ema(close, EMA_FAST)
    df["EMA_SLOW"] = ema(close, EMA_SLOW)
    df["EMA_TREND"] = ema(close, EMA_TREND)
    df["RSI"] = rsi(close, RSI_PERIOD)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    close_now = float(last["Close"])
    ema_fast_now = float(last["EMA_FAST"])
    ema_slow_now = float(last["EMA_SLOW"])
    ema_trend_now = float(last["EMA_TREND"])
    rsi_now = float(last["RSI"])

    prev_fast = float(prev["EMA_FAST"])
    prev_slow = float(prev["EMA_SLOW"])

    cross_up = (prev_fast <= prev_slow) and (ema_fast_now > ema_slow_now)
    cross_down = (prev_fast >= prev_slow) and (ema_fast_now < ema_slow_now)

    # Strong BUY
    if close_now > ema_trend_now and cross_up and rsi_now > RSI_BUY and close_now > ema_slow_now:
        return "BUY", "Strong: >EMA200 + EMA20 cross up EMA50 + RSI>55 + close>EMA50", last

    # Strong SELL
    if close_now < ema_trend_now and cross_down and rsi_now < RSI_SELL and close_now < ema_slow_now:
        return "SELL", "Strong: <EMA200 + EMA20 cross down EMA50 + RSI<45 + close<EMA50", last

    return "HOLD", "Filtered: not strong", last


# =========================
# MAIN
# =========================
def main():
    state = load_state()

    ticker, df = fetch_data()
    sig, reason, last = compute_signal(df)

    candle_time = df.index[-1]
    if candle_time.tzinfo is None:
        candle_time = candle_time.tz_localize(timezone.utc, ambiguous="NaT", nonexistent="NaT")
    else:
        candle_time = candle_time.astimezone(timezone.utc)

    candle_iso = candle_time.isoformat()

    price = float(last["Close"])
    # we may not always have indicators if data weird - but here we do
    rsi_now = float(df["RSI"].iloc[-1])
    ema200 = float(df["EMA_TREND"].iloc[-1])
    ema50 = float(df["EMA_SLOW"].iloc[-1])

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

    # Logs
    print(msg.replace("\n", " | "))

    # Anti-duplicate per candle
    last_candle_sent = state.get("last_candle_time")
    last_signal_sent = state.get("last_signal")

    should_send = False
    if sig in ("BUY", "SELL"):
        if candle_iso != last_candle_sent or sig != last_signal_sent:
            should_send = True
    else:
        if SEND_HEARTBEAT and candle_iso != last_candle_sent:
            should_send = True

    if should_send:
        ok = send_telegram(msg)
        if ok:
            state["last_candle_time"] = candle_iso
            state["last_signal"] = sig
            state["last_sent_at_utc"] = datetime.now(timezone.utc).isoformat()
            save_state(state)
send_telegram("TEST MESSAGE ✅")

if __name__ == "__main__":
    main()