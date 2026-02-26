import os
import time
import math
from datetime import datetime, timezone

import requests
import pandas as pd
import yfinance as yf

# =========================
# ENV
# =========================
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

SYMBOL = os.getenv("SYMBOL", "GC=F")       # Gold futures on Yahoo
INTERVAL = os.getenv("INTERVAL", "15m")
PERIOD = os.getenv("PERIOD", "5d")

SLEEP_SECONDS = int(os.getenv("SLEEP_SECONDS", "180"))  # 3 minutes
LOOKBACK_SWEEP = int(os.getenv("LOOKBACK_SWEEP", "20")) # lookback for liquidity sweep
ATR_MULT_STRONG = float(os.getenv("ATR_MULT_STRONG", "1.2"))
ATR_MULT_EVENT = float(os.getenv("ATR_MULT_EVENT", "2.0"))
VOL_Z_EVENT = float(os.getenv("VOL_Z_EVENT", "1.2"))

# Optional: real news integration (leave empty if you don't have it)
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "").strip()
NEWS_RISK_MINUTES = int(os.getenv("NEWS_RISK_MINUTES", "0"))  # if you later integrate calendar

# =========================
# STATE (anti-repeat)
# =========================
last_candle_ts = None
last_signal_sent = None
last_alert_ts = None

# =========================
# HELPERS
# =========================
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _safe_float(x, default=None):
    try:
        if x is None:
            return default
        if isinstance(x, (float, int)):
            return float(x)
        return float(x)
    except Exception:
        return default

def send_telegram(text: str):
    if not BOT_TOKEN or not CHAT_ID:
        print("Missing BOT_TOKEN or CHAT_ID in Environment Variables")
        return
    if not text:
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}

    try:
        r = requests.post(url, data=payload, timeout=15)
        if r.status_code != 200:
            print("Telegram error:", r.status_code, r.text[:200])
    except Exception as e:
        print("Telegram exception:", e)

def fetch_gold() -> pd.DataFrame:
    df = yf.download(SYMBOL, interval=INTERVAL, period=PERIOD, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()

    # Fix MultiIndex columns if any
    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(0)

    # Ensure required columns exist
    for col in ["Open", "High", "Low", "Close"]:
        if col not in df.columns:
            return pd.DataFrame()

    # Sometimes index is timezone-naive; keep as is but stable stringify/compare
    return df.dropna()

def rsi14(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean()
    rs = avg_gain / avg_loss.replace(0, math.nan)
    return 100 - (100 / (1 + rs))

def atr14(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def zscore(series: pd.Series, length: int = 30) -> pd.Series:
    mu = series.rolling(length).mean()
    sd = series.rolling(length).std()
    return (series - mu) / sd.replace(0, math.nan)

# =========================
# SMART DETECTORS
# =========================
def detect_liquidity_sweep(df: pd.DataFrame, lookback: int = 20):
    """
    Sweep up = wick above previous highs then close back below that zone -> bearish
    Sweep down = wick below previous lows then close back above -> bullish
    """
    if len(df) < lookback + 5:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]
    window = df.iloc[-(lookback + 1):-1]

    prev_high = window["High"].max()
    prev_low = window["Low"].min()

    # using last candle
    sweep_up = (last["High"] > prev_high) and (last["Close"] < prev_high)
    sweep_down = (last["Low"] < prev_low) and (last["Close"] > prev_low)

    if sweep_up:
        return ("SWEEP_UP", prev_high, prev_low)
    if sweep_down:
        return ("SWEEP_DOWN", prev_high, prev_low)
    return None

def detect_strong_candle(df: pd.DataFrame, atr: float):
    """
    Strong candle: range > ATR_MULT_STRONG * ATR and close near extreme
    """
    if len(df) < 5 or atr is None or atr == 0:
        return None

    last = df.iloc[-1]
    rng = float(last["High"] - last["Low"])
    body = abs(float(last["Close"] - last["Open"]))

    if rng < ATR_MULT_STRONG * atr:
        return None

    # close near high/low (strong directional)
    close_pos = (last["Close"] - last["Low"]) / (rng if rng else 1.0)  # 0..1
    if close_pos > 0.8:
        return "BULL_STRONG"
    if close_pos < 0.2:
        return "BEAR_STRONG"
    return "STRONG"

def detect_fake_breakout(df: pd.DataFrame, lookback: int = 20):
    """
    Fake breakout idea:
    - last candle breaks above recent high but closes back inside => already covered by sweep_up
    - OR last closes above, then next candle closes back inside (needs 2 candles)
    We'll implement a simple 2-candle trap:
    prev closes above prev_high, and last closes back below prev_high => bull trap (bearish)
    prev closes below prev_low, and last closes back above prev_low => bear trap (bullish)
    """
    if len(df) < lookback + 6:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]
    window = df.iloc[-(lookback + 2):-2]
    prev_high = window["High"].max()
    prev_low = window["Low"].min()

    bull_trap = (prev["Close"] > prev_high) and (last["Close"] < prev_high)
    bear_trap = (prev["Close"] < prev_low) and (last["Close"] > prev_low)

    if bull_trap:
        return ("BULL_TRAP", prev_high, prev_low)
    if bear_trap:
        return ("BEAR_TRAP", prev_high, prev_low)
    return None

def detect_volatility_event(df: pd.DataFrame, atr: float):
    """
    Proxy for "news/bank intervention": volatility spike
    """
    if len(df) < 35 or atr is None or atr == 0:
        return None

    last = df.iloc[-1]
    rng = float(last["High"] - last["Low"])

    vol_spike = None
    if "Volume" in df.columns:
        vz = zscore(df["Volume"], 30).iloc[-1]
        vol_spike = _safe_float(vz, 0)

    big_move = rng >= (ATR_MULT_EVENT * atr)
    volume_ok = (vol_spike is not None and vol_spike >= VOL_Z_EVENT)

    if big_move and (("Volume" not in df.columns) or volume_ok):
        return ("VOL_SPIKE", rng, atr, vol_spike)
    return None

def news_risk_flag():
    """
    Real news needs an API/calendar.
    For now: if you later add it, return True around scheduled events.
    """
    if not NEWS_API_KEY or NEWS_RISK_MINUTES <= 0:
        return False
    # Placeholder: you can integrate a calendar later
    return False

# =========================
# SIGNAL LOGIC
# =========================
def make_signal(df: pd.DataFrame):
    if df is None or df.empty or len(df) < 60:
        return None

    close = df["Close"]
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    rsi = rsi14(close, 14)
    atr = atr14(df, 14)

    price = float(close.iloc[-1])
    last_sma20 = float(sma20.iloc[-1])
    last_sma50 = float(sma50.iloc[-1])
    last_rsi = float(rsi.iloc[-1])
    last_atr = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else None

    # ===== patterns =====
    sweep = detect_liquidity_sweep(df, LOOKBACK_SWEEP)
    trap = detect_fake_breakout(df, LOOKBACK_SWEEP)
    strong = detect_strong_candle(df, last_atr)
    event = detect_volatility_event(df, last_atr)
    news_flag = news_risk_flag()

    # ===== base signal =====
    signal = None
    reasons = []

    trend_up = last_sma20 > last_sma50
    trend_down = last_sma20 < last_sma50

    # main trend signal (basic)
    if trend_up and last_rsi > 52:
        signal = "BUY"
        reasons.append("TrendUp + RSI>52")
    elif trend_down and last_rsi < 48:
        signal = "SELL"
        reasons.append("TrendDown + RSI<48")

    # override / boost by traps & sweeps
    if sweep:
        if sweep[0] == "SWEEP_UP":
            signal = "SELL"
            reasons.append("Liquidity Sweep UP (stop hunt)")
        elif sweep[0] == "SWEEP_DOWN":
            signal = "BUY"
            reasons.append("Liquidity Sweep DOWN (stop hunt)")

    if trap:
        if trap[0] == "BULL_TRAP":
            signal = "SELL"
            reasons.append("Bull Trap (fake breakout)")
        elif trap[0] == "BEAR_TRAP":
            signal = "BUY"
            reasons.append("Bear Trap (fake breakdown)")

    # strong candle confirmation
    if strong == "BULL_STRONG":
        reasons.append("Strong Bull Candle")
    elif strong == "BEAR_STRONG":
        reasons.append("Strong Bear Candle")

    # If still no signal -> don't send WAIT
    if not signal:
        return None

    # ===== confidence scoring (0..95) =====
    # Distance between SMAs + RSI distance from 50 + confirmations
    trend_strength = abs(last_sma20 - last_sma50)
    rsi_strength = abs(last_rsi - 50)

    confidence = 50.0
    confidence += min(25.0, trend_strength * 2.0)     # depends on price scale, but capped
    confidence += min(20.0, rsi_strength * 0.8)

    if sweep:
        confidence += 10
    if trap:
        confidence += 8
    if strong in ("BULL_STRONG", "BEAR_STRONG"):
        confidence += 7

    # volatility event: warn + reduce certainty because market is wild
    event_note = ""
    if event:
        event_note = "⚠️ VOL SPIKE (possible news / banks / liquidity)"
        confidence -= 8

    if news_flag:
        event_note = "⚠️ NEWS RISK WINDOW"
        confidence -= 10

    confidence = max(10.0, min(95.0, confidence))
    confidence = round(confidence, 1)

    # ===== message =====
    # keep it clean + reasons
    reasons_txt = " | ".join(reasons[:3])  # no spam
    msg = (
        f"📊 GOLD SCALP ({INTERVAL})\n\n"
        f"💰 Price: {round(price, 2)}\n"
        f"📈 SMA20: {round(last_sma20, 2)}\n"
        f"📉 SMA50: {round(last_sma50, 2)}\n"
        f"📊 RSI: {round(last_rsi, 2)}\n"
        f"{'📏 ATR: ' + str(round(last_atr, 2)) if last_atr else ''}\n\n"
        f"🚦 Signal: {signal}\n"
        f"🔥 Confidence: {confidence}%\n"
        f"🧠 Reason: {reasons_txt}\n"
        f"{event_note}"
    ).strip()

    return signal, msg, confidence

# =========================
# MAIN LOOP
# =========================
def main():
    global last_candle_ts, last_signal_sent, last_alert_ts

    while True:
        try:
            df = fetch_gold()
            if df.empty:
                time.sleep(SLEEP_SECONDS)
                continue

            current_ts = str(df.index[-1])

            # Only react on a NEW candle timestamp
            if current_ts != last_candle_ts:
                result = make_signal(df)

                if result:
                    signal, msg, conf = result

                    # Anti-repeat: don’t spam same signal again
                    if signal != last_signal_sent:
                        send_telegram(msg)
                        last_signal_sent = signal
                        last_alert_ts = current_ts

                last_candle_ts = current_ts

            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            print("Error:", e)
            time.sleep(120)

if __name__ == "__main__":
    main()