import os
import time
import math
import requests
import yfinance as yf
import pandas as pd

# =========================
# CONFIG
# =========================
SYMBOL = "GC=F"
INTERVAL = "15m"
PERIOD = "5d"
SLEEP_SECONDS = 180

MAX_CONFIDENCE = 95.0
MIN_BARS = 120

# Alerts config
FAKEOUT_N = 20
ATR_LEN = 14
ATR_FAST = 5
ATR_SLOW = 20

LIQUIDITY_SPIKE_MULT = 2.5
VOL_SURGE_MULT = 1.8

# Telegram env vars
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

if not BOT_TOKEN or not CHAT_ID:
    raise RuntimeError("Missing env vars: BOT_TOKEN and/or CHAT_ID. Add them in Render Environment Variables.")

TG_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"


# =========================
# TELEGRAM
# =========================
def send_telegram(text: str) -> None:
    payload = {"chat_id": CHAT_ID, "text": text}
    try:
        requests.post(TG_URL, data=payload, timeout=15)
    except Exception as e:
        print("Telegram send error:", e)


# =========================
# INDICATORS
# =========================
def compute_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean()

    rs = avg_gain / avg_loss.replace(0, math.nan)
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(length).mean()


# =========================
# DATA FETCH
# =========================
def fetch_gold() -> pd.DataFrame:
    df = yf.download(
        SYMBOL,
        interval=INTERVAL,
        period=PERIOD,
        auto_adjust=False,
        progress=False
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # Fix MultiIndex columns if any
    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(0)

    needed = {"Open", "High", "Low", "Close"}
    if not needed.issubset(set(df.columns)):
        return pd.DataFrame()

    return df.dropna().copy()


# =========================
# ALERTS (impact detection)
# =========================
def detect_alerts(df: pd.DataFrame) -> list[str]:
    alerts: list[str] = []
    if len(df) < max(MIN_BARS, ATR_SLOW + ATR_LEN + FAKEOUT_N + 5):
        return alerts

    last = df.iloc[-1]

    # ATRs
    atr14 = compute_atr(df, ATR_LEN)
    atr_fast = compute_atr(df, ATR_FAST)
    atr_slow = compute_atr(df, ATR_SLOW)

    last_atr14 = float(atr14.iloc[-1]) if not pd.isna(atr14.iloc[-1]) else None
    last_atr_fast = float(atr_fast.iloc[-1]) if not pd.isna(atr_fast.iloc[-1]) else None
    last_atr_slow = float(atr_slow.iloc[-1]) if not pd.isna(atr_slow.iloc[-1]) else None

    # Candle range
    candle_range = float(last["High"] - last["Low"])

    # 1) Liquidity Spike
    if last_atr14 and candle_range > (LIQUIDITY_SPIKE_MULT * last_atr14):
        alerts.append(f"Liquidity Spike (range {candle_range:.2f} > {LIQUIDITY_SPIKE_MULT}×ATR)")

    # 2) Volatility Surge
    if last_atr_fast and last_atr_slow and last_atr_fast > (VOL_SURGE_MULT * last_atr_slow):
        alerts.append(f"Volatility Surge (ATR{ATR_FAST} > {VOL_SURGE_MULT}×ATR{ATR_SLOW})")

    # 3) Fakeout detection
    lookback_high = df["High"].iloc[-(FAKEOUT_N + 1):-1].max()
    lookback_low = df["Low"].iloc[-(FAKEOUT_N + 1):-1].min()

    last_close = float(last["Close"])
    last_high = float(last["High"])
    last_low = float(last["Low"])

    if last_high > float(lookback_high) and last_close < float(lookback_high):
        alerts.append(f"Fakeout UP (broke {lookback_high:.2f} then closed back below)")

    if last_low < float(lookback_low) and last_close > float(lookback_low):
        alerts.append(f"Fakeout DOWN (broke {lookback_low:.2f} then closed back above)")

    # 4) News-like whipsaw candle (long wick vs body)
    body = abs(float(last["Close"] - last["Open"]))
    upper_wick = float(last["High"] - max(last["Close"], last["Open"]))
    lower_wick = float(min(last["Close"], last["Open"]) - last["Low"])
    wick = upper_wick + lower_wick

    if body < 1e-9:
        body = 1e-9

    wick_ratio = wick / body

    if last_atr14 and wick_ratio >= 3.0 and candle_range > (1.2 * last_atr14):
        alerts.append("News-like Whipsaw (long wicks + big range)")

    return alerts


# =========================
# SIGNAL + CONFIDENCE (Option B)
# =========================
def make_signal(df: pd.DataFrame) -> tuple[str | None, str | None, bool]:
    """
    Returns: (message_to_send, signal_name, has_alerts)
    If message is None -> don't send anything
    """
    if df.empty or len(df) < MIN_BARS:
        return None, None, False

    close = df["Close"]
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    rsi = compute_rsi(close, 14)

    last_price = float(close.iloc[-1])

    if pd.isna(sma20.iloc[-1]) or pd.isna(sma50.iloc[-1]) or pd.isna(rsi.iloc[-1]):
        return None, None, False

    last_sma20 = float(sma20.iloc[-1])
    last_sma50 = float(sma50.iloc[-1])
    last_rsi = float(rsi.iloc[-1])

    # Decide signal (no WAIT)
    if last_sma20 > last_sma50 and last_rsi > 50:
        signal = "BUY"
    elif last_sma20 < last_sma50 and last_rsi < 50:
        signal = "SELL"
    else:
        return None, None, False

    # Alerts
    alerts = detect_alerts(df)
    has_alerts = len(alerts) > 0

    # Confidence (heuristic)
    trend_strength = abs(last_sma20 - last_sma50)
    rsi_strength = abs(last_rsi - 50)

    atr14 = compute_atr(df, ATR_LEN)
    last_atr14 = float(atr14.iloc[-1]) if not pd.isna(atr14.iloc[-1]) else None
    scale = last_atr14 if last_atr14 and last_atr14 > 0 else (last_price * 0.001)

    trend_score = (trend_strength / scale) * 20.0
    rsi_score = (rsi_strength / 50.0) * 30.0

    confidence = 50.0 + trend_score + rsi_score

    # Penalize a bit if alerts exist (still sending signal, but with caution)
    if has_alerts:
        confidence -= 10.0

    confidence = max(5.0, min(MAX_CONFIDENCE, round(confidence, 1)))

    # Build alert block (Option B)
    alert_block = ""
    if has_alerts:
        alert_lines = "\n".join([f"⚠️ {a}" for a in alerts])
        alert_block = (
            f"⚠️ ALERTS (High impact conditions)\n"
            f"{alert_lines}\n"
            f"⚠️ Caution: volatility / possible fakeout.\n\n"
        )

    msg = (
        f"{alert_block}"
        f"📊 GOLD SCALP (15m)\n\n"
        f"💰 Price: {last_price:.2f}\n"
        f"📈 SMA20: {last_sma20:.2f}\n"
        f"📉 SMA50: {last_sma50:.2f}\n"
        f"📊 RSI: {last_rsi:.2f}\n\n"
        f"🚦 Signal: {signal}\n"
        f"🔥 Confidence: {confidence}%\n"
    )

    return msg, signal, has_alerts


# =========================
# LOOP (anti-spam)
# =========================
def main():
    last_candle_ts = None
    last_signal_sent = None
    last_alert_state_sent = None  # True/False

    while True:
        try:
            df = fetch_gold()
            if df.empty:
                print("No data, retry...")
                time.sleep(SLEEP_SECONDS)
                continue

            current_ts = str(df.index[-1])

            msg, signal, has_alerts = make_signal(df)

            if msg is not None:
                # Send only when candle changes AND (signal changed OR alert state changed)
                if current_ts != last_candle_ts:
                    if (signal != last_signal_sent) or (has_alerts != last_alert_state_sent):
                        send_telegram(msg)
                        last_signal_sent = signal
                        last_alert_state_sent = has_alerts
                    last_candle_ts = current_ts
            else:
                # No signal -> just update candle timestamp when it changes
                if current_ts != last_candle_ts:
                    last_candle_ts = current_ts

            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            print("Error:", e)
            time.sleep(120)


if __name__ == "__main__":
    main()