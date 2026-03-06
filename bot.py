import os
import time
import math
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone

# =========================
# SETTINGS
# =========================
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("CHAT_ID", "").strip()

SYMBOL = "GC=F"          # Gold Futures
INTERVAL_MAIN = "15m"    # Main timeframe
INTERVAL_HTF = "60m"     # Higher timeframe
PERIOD = "10d"           # Enough candles for indicators
CHECK_EVERY_SECONDS = 60

# Signal rules
MIN_STRENGTH_TO_SEND = 80
EMA_GAP_THRESHOLD = 0.0008     # stronger separation
RSI_BUY_MIN = 55
RSI_BUY_MAX = 68
RSI_SELL_MIN = 32
RSI_SELL_MAX = 45
ATR_MIN_RATIO = 0.0012         # ATR / price minimum to avoid dead market
VOLUME_LOOKBACK = 20
RSI_LENGTH = 14
ATR_LENGTH = 14

# Risk settings
SL_ATR_MULTIPLIER = 1.2
TP1_RR = 1.0
TP2_RR = 1.7

# Anti-spam
last_signal_key = None
last_signal_time = None

# =========================
# HELPERS
# =========================
def send_telegram_message(text: str):
    if not BOT_TOKEN or not CHAT_ID:
        print("ERROR: BOT_TOKEN or CHAT_ID missing.")
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }

    try:
        r = requests.post(url, json=payload, timeout=20)
        print("Telegram response:", r.status_code, r.text)
    except Exception as e:
        print("Telegram send error:", e)


def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except:
        return default


def get_data(symbol: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(
        tickers=symbol,
        interval=interval,
        period=period,
        auto_adjust=False,
        progress=False,
        threads=False
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # Fix multi-index columns if returned
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Normalize column names
    rename_map = {}
    for col in df.columns:
        lc = str(col).lower()
        if "open" in lc:
            rename_map[col] = "Open"
        elif "high" in lc:
            rename_map[col] = "High"
        elif "low" in lc:
            rename_map[col] = "Low"
        elif "close" in lc:
            rename_map[col] = "Close"
        elif "volume" in lc:
            rename_map[col] = "Volume"

    df = df.rename(columns=rename_map)

    needed = ["Open", "High", "Low", "Close", "Volume"]
    for c in needed:
        if c not in df.columns:
            df[c] = 0.0

    df = df[needed].copy()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df


def ema(series: pd.Series, length: int):
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, length: int = 14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, min_periods=length, adjust=False).mean()


def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["EMA20"] = ema(df["Close"], 20)
    df["EMA50"] = ema(df["Close"], 50)
    df["RSI"] = rsi(df["Close"], RSI_LENGTH)
    df["ATR"] = atr(df, ATR_LENGTH)
    df["VOL_MA"] = df["Volume"].rolling(VOLUME_LOOKBACK).mean()

    df["BullishCandle"] = df["Close"] > df["Open"]
    df["BearishCandle"] = df["Close"] < df["Open"]
    return df


def get_higher_trend(df_1h: pd.DataFrame) -> str:
    if df_1h.empty or len(df_1h) < 60:
        return "NEUTRAL"

    last = df_1h.iloc[-1]
    close = safe_float(last["Close"])
    ema20 = safe_float(last["EMA20"])
    ema50 = safe_float(last["EMA50"])
    r = safe_float(last["RSI"])

    if close > ema20 > ema50 and r > 52:
        return "BULLISH"
    elif close < ema20 < ema50 and r < 48:
        return "BEARISH"
    return "NEUTRAL"


def compute_strength(signal_type: str, row, htf_trend: str) -> int:
    score = 0
    close = safe_float(row["Close"])
    ema20 = safe_float(row["EMA20"])
    ema50 = safe_float(row["EMA50"])
    r = safe_float(row["RSI"])
    atr_val = safe_float(row["ATR"])
    vol = safe_float(row["Volume"])
    vol_ma = safe_float(row["VOL_MA"], 1.0)

    ema_gap_ratio = abs(ema20 - ema50) / close if close else 0
    atr_ratio = atr_val / close if close else 0

    if signal_type == "BUY":
        if ema20 > ema50:
            score += 30
        if close > ema20 and close > ema50:
            score += 20
        if RSI_BUY_MIN <= r <= RSI_BUY_MAX:
            score += 15
        if vol > vol_ma:
            score += 15
        if htf_trend == "BULLISH":
            score += 20

        if ema_gap_ratio < EMA_GAP_THRESHOLD:
            score -= 20
        if atr_ratio < ATR_MIN_RATIO:
            score -= 15
        if not bool(row["BullishCandle"]):
            score -= 10

    elif signal_type == "SELL":
        if ema20 < ema50:
            score += 30
        if close < ema20 and close < ema50:
            score += 20
        if RSI_SELL_MIN <= r <= RSI_SELL_MAX:
            score += 15
        if vol > vol_ma:
            score += 15
        if htf_trend == "BEARISH":
            score += 20

        if ema_gap_ratio < EMA_GAP_THRESHOLD:
            score -= 20
        if atr_ratio < ATR_MIN_RATIO:
            score -= 15
        if not bool(row["BearishCandle"]):
            score -= 10

    score = max(0, min(100, score))
    return int(score)


def build_signal(df_15m: pd.DataFrame, htf_trend: str):
    if df_15m.empty or len(df_15m) < 60:
        return None

    row = df_15m.iloc[-1]

    close = safe_float(row["Close"])
    ema20 = safe_float(row["EMA20"])
    ema50 = safe_float(row["EMA50"])
    r = safe_float(row["RSI"])
    atr_val = safe_float(row["ATR"])
    vol = safe_float(row["Volume"])
    vol_ma = safe_float(row["VOL_MA"], 1.0)
    ema_gap_ratio = abs(ema20 - ema50) / close if close else 0
    atr_ratio = atr_val / close if close else 0

    # Avoid bad market state
    if close <= 0 or atr_val <= 0:
        return None

    # Avoid sideways / weak conditions
    if ema_gap_ratio < EMA_GAP_THRESHOLD:
        return None
    if atr_ratio < ATR_MIN_RATIO:
        return None
    if 47 <= r <= 53:
        return None

    # Avoid chasing huge candle
    candle_range = safe_float(row["High"]) - safe_float(row["Low"])
    if candle_range > atr_val * 1.8:
        return None

    signal_type = None
    reason_parts = []

    # BUY
    if (
        ema20 > ema50 and
        close > ema20 and
        close > ema50 and
        RSI_BUY_MIN <= r <= RSI_BUY_MAX and
        vol > vol_ma and
        htf_trend == "BULLISH" and
        bool(row["BullishCandle"])
    ):
        signal_type = "BUY"
        reason_parts = [
            "EMA20 > EMA50",
            "price above EMA20/EMA50",
            "RSI supportive",
            "volume above average",
            "1H trend bullish"
        ]

    # SELL
    elif (
        ema20 < ema50 and
        close < ema20 and
        close < ema50 and
        RSI_SELL_MIN <= r <= RSI_SELL_MAX and
        vol > vol_ma and
        htf_trend == "BEARISH" and
        bool(row["BearishCandle"])
    ):
        signal_type = "SELL"
        reason_parts = [
            "EMA20 < EMA50",
            "price below EMA20/EMA50",
            "RSI supportive",
            "volume above average",
            "1H trend bearish"
        ]

    if not signal_type:
        return None

    strength = compute_strength(signal_type, row, htf_trend)
    if strength < MIN_STRENGTH_TO_SEND:
        return None

    # SL / TP
    if signal_type == "BUY":
        sl = close - (atr_val * SL_ATR_MULTIPLIER)
        risk = close - sl
        tp1 = close + (risk * TP1_RR)
        tp2 = close + (risk * TP2_RR)
    else:
        sl = close + (atr_val * SL_ATR_MULTIPLIER)
        risk = sl - close
        tp1 = close - (risk * TP1_RR)
        tp2 = close - (risk * TP2_RR)

    signal_time = df_15m.index[-1]
    if hasattr(signal_time, "to_pydatetime"):
        signal_time = signal_time.to_pydatetime()

    return {
        "type": signal_type,
        "price": round(close, 2),
        "ema20": round(ema20, 2),
        "ema50": round(ema50, 2),
        "rsi": round(r, 2),
        "strength": strength,
        "sl": round(sl, 2),
        "tp1": round(tp1, 2),
        "tp2": round(tp2, 2),
        "reason": ", ".join(reason_parts),
        "time": signal_time,
        "source": SYMBOL
    }


def format_signal_message(signal: dict) -> str:
    icon = "🟢" if signal["type"] == "BUY" else "🔴"

    signal_time = signal["time"]
    if isinstance(signal_time, datetime):
        if signal_time.tzinfo is None:
            signal_time = signal_time.replace(tzinfo=timezone.utc)
        signal_time_str = signal_time.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    else:
        signal_time_str = str(signal_time)

    return (
        f"🔥 *GOLD SIGNAL (15m)*\n"
        f"{icon} *Type:* {signal['type']}\n"
        f"*Price:* {signal['price']}\n"
        f"*EMA20:* {signal['ema20']}\n"
        f"*EMA50:* {signal['ema50']}\n"
        f"*RSI:* {signal['rsi']}\n"
        f"*Strength:* {signal['strength']}%\n"
        f"*SL:* {signal['sl']}\n"
        f"*TP1:* {signal['tp1']}\n"
        f"*TP2:* {signal['tp2']}\n"
        f"*Reason:* {signal['reason']}\n"
        f"*Time:* {signal_time_str}\n"
        f"*Source:* {signal['source']}"
    )


def should_send_signal(signal: dict) -> bool:
    global last_signal_key, last_signal_time

    if not signal:
        return False

    signal_time = signal["time"]
    key = f"{signal['type']}_{signal_time}_{signal['price']}"

    if key == last_signal_key:
        return False

    # Prevent repeating same direction too fast on same candle structure
    if last_signal_time is not None and isinstance(signal_time, datetime):
        diff = abs((signal_time - last_signal_time).total_seconds())
        if diff < 10 * 60 and signal["type"] in str(last_signal_key):
            return False

    last_signal_key = key
    last_signal_time = signal_time
    return True


def run_bot():
    print("Bot started...")

    startup_msg = (
        "✅ Gold signal bot is running.\n"
        "Timeframe: 15m\n"
        "Filters: EMA20/50 + RSI + ATR + Volume + 1H trend"
    )
    send_telegram_message(startup_msg)

    while True:
        try:
            # 15m data
            df_15m = get_data(SYMBOL, INTERVAL_MAIN, PERIOD)
            df_15m = prepare_indicators(df_15m)

            # 1H data
            df_1h = get_data(SYMBOL, INTERVAL_HTF, "20d")
            df_1h = prepare_indicators(df_1h)
            htf_trend = get_higher_trend(df_1h)

            signal = build_signal(df_15m, htf_trend)

            if signal and should_send_signal(signal):
                msg = format_signal_message(signal)
                print("Sending signal:", msg)
                send_telegram_message(msg)
            else:
                print(f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC | No strong signal")

        except Exception as e:
            print("Main loop error:", e)

        time.sleep(CHECK_EVERY_SECONDS)


if __name__ == "__main__":
    run_bot()