import os
import time
import traceback
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf
import requests


# =========================
# الإعدادات
# =========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

SYMBOL = os.getenv("SYMBOL", "GC=F").strip()          # Gold Futures
INTERVAL = os.getenv("INTERVAL", "15m").strip()       # 15m
PERIOD = os.getenv("PERIOD", "5d").strip()            # enough candles for indicators
CHECK_SECONDS = int(os.getenv("CHECK_SECONDS", "60")) # check every 60 sec

EMA_FAST = 20
EMA_SLOW = 50
RSI_LEN = 14
ATR_LEN = 14
VOL_MA_LEN = 20

MIN_RSI_BUY = 55
MAX_RSI_SELL = 45

# لمنع تكرار نفس الإشارة
LAST_SIGNAL_FILE = "last_signal.txt"


# =========================
# أدوات مساعدة
# =========================
def log(message: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def send_telegram(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log("Telegram variables missing.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML"
    }

    response = requests.post(url, data=payload, timeout=20)
    if response.status_code != 200:
        raise Exception(f"Telegram error {response.status_code}: {response.text}")


def safe_numeric_series(df: pd.DataFrame, col_name: str) -> pd.Series:
    if col_name not in df.columns:
        raise ValueError(f"Column {col_name} not found. Available columns: {list(df.columns)}")

    series = df[col_name]

    # لو رجع عمود بشكل DataFrame أو MultiIndex نعالجه
    if isinstance(series, pd.DataFrame):
        if series.shape[1] == 0:
            raise ValueError(f"Column {col_name} is empty DataFrame.")
        series = series.iloc[:, 0]

    return pd.to_numeric(series, errors="coerce")


def calculate_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calculate_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = safe_numeric_series(df, "High")
    low = safe_numeric_series(df, "Low")
    close = safe_numeric_series(df, "Close")

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / length, adjust=False).mean()
    return atr


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    # إذا yfinance رجع MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for col in df.columns:
            if isinstance(col, tuple):
                new_cols.append(col[0])
            else:
                new_cols.append(col)
        df.columns = new_cols
    return df


def get_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        tickers=symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False
    )

    if df is None or df.empty:
        raise ValueError("No data returned from Yahoo Finance.")

    df = flatten_columns(df).copy()

    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

    # نحتفظ فقط بالأعمدة المطلوبة
    df = df[needed].copy()

    for col in needed:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)

    if len(df) < 60:
        raise ValueError(f"Not enough candles: {len(df)}")

    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = safe_numeric_series(df, "Close")
    volume = safe_numeric_series(df, "Volume")

    # هنا أصلحنا الخطأ: Close وليس [['Close']]
    df["EMA20"] = close.ewm(span=EMA_FAST, adjust=False).mean()
    df["EMA50"] = close.ewm(span=EMA_SLOW, adjust=False).mean()
    df["RSI"] = calculate_rsi(close, RSI_LEN)
    df["ATR"] = calculate_atr(df, ATR_LEN)
    df["VOL_MA"] = volume.rolling(VOL_MA_LEN).mean()

    return df


def get_higher_trend(symbol: str) -> str:
    df_1h = get_data(symbol, period="10d", interval="1h")
    df_1h = add_indicators(df_1h)

    last = df_1h.iloc[-1]
    if pd.isna(last["EMA20"]) or pd.isna(last["EMA50"]):
        return "NEUTRAL"

    if last["EMA20"] > last["EMA50"]:
        return "UP"
    if last["EMA20"] < last["EMA50"]:
        return "DOWN"
    return "NEUTRAL"


def signal_strength(row: pd.Series, trend_1h: str) -> int:
    score = 0

    if row["EMA20"] > row["EMA50"]:
        score += 25
    if row["Close"] > row["EMA20"]:
        score += 20
    if row["RSI"] >= MIN_RSI_BUY:
        score += 15
    if row["Volume"] > row["VOL_MA"]:
        score += 15
    if trend_1h == "UP":
        score += 25

    return min(score, 100)


def signal_strength_sell(row: pd.Series, trend_1h: str) -> int:
    score = 0

    if row["EMA20"] < row["EMA50"]:
        score += 25
    if row["Close"] < row["EMA20"]:
        score += 20
    if row["RSI"] <= MAX_RSI_SELL:
        score += 15
    if row["Volume"] > row["VOL_MA"]:
        score += 15
    if trend_1h == "DOWN":
        score += 25

    return min(score, 100)


def build_signal(df: pd.DataFrame, trend_1h: str):
    last = df.iloc[-1]
    prev = df.iloc[-2]

    # حماية من NaN
    required = ["Close", "EMA20", "EMA50", "RSI", "ATR", "Volume", "VOL_MA"]
    for col in required:
        if pd.isna(last[col]):
            return None

    buy_cross = prev["EMA20"] <= prev["EMA50"] and last["EMA20"] > last["EMA50"]
    sell_cross = prev["EMA20"] >= prev["EMA50"] and last["EMA20"] < last["EMA50"]

    buy_ok = (
        last["EMA20"] > last["EMA50"]
        and last["Close"] > last["EMA20"]
        and last["RSI"] >= MIN_RSI_BUY
        and trend_1h == "UP"
        and last["Volume"] >= last["VOL_MA"]
    )

    sell_ok = (
        last["EMA20"] < last["EMA50"]
        and last["Close"] < last["EMA20"]
        and last["RSI"] <= MAX_RSI_SELL
        and trend_1h == "DOWN"
        and last["Volume"] >= last["VOL_MA"]
    )

    price = float(last["Close"])
    atr = float(last["ATR"])

    if buy_ok:
        strength = signal_strength(last, trend_1h)
        sl = price - (1.2 * atr)
        tp1 = price + (1.8 * atr)
        tp2 = price + (3.0 * atr)

        return {
            "type": "BUY",
            "price": price,
            "ema20": float(last["EMA20"]),
            "ema50": float(last["EMA50"]),
            "rsi": float(last["RSI"]),
            "atr": atr,
            "strength": strength,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "reason": "EMA20 > EMA50, price above EMA20, RSI supportive, volume confirmed, 1H trend UP",
            "cross": buy_cross,
        }

    if sell_ok:
        strength = signal_strength_sell(last, trend_1h)
        sl = price + (1.2 * atr)
        tp1 = price - (1.8 * atr)
        tp2 = price - (3.0 * atr)

        return {
            "type": "SELL",
            "price": price,
            "ema20": float(last["EMA20"]),
            "ema50": float(last["EMA50"]),
            "rsi": float(last["RSI"]),
            "atr": atr,
            "strength": strength,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "reason": "EMA20 < EMA50, price below EMA20, RSI supportive, volume confirmed, 1H trend DOWN",
            "cross": sell_cross,
        }

    return None


def format_signal_message(signal: dict) -> str:
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    cross_text = "Yes" if signal.get("cross") else "No"

    return (
        f"🔥 <b>GOLD SIGNAL ({INTERVAL})</b>\n"
        f"Type: <b>{signal['type']}</b>\n"
        f"Price: {signal['price']:.2f}\n"
        f"EMA20: {signal['ema20']:.2f}\n"
        f"EMA50: {signal['ema50']:.2f}\n"
        f"RSI: {signal['rsi']:.2f}\n"
        f"ATR: {signal['atr']:.2f}\n"
        f"Strength: {signal['strength']}%\n"
        f"EMA Cross Now: {cross_text}\n"
        f"SL: {signal['sl']:.2f}\n"
        f"TP1: {signal['tp1']:.2f}\n"
        f"TP2: {signal['tp2']:.2f}\n"
        f"Reason: {signal['reason']}\n"
        f"Time: {now_utc}\n"
        f"Source: {SYMBOL}"
    )


def load_last_signal_key() -> str:
    if not os.path.exists(LAST_SIGNAL_FILE):
        return ""
    try:
        with open(LAST_SIGNAL_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def save_last_signal_key(key: str) -> None:
    with open(LAST_SIGNAL_FILE, "w", encoding="utf-8") as f:
        f.write(key)


def make_signal_key(signal: dict) -> str:
    # مفتاح بسيط لمنع تكرار نفس الإشارة
    return f"{signal['type']}_{round(signal['price'], 2)}_{round(signal['ema20'], 2)}_{round(signal['ema50'], 2)}"


def run_bot() -> None:
    log("Bot started successfully.")

    # رسالة تشغيل أول مرة
    try:
        send_telegram(
            f"✅ Gold signal bot is running.\n"
            f"Timeframe: {INTERVAL}\n"
            f"Filters: EMA20/50 + RSI + ATR + Volume + 1H trend"
        )
    except Exception as e:
        log(f"Startup telegram error: {e}")

    while True:
        try:
            df = get_data(SYMBOL, PERIOD, INTERVAL)
            df = add_indicators(df)

            trend_1h = get_higher_trend(SYMBOL)
            signal = build_signal(df, trend_1h)

            if signal:
                signal_key = make_signal_key(signal)
                last_key = load_last_signal_key()

                if signal_key != last_key:
                    message = format_signal_message(signal)
                    send_telegram(message)
                    save_last_signal_key(signal_key)
                    log(f"Signal sent: {signal['type']} at {signal['price']:.2f}")
                else:
                    log("Same signal already sent before. Skipping.")
            else:
                log("No valid signal right now.")

        except Exception as e:
            log(f"Main loop error: {e}")
            log(traceback.format_exc())

        time.sleep(CHECK_SECONDS)


if __name__ == "__main__":
    run_bot()