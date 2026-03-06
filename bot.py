import os
import time
import requests
import pandas as pd
import yfinance as yf

# =========================
# إعدادات البوت
# =========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "PUT_YOUR_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "PUT_YOUR_CHAT_ID_HERE")

SYMBOL = "XAUUSD=X"
INTERVAL = "15m"
PERIOD = "5d"

CHECK_EVERY_SECONDS = 300   # كل 5 دقائق
RSI_PERIOD = 14
EMA_FAST = 9
EMA_SLOW = 21

last_sent_candle = None


# =========================
# إرسال رسالة تلغرام
# =========================
def send_telegram(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text
    }
    try:
        r = requests.post(url, data=payload, timeout=20)
        print("Telegram status:", r.status_code, r.text)
    except Exception as e:
        print("Telegram send error:", e)


# =========================
# جلب البيانات
# =========================
def get_data():
    for attempt in range(4):
        try:
            df = yf.download(
                tickers=SYMBOL,
                interval=INTERVAL,
                period=PERIOD,
                progress=False,
                auto_adjust=False,
                threads=False
            )

            if df is None or df.empty:
                print("No data received.")
                return None

            df = df.dropna().copy()

            # إذا كانت الأعمدة MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            needed = ["Open", "High", "Low", "Close"]
            for col in needed:
                if col not in df.columns:
                    print("Missing column:", col)
                    return None

            return df

        except Exception as e:
            print(f"Data fetch error (attempt {attempt+1}/4):", e)
            time.sleep(15 * (attempt + 1))

    return None


# =========================
# حساب RSI
# =========================
def compute_rsi(series: pd.Series, period: int = 14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# =========================
# تجهيز المؤشرات
# =========================
def prepare_indicators(df: pd.DataFrame):
    df = df.copy()

    df["EMA_FAST"] = df["Close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["EMA_SLOW"] = df["Close"].ewm(span=EMA_SLOW, adjust=False).mean()
    df["RSI"] = compute_rsi(df["Close"], RSI_PERIOD)

    return df


# =========================
# استخراج الإشارة
# =========================
def generate_signal(df: pd.DataFrame):
    global last_sent_candle

    if len(df) < 30:
        print("Not enough candles yet.")
        return None

    # آخر شمعة مغلقة
    last_closed = df.iloc[-2]
    prev_closed = df.iloc[-3]
    candle_time = str(df.index[-2])

    # لا نكرر نفس الشمعة
    if last_sent_candle == candle_time:
        print("Signal already sent for this candle:", candle_time)
        return None

    buy_condition = (
        last_closed["EMA_FAST"] > last_closed["EMA_SLOW"] and
        prev_closed["EMA_FAST"] <= prev_closed["EMA_SLOW"] and
        45 <= last_closed["RSI"] <= 70
    )

    sell_condition = (
        last_closed["EMA_FAST"] < last_closed["EMA_SLOW"] and
        prev_closed["EMA_FAST"] >= prev_closed["EMA_SLOW"] and
        30 <= last_closed["RSI"] <= 55
    )

    entry = float(last_closed["Close"])

    if buy_condition:
        sl = round(entry - 7.0, 2)
        tp = round(entry + 14.0, 2)
        last_sent_candle = candle_time
        return {
            "type": "BUY",
            "entry": round(entry, 2),
            "sl": sl,
            "tp": tp,
            "time": candle_time,
            "rsi": round(float(last_closed["RSI"]), 2)
        }

    if sell_condition:
        sl = round(entry + 7.0, 2)
        tp = round(entry - 14.0, 2)
        last_sent_candle = candle_time
        return {
            "type": "SELL",
            "entry": round(entry, 2),
            "sl": sl,
            "tp": tp,
            "time": candle_time,
            "rsi": round(float(last_closed["RSI"]), 2)
        }

    return None


# =========================
# تنسيق الرسالة
# =========================
def format_signal(signal: dict):
    return (
        f"XAUUSD {signal['type']} SIGNAL\n"
        f"Time: {signal['time']}\n"
        f"Entry: {signal['entry']}\n"
        f"SL: {signal['sl']}\n"
        f"TP: {signal['tp']}\n"
        f"RSI: {signal['rsi']}\n"
        f"TF: 15m"
    )


# =========================
# تشغيل رئيسي
# =========================
def main():
    send_telegram("Bot started successfully ✅")

    while True:
        try:
            print("Checking market data...")

            df = get_data()
            if df is not None and not df.empty:
                df = prepare_indicators(df)
                signal = generate_signal(df)

                if signal:
                    msg = format_signal(signal)
                    print("Sending signal:", msg)
                    send_telegram(msg)
                else:
                    print("No valid signal.")
            else:
                print("No data available.")

        except Exception as e:
            print("Main loop error:", e)

        time.sleep(CHECK_EVERY_SECONDS)


if __name__ == "__main__":
    main()