import os
import time
import math
import requests
import pandas as pd
import yfinance as yf

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

SYMBOL = "GC=F"
INTERVAL = "15m"
PERIOD = "5d"          # كافي للحسابات
SLEEP_SECONDS = 180    # كل 3 دقايق (خفيف على yfinance)

# منع التكرار
last_candle_ts = None
last_signal_sent = None
last_alert_ts = None


def send_telegram(text: str):
    if not BOT_TOKEN or not CHAT_ID:
        print("Missing BOT_TOKEN or CHAT_ID in env vars")
        return

    if not text:
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text
    }
    try:
        r = requests.post(url, data=payload, timeout=20)
        if r.status_code != 200:
            print("Telegram error:", r.status_code, r.text[:200])
    except Exception as e:
        print("Telegram request failed:", e)


def fetch_gold():
    df = yf.download(
        SYMBOL,
        interval=INTERVAL,
        period=PERIOD,
        progress=False,
        auto_adjust=False,
        threads=False
    )

    if df is None or df.empty:
        return None

    # إذا columns MultiIndex
    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(0)

    # تأكد عندنا الأعمدة المطلوبة
    needed = {"Open", "High", "Low", "Close"}
    if not needed.issubset(set(df.columns)):
        return None

    df = df.dropna()
    return df


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
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


def compute_signal_and_confidence(df: pd.DataFrame):
    close = df["Close"]

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    rsi14 = rsi(close, 14)

    last_price = float(close.iloc[-1])
    last_sma20 = float(sma20.iloc[-1])
    last_sma50 = float(sma50.iloc[-1])
    last_rsi = float(rsi14.iloc[-1])

    # إشارة (بدون WAIT)
    if last_sma20 > last_sma50 and last_rsi > 50:
        signal = "BUY"
    elif last_sma20 < last_sma50 and last_rsi < 50:
        signal = "SELL"
    else:
        signal = None  # لا نرسل شيء

    # Confidence تقريبية (0-95)
    # قوة الترند = فرق SMA كنسبة من السعر
    trend_strength = abs(last_sma20 - last_sma50) / max(last_price, 1e-9)
    # قوة RSI = بعده عن 50
    rsi_strength = abs(last_rsi - 50) / 50.0

    conf = 50 + (trend_strength * 2500) + (rsi_strength * 35)
    conf = min(conf, 95)
    conf = round(conf, 1)

    return signal, conf, last_price, round(last_sma20, 2), round(last_sma50, 2), round(last_rsi, 2)


def detect_volatility_alert(df: pd.DataFrame):
    """
    تنبيه شمعة قوية/سيولة:
    - Range الشمعة الأخيرة > 1.8 * ATR
    - أو جسم الشمعة كبير
    """
    if len(df) < 60:
        return None

    a = atr(df, 14)
    last_atr = float(a.iloc[-1]) if not math.isnan(a.iloc[-1]) else None
    if not last_atr or last_atr <= 0:
        return None

    o = float(df["Open"].iloc[-1])
    h = float(df["High"].iloc[-1])
    l = float(df["Low"].iloc[-1])
    c = float(df["Close"].iloc[-1])

    candle_range = h - l
    body = abs(c - o)

    if candle_range > 1.8 * last_atr or body > 1.2 * last_atr:
        strength = round((candle_range / last_atr), 2)
        direction = "🟢 Bullish impulse" if c > o else "🔴 Bearish impulse"
        return f"⚡ Volatility Spike ({INTERVAL})\n{direction}\nRange/ATR: {strength}x"

    return None


def detect_false_breakout(df: pd.DataFrame):
    """
    كسر كاذب بسيط:
    - عمل High فوق أعلى 20 شمعة وبعدين أغلق تحتها (false break up)
    - أو عمل Low تحت أدنى 20 شمعة وبعدين أغلق فوقها (false break down)
    """
    if len(df) < 40:
        return None

    lookback = 20
    prev_high = float(df["High"].iloc[-(lookback+1):-1].max())
    prev_low = float(df["Low"].iloc[-(lookback+1):-1].min())

    last_high = float(df["High"].iloc[-1])
    last_low = float(df["Low"].iloc[-1])
    last_close = float(df["Close"].iloc[-1])

    if last_high > prev_high and last_close < prev_high:
        return f"🧨 False Break Up ({INTERVAL})\nBroke above {round(prev_high,2)} then closed back under."
    if last_low < prev_low and last_close > prev_low:
        return f"🧨 False Break Down ({INTERVAL})\nBroke below {round(prev_low,2)} then closed back above."
    return None


def format_signal_message(signal, conf, price, sma20, sma50, rsi14):
    return (
        f"📊 GOLD SCALP ({INTERVAL})\n\n"
        f"💰 Price: {price}\n"
        f"📈 SMA20: {sma20}\n"
        f"📉 SMA50: {sma50}\n"
        f"📊 RSI: {rsi14}\n\n"
        f"🚦 Signal: {signal}\n"
        f"🔥 Confidence: {conf}%"
    )


def main_loop():
    global last_candle_ts, last_signal_sent, last_alert_ts

    while True:
        try:
            df = fetch_gold()
            if df is None:
                print("No data")
                time.sleep(30)
                continue

            current_ts = str(df.index[-1])

            # حساب الإشارة
            signal, conf, price, sma20, sma50, rsi14 = compute_signal_and_confidence(df)

            # Alert events
            vol_alert = detect_volatility_alert(df)
            fb_alert = detect_false_breakout(df)

            # نرسل Alerts مرة واحدة لكل شمعة (حتى ما يكرر)
            # + نرسل signal فقط إذا تغيّر
            if current_ts != last_candle_ts:
                # شمعة جديدة
                if signal and signal != last_signal_sent:
                    send_telegram(format_signal_message(signal, conf, price, sma20, sma50, rsi14))
                    last_signal_sent = signal

                # Alerts (إذا موجودة)
                alert_texts = []
                if vol_alert:
                    alert_texts.append(vol_alert)
                if fb_alert:
                    alert_texts.append(fb_alert)

                if alert_texts:
                    send_telegram("\n\n".join(alert_texts))

                last_candle_ts = current_ts

            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            print("Error:", e)
            time.sleep(120)


if __name__ == "__main__":
    main_loop()