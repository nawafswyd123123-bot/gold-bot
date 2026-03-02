import time
import random
import requests
import yfinance as yf

TOKEN = "PUT_YOUR_TOKEN"
CHAT_ID = "PUT_YOUR_CHAT_ID"

SYMBOL = "GC=F"
INTERVAL = "15m"
PERIOD = "5d"

CHECK_SEC = 15 * 60          # فحص كل 15 دقيقة
RATE_LIMIT_WAIT = 60 * 60    # إذا انحظر: انتظر 60 دقيقة
MAX_RETRIES = 2              # محاولات قليلة فقط (حتى ما يزيد الحظر)

last_signal = None


def send(msg: str):
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": msg}, timeout=20)
    except Exception as e:
        print("Telegram error:", e)


def fetch_df():
    for attempt in range(MAX_RETRIES):
        try:
            df = yf.download(
                SYMBOL,
                interval=INTERVAL,
                period=PERIOD,
                progress=False,
                threads=False
            )
            if df is not None and not df.empty:
                return df

            # إذا رجّع فاضي، نطر شوي
            time.sleep(5)

        except Exception as e:
            err = str(e).lower()
            print("Yahoo error:", e)

            # ✅ إذا Rate Limit: وقف ساعة كاملة
            if "rate limit" in err or "too many requests" in err:
                print("Rate limited. Sleeping 60 minutes...")
                send("⚠️ Yahoo Rate Limit — رح انطر 60 دقيقة وبكفّي لحالي.")
                time.sleep(RATE_LIMIT_WAIT + random.randint(0, 120))
                return None

            # غير هيك: نطر شوي ونحاول مرة ثانية
            time.sleep(10)

    return None


def get_signal():
    df = fetch_df()
    if df is None:
        return None

    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    ema50 = df["EMA50"].iloc[-1].item()
    ema200 = df["EMA200"].iloc[-1].item()
    rsi = df["RSI"].iloc[-1].item()

    if ema50 > ema200 and rsi < 30:
        return "BUY"
    if ema50 < ema200 and rsi > 70:
        return "SELL"
    return None


send("✅ Gold Bot Started (M15) — Stable + Anti RateLimit")

while True:
    try:
        sig = get_signal()

        if sig and sig != last_signal:
            send(f"🔥 GOLD M15 SIGNAL: {sig}\nSymbol: {SYMBOL}")
            last_signal = sig

        # ✅ مهم: نضيف ثواني عشوائية بسيطة حتى ما يصير نمط ثابت
        time.sleep(CHECK_SEC + random.randint(5, 25))

    except Exception as e:
        print("Loop error:", e)
        time.sleep(60)