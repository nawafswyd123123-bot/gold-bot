import time
import requests
import yfinance as yf

# ====== SETTINGS ======
TOKEN = ""
CHAT_ID = "PUT_YOUR_CHAT_ID_HERE"

SYMBOL = "GC=F"        # Gold Futures on Yahoo (stable)
INTERVAL = "15m"       # 15 minutes timeframe
PERIOD = "5d"          # More history helps indicators
CHECK_EVERY_SEC = 15 * 60  # 15 minutes

last_signal = None
last_sent_at = 0
COOLDOWN_SEC = 15 * 60  # prevent repeats within same candle


# ====== TELEGRAM ======
def send_telegram(text: str) -> bool:
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        r = requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=20)
        return r.status_code == 200
    except Exception as e:
        print("Telegram error:", e)
        return False


# ====== INDICATORS ======
def ema(series, span: int):
    return series.ewm(span=span, adjust=False).mean()

def rsi(close, length: int = 14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ====== SIGNAL LOGIC (STRONG) ======
def get_signal():
    try:
        df = yf.download(SYMBOL, interval=INTERVAL, period=PERIOD, progress=False)

        if df is None or df.empty:
            print("No data received from Yahoo (market closed or Yahoo issue).")
            return None, None

        # Indicators
        df["EMA50"] = ema(df["Close"], 50)
        df["EMA200"] = ema(df["Close"], 200)
        df["RSI"] = rsi(df["Close"], 14)

        last = df.iloc[-1]
        price = float(last["Close"])
        ema50 = float(last["EMA50"])
        ema200 = float(last["EMA200"])
        r = float(last["RSI"])

        # Strong filter: Trend + RSI extreme
        if ema50 > ema200 and r <= 30:
            return "BUY", price
        if ema50 < ema200 and r >= 70:
            return "SELL", price

        return None, price

    except Exception as e:
        print("Data error:", e)
        return None, None


# ====== MAIN LOOP ======
def main():
    global last_signal, last_sent_at

    send_telegram("✅ Gold Signal Bot started (M15)")

    while True:
        signal, price = get_signal()

        now = int(time.time())

        # Send only when: signal exists AND changed AND cooldown passed
        if signal and signal != last_signal and (now - last_sent_at) >= COOLDOWN_SEC:
            msg = f"🔥 GOLD SIGNAL (M15)\nSignal: {signal}\nPrice: {price:.2f}\nSymbol: {SYMBOL}"
            ok = send_telegram(msg)
            if ok:
                last_signal = signal
                last_sent_at = now

        time.sleep(CHECK_EVERY_SEC)


if __name__ == "__main__":
    main()