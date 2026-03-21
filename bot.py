import time
import math
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime

# =========================
# TELEGRAM SETTINGS
# =========================
TELEGRAM_BOT_TOKEN = "PUT_YOUR_BOT_TOKEN_HERE"
TELEGRAM_CHAT_ID = "PUT_YOUR_CHAT_ID_HERE"

# =========================
# MARKET SETTINGS
# =========================
SYMBOL = "GC=F"           # Gold Futures on Yahoo
ENTRY_INTERVAL = "15m"
HTF_INTERVAL = "60m"

ENTRY_PERIOD = "7d"
HTF_PERIOD = "30d"

CHECK_EVERY_SECONDS = 60

# =========================
# STRATEGY SETTINGS
# =========================
EMA_FAST = 20
EMA_SLOW = 50
RSI_PERIOD = 14
ATR_PERIOD = 14
VOL_MA_PERIOD = 20

MIN_ATR = 0.80            # ignore weak candles
VOLUME_MULTIPLIER = 1.10  # current volume must be > avg volume * this
RR_TP1 = 1.2
RR_TP2 = 2.0

last_sent_candle_time = None


# =========================
# TELEGRAM
# =========================
def send_telegram_message(text: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, data=payload, timeout=20)
        response.raise_for_status()
        print("Telegram message sent.")
    except Exception as e:
        print("Telegram error:", e)


# =========================
# INDICATORS
# =========================
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return atr


# =========================
# DATA LOADER
# =========================
def download_data(symbol: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(
        tickers=symbol,
        interval=interval,
        period=period,
        auto_adjust=False,
        progress=False
    )

    if df is None or df.empty:
        raise ValueError(f"No data for {symbol} {interval}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    needed = ["Open", "High", "Low", "Close", "Volume"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df = df.dropna().copy()
    return df


def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["EMA20"] = df["Close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=EMA_SLOW, adjust=False).mean()
    df["RSI"] = calculate_rsi(df["Close"], RSI_PERIOD)
    df["ATR"] = calculate_atr(df, ATR_PERIOD)
    df["VOL_MA"] = df["Volume"].rolling(VOL_MA_PERIOD).mean()

    return df.dropna().copy()


# =========================
# HELPERS
# =========================
def get_entry_data() -> pd.DataFrame:
    df = download_data(SYMBOL, ENTRY_INTERVAL, ENTRY_PERIOD)
    return prepare_indicators(df)


def get_htf_data() -> pd.DataFrame:
    df = download_data(SYMBOL, HTF_INTERVAL, HTF_PERIOD)
    return prepare_indicators(df)


def is_new_closed_candle(candle_time, last_time) -> bool:
    if last_time is None:
        return True
    return str(candle_time) != str(last_time)


def format_signal(signal: dict) -> str:
    return (
        f"🔥 *GOLD SIGNAL ({ENTRY_INTERVAL})*\n"
        f"*Type:* {signal['type']}\n"
        f"*Price:* {signal['price']:.2f}\n"
        f"*EMA20:* {signal['ema20']:.2f}\n"
        f"*EMA50:* {signal['ema50']:.2f}\n"
        f"*RSI:* {signal['rsi']:.2f}\n"
        f"*ATR:* {signal['atr']:.2f}\n"
        f"*Volume Ratio:* {signal['volume_ratio']:.2f}x\n"
        f"*1H Trend:* {signal['htf_trend']}\n"
        f"*SL:* {signal['sl']:.2f}\n"
        f"*TP1:* {signal['tp1']:.2f}\n"
        f"*TP2:* {signal['tp2']:.2f}\n"
        f"*Reason:* {signal['reason']}\n"
        f"*Time:* {signal['time']}\n"
        f"*Source:* {SYMBOL}"
    )


# =========================
# STRATEGY
# =========================
def detect_signal(entry_df: pd.DataFrame, htf_df: pd.DataFrame):
    if len(entry_df) < 60 or len(htf_df) < 60:
        return None

    # Last fully closed candle on 15m
    c1 = entry_df.iloc[-1]
    c2 = entry_df.iloc[-2]
    c3 = entry_df.iloc[-3]
    c4 = entry_df.iloc[-4]
    c5 = entry_df.iloc[-5]

    # Higher timeframe last closed candle
    h1 = htf_df.iloc[-1]

    close_1 = float(c1["Close"])
    open_1 = float(c1["Open"])
    high_1 = float(c1["High"])
    low_1 = float(c1["Low"])

    close_2 = float(c2["Close"])
    high_2 = float(c2["High"])
    low_2 = float(c2["Low"])

    ema20 = float(c1["EMA20"])
    ema50 = float(c1["EMA50"])
    rsi = float(c1["RSI"])
    atr = float(c1["ATR"])

    vol_now = float(c1["Volume"])
    vol_avg = float(c1["VOL_MA"]) if not math.isnan(float(c1["VOL_MA"])) else 0.0
    volume_ratio = vol_now / vol_avg if vol_avg > 0 else 0.0

    htf_ema20 = float(h1["EMA20"])
    htf_ema50 = float(h1["EMA50"])
    htf_close = float(h1["Close"])

    htf_up = htf_ema20 > htf_ema50 and htf_close > htf_ema20
    htf_down = htf_ema20 < htf_ema50 and htf_close < htf_ema20

    recent_high = max(float(c2["High"]), float(c3["High"]), float(c4["High"]), float(c5["High"]))
    recent_low = min(float(c2["Low"]), float(c3["Low"]), float(c4["Low"]), float(c5["Low"]))

    bullish_body = close_1 > open_1
    bearish_body = close_1 < open_1

    strong_volume = volume_ratio >= VOLUME_MULTIPLIER
    strong_atr = atr >= MIN_ATR

    trend_up = ema20 > ema50 and close_1 > ema20
    trend_down = ema20 < ema50 and close_1 < ema20

    # BUY:
    # 1) 15m trend up
    # 2) 1H trend up
    # 3) previous candle made fake breakdown below recent low or dipped under EMA20
    # 4) current candle closes back strong above EMA20 and previous high
    buy_condition = (
        trend_up
        and htf_up
        and strong_volume
        and strong_atr
        and (
            low_2 < recent_low
            or low_2 < float(c2["EMA20"])
        )
        and close_1 > ema20
        and close_1 > high_2
        and bullish_body
        and rsi > 52
    )

    # SELL:
    # 1) 15m trend down
    # 2) 1H trend down
    # 3) previous candle made fake breakout above recent high or spiked above EMA20
    # 4) current candle closes back strong below EMA20 and previous low
    sell_condition = (
        trend_down
        and htf_down
        and strong_volume
        and strong_atr
        and (
            high_2 > recent_high
            or high_2 > float(c2["EMA20"])
        )
        and close_1 < ema20
        and close_1 < low_2
        and bearish_body
        and rsi < 48
    )

    candle_time = c1.name

    if buy_condition:
        sl = min(low_1, low_2) - (atr * 0.20)
        risk = close_1 - sl
        if risk <= 0:
            return None

        tp1 = close_1 + (risk * RR_TP1)
        tp2 = close_1 + (risk * RR_TP2)

        return {
            "type": "BUY",
            "price": close_1,
            "ema20": ema20,
            "ema50": ema50,
            "rsi": rsi,
            "atr": atr,
            "volume_ratio": volume_ratio,
            "htf_trend": "UP",
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "reason": "15m uptrend + 1H uptrend + fake breakdown recovery + strong volume",
            "time": str(candle_time)
        }

    if sell_condition:
        sl = max(high_1, high_2) + (atr * 0.20)
        risk = sl - close_1
        if risk <= 0:
            return None

        tp1 = close_1 - (risk * RR_TP1)
        tp2 = close_1 - (risk * RR_TP2)

        return {
            "type": "SELL",
            "price": close_1,
            "ema20": ema20,
            "ema50": ema50,
            "rsi": rsi,
            "atr": atr,
            "volume_ratio": volume_ratio,
            "htf_trend": "DOWN",
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "reason": "15m downtrend + 1H downtrend + fake breakout failure + strong volume",
            "time": str(candle_time)
        }

    return None


# =========================
# MAIN LOOP
# =========================
def main():
    global last_sent_candle_time

    print("✅ Gold signal bot is running.")
    print(f"Entry TF: {ENTRY_INTERVAL}")
    print(f"HTF Filter: {HTF_INTERVAL}")
    print("Filters: EMA20/50 + RSI + ATR + Volume + 1H trend + Fake Breakout")

    send_telegram_message(
        "✅ Gold signal bot is running.\n"
        f"Entry TF: {ENTRY_INTERVAL}\n"
        f"HTF Filter: {HTF_INTERVAL}\n"
        "Filters: EMA20/50 + RSI + ATR + Volume + 1H trend + Fake Breakout"
    )

    while True:
        try:
            entry_df = get_entry_data()
            htf_df = get_htf_data()

            last_candle_time = entry_df.iloc[-1].name

            signal = detect_signal(entry_df, htf_df)

            if signal and is_new_closed_candle(last_candle_time, last_sent_candle_time):
                msg = format_signal(signal)
                print(msg)
                send_telegram_message(msg)
                last_sent_candle_time = last_candle_time
            else:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No new signal.")

        except Exception as e:
            print("Bot error:", e)

        time.sleep(CHECK_EVERY_SECONDS)


if __name__ == "__main__":
    main()