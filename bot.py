import time
import math
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime

# =========================
# TELEGRAM SETTINGS
# =========================
TELEGRAM_BOT_TOKEN = "8772073953:AAEJyq-Tx09me6fdjdiU59r79CIVxKY4WgM"
TELEGRAM_CHAT_ID = "6150648369"

# =========================
# MARKET SETTINGS
# =========================
SYMBOL = "GC=F"
ENTRY_INTERVAL = "15m"
HTF_INTERVAL = "60m"

ENTRY_PERIOD = "10d"
HTF_PERIOD = "45d"

# مهم: خفف عدد الطلبات حتى ما يعمل Rate Limit
CHECK_EVERY_SECONDS = 300   # كل 5 دقائق

# =========================
# STRATEGY SETTINGS
# =========================
EMA_FAST = 20
EMA_SLOW = 50
RSI_PERIOD = 14
ATR_PERIOD = 14
VOL_MA_PERIOD = 20
ADX_PERIOD = 14

MIN_ATR = 6.0
MIN_BODY_TO_ATR = 0.45
VOLUME_MULTIPLIER = 1.20
MIN_ADX = 18

RR_TP1 = 1.2
RR_TP2 = 2.0

COOLDOWN_CANDLES = 4

last_sent_candle_time = None
last_signal_type = None
cooldown_count = 0

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

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    plus_di = 100 * (
        plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        / atr.replace(0, 1e-10)
    )
    minus_di = 100 * (
        minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        / atr.replace(0, 1e-10)
    )

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)) * 100
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return adx

# =========================
# DATA LOADER
# =========================
def download_data(symbol: str, interval: str, period: str) -> pd.DataFrame | None:
    try:
        df = yf.download(
            tickers=symbol,
            interval=interval,
            period=period,
            auto_adjust=False,
            progress=False,
            threads=False
        )
    except Exception as e:
        print("Download error:", e)
        return None

    if df is None or df.empty:
        print(f"No data for {symbol} {interval}")
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    needed = ["Open", "High", "Low", "Close", "Volume"]
    for col in needed:
        if col not in df.columns:
            print(f"Missing column: {col}")
            return None

    df = df.dropna().copy()
    if df.empty:
        return None

    return df


def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None

    df = df.copy()

    df["EMA20"] = df["Close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=EMA_SLOW, adjust=False).mean()
    df["RSI"] = calculate_rsi(df["Close"], RSI_PERIOD)
    df["ATR"] = calculate_atr(df, ATR_PERIOD)
    df["VOL_MA"] = df["Volume"].rolling(VOL_MA_PERIOD).mean()
    df["ADX"] = calculate_adx(df, ADX_PERIOD)
    df["BODY"] = (df["Close"] - df["Open"]).abs()
    df["RANGE"] = df["High"] - df["Low"]

    df = df.dropna().copy()
    if df.empty:
        return None

    return df


# =========================
# HELPERS
# =========================
def get_entry_data() -> pd.DataFrame | None:
    df = download_data(SYMBOL, ENTRY_INTERVAL, ENTRY_PERIOD)
    return prepare_indicators(df)


def get_htf_data() -> pd.DataFrame | None:
    df = download_data(SYMBOL, HTF_INTERVAL, HTF_PERIOD)
    return prepare_indicators(df)


def is_new_closed_candle(candle_time, last_time) -> bool:
    if last_time is None:
        return True
    return str(candle_time) != str(last_time)


def candle_strength(candle: pd.Series) -> float:
    atr = float(candle["ATR"])
    body = float(candle["BODY"])
    if atr <= 0:
        return 0.0
    return body / atr


def format_signal(signal: dict) -> str:
    return (
        f"🔥 *GOLD SIGNAL ({ENTRY_INTERVAL})*\n"
        f"*Type:* {signal['type']}\n"
        f"*Price:* {signal['price']:.2f}\n"
        f"*EMA20:* {signal['ema20']:.2f}\n"
        f"*EMA50:* {signal['ema50']:.2f}\n"
        f"*RSI:* {signal['rsi']:.2f}\n"
        f"*ATR:* {signal['atr']:.2f}\n"
        f"*ADX:* {signal['adx']:.2f}\n"
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
    if entry_df is None or htf_df is None:
        return None

    if len(entry_df) < 80 or len(htf_df) < 80:
        return None

    # latest closed candles only
    c1 = entry_df.iloc[-2]
    c2 = entry_df.iloc[-3]
    c3 = entry_df.iloc[-4]
    c4 = entry_df.iloc[-5]
    c5 = entry_df.iloc[-6]
    c6 = entry_df.iloc[-7]

    h1 = htf_df.iloc[-2]
    h2 = htf_df.iloc[-3]

    close_1 = float(c1["Close"])
    open_1 = float(c1["Open"])
    high_1 = float(c1["High"])
    low_1 = float(c1["Low"])

    high_2 = float(c2["High"])
    low_2 = float(c2["Low"])

    ema20 = float(c1["EMA20"])
    ema50 = float(c1["EMA50"])
    rsi = float(c1["RSI"])
    atr = float(c1["ATR"])
    adx = float(c1["ADX"])

    vol_now = float(c1["Volume"])
    vol_avg = float(c1["VOL_MA"]) if not math.isnan(float(c1["VOL_MA"])) else 0.0
    volume_ratio = vol_now / vol_avg if vol_avg > 0 else 0.0

    htf_ema20 = float(h1["EMA20"])
    htf_ema50 = float(h1["EMA50"])
    htf_close = float(h1["Close"])
    htf_prev_close = float(h2["Close"])

    htf_up = htf_ema20 > htf_ema50 and htf_close > htf_ema20 and htf_close >= htf_prev_close
    htf_down = htf_ema20 < htf_ema50 and htf_close < htf_ema20 and htf_close <= htf_prev_close

    recent_high = max(float(c2["High"]), float(c3["High"]), float(c4["High"]), float(c5["High"]), float(c6["High"]))
    recent_low = min(float(c2["Low"]), float(c3["Low"]), float(c4["Low"]), float(c5["Low"]), float(c6["Low"]))

    bullish_body = close_1 > open_1
    bearish_body = close_1 < open_1

    body_strength = candle_strength(c1)

    strong_volume = volume_ratio >= VOLUME_MULTIPLIER
    strong_atr = atr >= MIN_ATR
    strong_trend = adx >= MIN_ADX
    strong_body = body_strength >= MIN_BODY_TO_ATR

    trend_up = ema20 > ema50 and close_1 > ema20
    trend_down = ema20 < ema50 and close_1 < ema20

    fake_breakdown = low_2 < recent_low and float(c2["Close"]) > recent_low
    fake_breakout = high_2 > recent_high and float(c2["Close"]) < recent_high

    buy_condition = (
        trend_up
        and htf_up
        and strong_volume
        and strong_atr
        and strong_trend
        and strong_body
        and fake_breakdown
        and close_1 > high_2
        and bullish_body
        and rsi >= 55
    )

    sell_condition = (
        trend_down
        and htf_down
        and strong_volume
        and strong_atr
        and strong_trend
        and strong_body
        and fake_breakout
        and close_1 < low_2
        and bearish_body
        and rsi <= 45
    )

    candle_time = c1.name

    if buy_condition:
        sl = min(low_1, low_2) - (atr * 0.25)
        risk = close_1 - sl
        if risk <= 0:
            return None

        tp1 = close_1 + (risk * RR_TP1)
        tp2 = close_1 + (risk * RR_TP2)

        if (tp1 - close_1) < (atr * 0.8):
            return None

        return {
            "type": "BUY",
            "price": close_1,
            "ema20": ema20,
            "ema50": ema50,
            "rsi": rsi,
            "atr": atr,
            "adx": adx,
            "volume_ratio": volume_ratio,
            "htf_trend": "UP",
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "reason": "1H uptrend + fake breakdown reclaim + strong close + volume + ADX",
            "time": str(candle_time)
        }

    if sell_condition:
        sl = max(high_1, high_2) + (atr * 0.25)
        risk = sl - close_1
        if risk <= 0:
            return None

        tp1 = close_1 - (risk * RR_TP1)
        tp2 = close_1 - (risk * RR_TP2)

        if (close_1 - tp1) < (atr * 0.8):
            return None

        return {
            "type": "SELL",
            "price": close_1,
            "ema20": ema20,
            "ema50": ema50,
            "rsi": rsi,
            "atr": atr,
            "adx": adx,
            "volume_ratio": volume_ratio,
            "htf_trend": "DOWN",
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "reason": "1H downtrend + fake breakout failure + strong close + volume + ADX",
            "time": str(candle_time)
        }

    return None

# =========================
# MAIN LOOP
# =========================
def main():
    global last_sent_candle_time, last_signal_type, cooldown_count

    print("✅ Gold signal bot is running.")
    print(f"Entry TF: {ENTRY_INTERVAL}")
    print(f"HTF Filter: {HTF_INTERVAL}")
    print("Filters: EMA20/50 + RSI + ATR + ADX + Volume + 1H trend + Fake Breakout")

    send_telegram_message(
        "✅ Gold signal bot is running.\n"
        f"Entry TF: {ENTRY_INTERVAL}\n"
        f"HTF Filter: {HTF_INTERVAL}\n"
        "Filters: EMA20/50 + RSI + ATR + ADX + Volume + 1H trend + Fake Breakout"
    )

    while True:
        try:
            entry_df = get_entry_data()
            htf_df = get_htf_data()

            if entry_df is None or htf_df is None:
                print("Data fetch failed, retrying later...")
                time.sleep(120)
                continue

            last_closed_candle_time = entry_df.iloc[-2].name
            signal = detect_signal(entry_df, htf_df)

            if is_new_closed_candle(last_closed_candle_time, last_sent_candle_time):
                if cooldown_count > 0:
                    cooldown_count -= 1

                if signal:
                    if last_signal_type == signal["type"] and cooldown_count > 0:
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Same direction blocked by cooldown.")
                    else:
                        msg = format_signal(signal)
                        print(msg)
                        send_telegram_message(msg)
                        last_sent_candle_time = last_closed_candle_time
                        last_signal_type = signal["type"]
                        cooldown_count = COOLDOWN_CANDLES
                else:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No signal on closed candle.")
                    last_sent_candle_time = last_closed_candle_time
            else:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Waiting for candle close...")

        except Exception as e:
            print("Bot error:", e)
            time.sleep(120)

        time.sleep(CHECK_EVERY_SECONDS)


if __name__ == "__main__":
    main()