def make_signal(df):
    close = df["Close"]

    # تأكد انه Series مش DataFrame
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    price = round(float(close.iloc[-1]), 2)
    last_sma20 = round(float(sma20.iloc[-1]), 2)
    last_sma50 = round(float(sma50.iloc[-1]), 2)
    last_rsi = round(float(rsi.iloc[-1]), 2)

    # ====== Confidence (نسبة تقريبية - مش ضمان) ======
    trend_strength = abs(last_sma20 - last_sma50)
    rsi_strength = abs(last_rsi - 50)

    confidence = 50 + (trend_strength * 2) + (rsi_strength * 0.8)
    if confidence > 95:
        confidence = 95
    confidence = round(confidence, 1)

    # ====== Signal ======
    if last_sma20 > last_sma50 and last_rsi > 50:
        signal = "BUY"
    elif last_sma20 < last_sma50 and last_rsi < 50:
        signal = "SELL"
    else:
        return None  # ما تبعت WAIT نهائياً

    message = f"""
📊 GOLD SCALP (15m)

💰 Price: {price}
📈 SMA20: {last_sma20}
📉 SMA50: {last_sma50}
📊 RSI: {last_rsi}

🚦 Signal: {signal}
🔥 Confidence: {confidence}%
"""
    return message


last_candle = None
last_signal = None

while True:
    try:
        df = fetch_gold()
        current_candle = str(df.index[-1])

        if current_candle != last_candle:
            msg = make_signal(df)

            if msg is not None:
                # استخرج الإشارة من النص لتجنب التكرار
                current_signal = "BUY" if "Signal: BUY" in msg else "SELL"

                if current_signal != last_signal:
                    send_telegram(msg)
                    last_signal = current_signal

            last_candle = current_candle

        time.sleep(180)

    except Exception as e:
        print("Error:", e)
        time.sleep(120)