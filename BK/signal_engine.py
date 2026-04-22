class SignalEngine:
    def __init__(self):
        # Layer 2: Weights
        self.weights = {
            'ema_cross': 0.20,
            'rsi': 0.15,
            'macd': 0.20,
            'bollinger': 0.10,
            'volume': 0.10,
            'trend_sma': 0.25  # T?ng = 1.0
        }

    def generate_signal(self, df):
        if df.empty or len(df) < 50:
            return {"action": "HOLD", "score": 0, "details": {}}

        # TR?CH XU?T 3 N?N G?N NH?T ?? CONFIRM T?N HI?U (Ch?ng nhi?u)
        last_3 = df.tail(3)
        if len(last_3) < 3:
            return {"action": "HOLD", "score": 0, "details": {}}

        current = last_3.iloc[2]
        prev_1 = last_3.iloc[1]
        prev_2 = last_3.iloc[0]

        signals = {}  # direction: 1 (BUY), -1 (SELL), 0 (HOLD). Value = (direction, confidence)

        # 1. EMA Crossover (9 & 21) - B? L?C 3 N?N
        # Mua m?nh: N?n hi?n t?i v? n?n tr??c n?m tr?n, n?n tr??c n?a n?m d??i (X?c nh?n cross th?t)
        if current['EMA_9'] > current['EMA_21'] and prev_1['EMA_9'] > prev_1['EMA_21'] and prev_2['EMA_9'] <= prev_2[
            'EMA_21']:
            signals['ema_cross'] = (1, 0.9)
            # B?n m?nh: T??ng t? nh?ng chi?u ng??c l?i
        elif current['EMA_9'] < current['EMA_21'] and prev_1['EMA_9'] < prev_1['EMA_21'] and prev_2['EMA_9'] >= prev_2[
            'EMA_21']:
            signals['ema_cross'] = (-1, 0.9)
        else:
            # ?ang duy tr? trend, kh?ng c? ?i?m c?t
            if current['EMA_9'] > current['EMA_21']:
                signals['ema_cross'] = (1, 0.4)
            else:
                signals['ema_cross'] = (-1, 0.4)

        # 2. RSI (14)
        if current['RSI_14'] < 30:
            signals['rsi'] = (1, (30 - current['RSI_14']) / 30)
        elif current['RSI_14'] > 70:
            signals['rsi'] = (-1, (current['RSI_14'] - 70) / 30)
        else:
            signals['rsi'] = (0, 0.0)

        # 3. MACD
        if current['MACD_hist'] > 0 and prev_1['MACD_hist'] <= 0:
            signals['macd'] = (1, 0.8)
        elif current['MACD_hist'] < 0 and prev_1['MACD_hist'] >= 0:
            signals['macd'] = (-1, 0.8)
        else:
            signals['macd'] = (1 if current['MACD_hist'] > 0 else -1, 0.3)

        # 4. Bollinger Bands
        if current['close'] < current['BB_lower']:
            signals['bollinger'] = (1, 0.8)
        elif current['close'] > current['BB_upper']:
            signals['bollinger'] = (-1, 0.8)
        else:
            signals['bollinger'] = (0, 0)

        # 5. Volume Confirmation
        vol_conf = 0.8 if current['volume'] > current.get('VOL_MA', 0) else 0.3
        if current['close'] > current['open']:
            signals['volume'] = (1, vol_conf)
        else:
            signals['volume'] = (-1, vol_conf)

        # 6. Trend SMA (50 & 200)
        trend_direction = 1 if current['SMA_50'] > current['SMA_200'] else -1
        signals['trend_sma'] = (trend_direction, 0.6)

        predicted_trend_short = "BULLISH" if current['EMA_9'] > current['EMA_21'] else "BEARISH"
        predicted_trend_mid = "BULLISH" if current['SMA_50'] > current['SMA_200'] else "BEARISH"

        # --- B? L?C ADX (S?C M?NH XU H??NG) ---
        adx_value = current.get('ADX_14', 25)  # N?u ko c? ADX, m?c ??nh l? 25 (Trend ?ang kho?)
        adx_multiplier = 1.0
        if adx_value < 20:
            adx_multiplier = 0.5  # Th? tr??ng ?i ngang, gi?m 50% ?? tin c?y c?a MA/Trend

        # --- Layer 3: Ensemble Decision ---
        final_score = 0.0
        for key, (direction, confidence) in signals.items():
            # ?p d?ng b? l?c ADX cho c?c ch? b?o ??nh theo xu h??ng
            if key in ['ema_cross', 'trend_sma']:
                confidence *= adx_multiplier
            final_score += direction * confidence * self.weights.get(key, 0)

        action = "HOLD"
        if final_score > 0.6:
            action = "BUY"
        elif final_score < -0.6:
            action = "SELL"

        # Simulate Notification System
        if action != "HOLD":
            print(f"[ALERT] T?n hi?u {action} ph?t hi?n! Score: {final_score:.2f}")

        # --- RISK MANAGEMENT (Bonus) ---
        atr_value = current.get('ATR_14', 0)
        stop_loss = 0
        take_profit = 0
        if action == "BUY" and atr_value > 0:
            stop_loss = current['close'] - (1.5 * atr_value)
            take_profit = current['close'] + (3.0 * atr_value)  # RR 1:2
        elif action == "SELL" and atr_value > 0:
            stop_loss = current['close'] + (1.5 * atr_value)
            take_profit = current['close'] - (3.0 * atr_value)

        return {
            "action": action,
            "final_score": round(final_score, 2),
            "trends": {"short_term": predicted_trend_short, "mid_term": predicted_trend_mid},
            "risk_management": {
                "stop_loss": round(stop_loss, 2) if stop_loss else "N/A",
                "take_profit": round(take_profit, 2) if take_profit else "N/A"
            },
            "details": {k: f"{(v[0] * v[1]):.2f}" for k, v in signals.items()},
            "current_price": current['close']
        }