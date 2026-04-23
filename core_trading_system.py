import pandas as pd
import pandas_ta as ta
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ==========================================
# 0. L?T N?N LOGGING T?P TRUNG
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.FileHandler("quant_trading.log"), logging.StreamHandler()]
)
logger = logging.getLogger("QuantCore")


# ==========================================
# 1. ADAPTIVE LEARNING ENGINE
# ==========================================
class AdaptiveLearningEngine:
    def __init__(self):
        # Tr?ng s? m?c ??nh
        self.weights = {
            "macd": 0.3,
            "rsi": 0.3,
            "ema_cross": 0.2,
            "mtf_trend": 0.2
        }
        self.learning_rate = 0.05
        self.history_records = 0

    def update_weights(self, is_win: bool, signal_snapshot: dict):
        """C?p nh?t tr?ng s? d?a tr?n k?t qu? trade (Heuristic Learning)"""
        self.history_records += 1
        adjustment = self.learning_rate if is_win else -self.learning_rate

        for key in self.weights.keys():
            # N?u indicator ??ng thu?n v?i signal v? th?ng -> t?ng weight, thua -> gi?m
            if key in signal_snapshot and signal_snapshot[key] > 0:
                self.weights[key] += adjustment

        # Normalize l?i ?? t?ng lu?n = 1.0
        total = sum(v for v in self.weights.values() if v > 0)
        for k in self.weights.keys():
            self.weights[k] = max(0.01, self.weights[k] / total)  # Kh?ng cho weight v? ?m

        if self.history_records % 50 == 0:
            logger.info(f"Updated Weights after {self.history_records} trades: {self.weights}")


# ==========================================
# 2. INDICATOR ENGINE
# ==========================================
class IndicatorEngine:
    @staticmethod
    def calculate_safely(df: pd.DataFrame) -> pd.DataFrame:
        """T?nh to?n c?c ch? b?o, c? try-except ch?ng crash"""
        try:
            # Ch?y pandas-ta chi?n l??c
            df.ta.ema(length=9, append=True)
            df.ta.ema(length=21, append=True)
            df.ta.rsi(length=14, append=True)
            df.ta.macd(fast=12, slow=26, signal=9, append=True)
            df.ta.bbands(length=20, std=2, append=True)
            df.ta.atr(length=14, append=True)

            # ??m b?o kh?ng hardcode b?ng c?ch t? t?m c?t
            cols = df.columns
            df['EMA_9'] = df[[c for c in cols if 'EMA_9' in c][0]]
            df['EMA_21'] = df[[c for c in cols if 'EMA_21' in c][0]]
            df['RSI_14'] = df[[c for c in cols if 'RSI' in c][0]]
            df['MACD_Hist'] = df[[c for c in cols if 'MACDh' in c][0]]
            df['ATR'] = df[[c for c in cols if 'ATRr' in c][0]]
            df['BB_Upper'] = df[[c for c in cols if 'BBU' in c][0]]
            df['BB_Lower'] = df[[c for c in cols if 'BBL' in c][0]]

            return df.fillna(method='bfill')
        except Exception as e:
            logger.error(f"Indicator calculation failed: {e}")
            return df


# ==========================================
# 3. PREDICTION ENGINE
# ==========================================
class PredictionEngine:
    def __init__(self, atr_multiplier: float = 1.5):
        self.atr_multi = atr_multiplier

    def predict(self, current_row: pd.Series, direction: str) -> dict:
        """D? ph?ng Range KH?NG nh?y lo?n, d?a tr?n Volatility th?c t?"""
        try:
            mid_price = current_row['EMA_9']
            atr = current_row['ATR']
            range_width = atr * self.atr_multi

            if direction == "BULL":
                p_min = mid_price
                p_max = mid_price + range_width
            elif direction == "BEAR":
                p_min = mid_price - range_width
                p_max = mid_price
            else:  # HOLD / Sideway
                p_min = mid_price - (range_width / 2)
                p_max = mid_price + (range_width / 2)

            return {
                "direction": direction,
                "range": {"min": round(p_min, 2), "max": round(p_max, 2)},
                "confidence": round(1.0 - (atr / mid_price) * 100, 2)  # ATR c?ng nh?, t? tin c?ng cao
            }
        except KeyError:
            return {"direction": "UNKNOWN", "range": {"min": 0, "max": 0}, "confidence": 0}


# ==========================================
# 4. SIGNAL ENGINE (SMOOTHING & COOLDOWN)
# ==========================================
class SignalEngine:
    def __init__(self, learning_engine: AdaptiveLearningEngine):
        self.learning = learning_engine
        self.smoothed_score = 0.0
        self.consecutive_count = 0
        self.last_raw_direction = "HOLD"
        self.cooldown_counter = 0

    def generate_signal(self, row_1m: pd.Series, row_5m: pd.Series, row_15m: pd.Series) -> dict:
        # N?u ?ang cooldown, ?p HOLD
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return {"action": "HOLD", "score": self.smoothed_score,
                    "reason": f"Cooldown ({self.cooldown_counter} left)"}

        # 1. T?nh to?n ?i?m s? c? s? (Base Score) [-1 to 1]
        score = 0.0

        # MACD Normalize
        macd_val = np.clip(row_1m['MACD_Hist'] / (row_1m['ATR'] * 0.5 + 1e-9), -1, 1)
        score += macd_val * self.learning.weights['macd']

        # RSI Normalize (Scale 0-100 to -1 to 1)
        rsi_val = (row_1m['RSI_14'] - 50) / 50
        score += rsi_val * self.learning.weights['rsi']

        # EMA Cross
        ema_val = 1 if row_1m['EMA_9'] > row_1m['EMA_21'] else -1
        score += ema_val * self.learning.weights['ema_cross']

        # Multi-Timeframe Trend (15m Macro)
        mtf_val = 1 if row_15m['EMA_9'] > row_15m['EMA_21'] else -1
        score += mtf_val * self.learning.weights['mtf_trend']

        # 2. EMA Smoothing cho Score (alpha = 0.4 -> chu k? ~ 5 n?n)
        self.smoothed_score = (0.4 * score) + (0.6 * self.smoothed_score)

        # 3. Confirmation Logic (Ch?ng nhi?u)
        raw_direction = "HOLD"
        if self.smoothed_score > 0.3:
            raw_direction = "BUY"
        elif self.smoothed_score < -0.3:
            raw_direction = "SELL"

        if raw_direction == self.last_raw_direction and raw_direction != "HOLD":
            self.consecutive_count += 1
        else:
            self.consecutive_count = 1
            self.last_raw_direction = raw_direction

        # Ch? ra t?n hi?u th?t n?u ?? 3 n?n confirm
        final_action = "HOLD"
        reason = "Confirming trend"
        if self.consecutive_count >= 3:
            final_action = raw_direction
            reason = "Trend confirmed"

        return {
            "action": final_action,
            "score": round(self.smoothed_score, 3),
            "reason": reason,
            "snapshot": {"macd": macd_val, "rsi": rsi_val, "ema_cross": ema_val}
        }

    def trigger_cooldown(self, periods: int = 5):
        self.cooldown_counter = periods
        self.consecutive_count = 0


# ==========================================
# 5. TRADE ENGINE (SIMULATION)
# ==========================================
class TradeEngine:
    def __init__(self, initial_capital: float = 100.0, risk_pct: float = 0.02):
        self.capital = initial_capital
        self.risk_pct = risk_pct
        self.current_trade = None
        self.trade_history = []

    def process_tick(self, timestamp: str, high: float, low: float, close: float) -> dict:
        """Ki?m tra xem gi? ?? ch?m TP / SL ch?a"""
        if not self.current_trade:
            return {"status": "NONE"}

        trade = self.current_trade
        is_closed = False
        result = ""

        if trade['type'] == "LONG":
            if low <= trade['sl']:
                is_closed = True
                result = "LOSS"
                exit_price = trade['sl']
            elif high >= trade['tp']:
                is_closed = True
                result = "WIN"
                exit_price = trade['tp']
        elif trade['type'] == "SHORT":
            if high >= trade['sl']:
                is_closed = True
                result = "LOSS"
                exit_price = trade['sl']
            elif low <= trade['tp']:
                is_closed = True
                result = "WIN"
                exit_price = trade['tp']

        if is_closed:
            pnl = (exit_price - trade['entry']) / trade['entry'] * trade['size']
            if trade['type'] == "SHORT": pnl = -pnl

            self.capital += pnl
            trade.update({"exit_time": timestamp, "exit_price": exit_price, "pnl": round(pnl, 2), "result": result,
                          "status": "CLOSED"})
            self.trade_history.append(trade)

            # Tr? v? k?t qu? ?? b?o cho Signal Engine & Learning Engine
            closed_trade = self.current_trade
            self.current_trade = None
            return closed_trade

        return trade

    def open_position(self, timestamp: str, price: float, direction: str, atr: float):
        if self.current_trade: return  # ?ang c? l?nh th? kh?ng m? th?m

        # Risk Management Sizing
        size = self.capital * self.risk_pct

        # ??t TP / SL linh ho?t theo Volatility
        sl_dist = atr * 2.0
        tp_dist = atr * 3.0  # R:R = 1:1.5

        if direction == "BUY":
            trade_type = "LONG"
            sl, tp = price - sl_dist, price + tp_dist
        else:
            trade_type = "SHORT"
            sl, tp = price + sl_dist, price - tp_dist

        self.current_trade = {
            "entry_time": timestamp,
            "entry": price,
            "type": trade_type,
            "size": size,
            "sl": round(sl, 2),
            "tp": round(tp, 2),
            "status": "OPEN"
        }


# ==========================================
# 6. ORCHESTRATOR / SYSTEM BINDING
# ==========================================
class TradingOrchestrator:
    def __init__(self):
        self.learning = AdaptiveLearningEngine()
        self.signal = SignalEngine(self.learning)
        self.predictor = PredictionEngine()
        self.trade_sim = TradeEngine(initial_capital=100.0)
        self.forward_tests = []

    def on_new_data(self, df_1m: pd.DataFrame, df_5m: pd.DataFrame, df_15m: pd.DataFrame):
        try:
            # 1. T?nh to?n
            df_1 = IndicatorEngine.calculate_safely(df_1m).iloc[-1]
            df_5 = IndicatorEngine.calculate_safely(df_5m).iloc[-1]
            df_15 = IndicatorEngine.calculate_safely(df_15m).iloc[-1]
            current_price = df_1['close']
            ts = str(datetime.now())

            # 2. Check l?nh ?ang ch?y (Simulation)
            trade_status = self.trade_sim.process_tick(ts, df_1['high'], df_1['low'], df_1['close'])
            if trade_status.get("status") == "CLOSED":
                # Feedback loop
                is_win = trade_status['result'] == "WIN"
                # (? th?c t? b?n pass snapshot ???c l?u l?i l?c m? l?nh v?o ??y)
                self.learning.update_weights(is_win, {"macd": 1})
                self.signal.trigger_cooldown(5)

            # 3. T?nh Signal & Prediction
            sig_res = self.signal.generate_signal(df_1, df_5, df_15)

            pred_dir = "BULL" if sig_res['score'] > 0 else ("BEAR" if sig_res['score'] < 0 else "SIDEWAY")
            prediction = self.predictor.predict(df_1, pred_dir)

            # 4. Forward Testing Eval (T?c th?)
            # Gi? ??nh: L?u prediction tr??c ?? v? so s?nh ngay v?i gi? hi?n t?i
            if len(self.forward_tests) > 0:
                last_pred = self.forward_tests[-1]
                hit = last_pred['range']['min'] <= current_price <= last_pred['range']['max']
                last_pred['hit'] = hit
                last_pred['actual'] = current_price

            self.forward_tests.append({"time": ts, "range": prediction['range']})

            # 5. Execute Trade n?u ?? ?i?u ki?n
            if sig_res['action'] in ["BUY", "SELL"]:
                self.trade_sim.open_position(ts, current_price, sig_res['action'], df_1['ATR'])

            # 6. Structured Logging
            log_data = {
                "time": ts, "price": current_price,
                "signal": sig_res['action'], "score": sig_res['score'],
                "prediction": prediction,
                "indicators": {
                    "EMA_9": round(df_1['EMA_9'], 2),
                    "RSI": round(df_1['RSI_14'], 2),
                    "ATR": round(df_1['ATR'], 2)
                }
            }
            logger.info(f"State: {log_data}")

            return {
                "signal": sig_res['action'],
                "score": sig_res['score'],
                "confidence": prediction['confidence'],
                "prediction": prediction,
                "trade": self.trade_sim.current_trade or {"status": "NONE"},
                "indicators": log_data['indicators']
            }

        except Exception as e:
            logger.error(f"Tick error: {e}", exc_info=True)
            raise


# ==========================================
# 7. FASTAPI ENDPOINTS
# ==========================================
app = FastAPI(title="Production Quant Engine")
system = TradingOrchestrator()


# (Mock Data ?? test API - Trong th?c t? b?n fetch t? Binance)
def get_mock_df():
    df = pd.DataFrame(np.random.randn(100, 5) * 10 + 78000, columns=['open', 'high', 'low', 'close', 'volume'])
    df['high'] = df['close'] + 50
    df['low'] = df['close'] - 50
    return df


@app.get("/api/v1/bot/status")
async def get_status():
    """Endpoint ???c Web UI g?i m?i gi?y/ph?t"""
    try:
        # Thay ?o?n n?y b?ng code l?y d? li?u th?t t? db/binance memory
        df1, df5, df15 = get_mock_df(), get_mock_df(), get_mock_df()

        result = system.on_new_data(df1, df5, df15)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))