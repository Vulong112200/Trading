import pandas as pd
import json
import logging
from datetime import datetime
import numpy as np

# C?u h?nh logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SignalEngine")


# ==========================================
# 1. ADAPTIVE SCORER & FEEDBACK LOOP
# ==========================================
class AdaptiveScorer:
    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        # Tr?ng s? m?c ??nh ban ??u
        self.weights = {'EMA': 1.0, 'MACD': 1.0, 'RSI': 1.0, 'BB': 1.0, 'VOL': 1.0}
        self.pending_predictions = []
        self.log_file = f"ml_training_data_{self.symbol}_{self.timeframe}.jsonl"

    def normalize_weights(self):
        total = sum(self.weights.values())
        return {k: v / total for k, v in self.weights.items()}

    def register_prediction(self, pred_data: dict):
        self.pending_predictions.append(pred_data)

    def evaluate_and_learn(self, current_time_sec: int, current_price: float):
        """
        Logic Evaluation m?i: ?u ti?n Price Range v? Direction.
        """
        still_pending = []
        for p in self.pending_predictions:
            if current_time_sec >= p['target_time_sec']:
                # 1. Ki?m tra Direction
                actual_move = current_price - p['start_price']
                correct_direction = (p['direction'] == "BULLISH" and actual_move > 0) or \
                                    (p['direction'] == "BEARISH" and actual_move < 0)

                # 2. Ki?m tra Hit Range
                hit_range = (p['range_min'] <= current_price <= p['range_max'])

                # 3. Ph?n lo?i k?t qu? (WIN / PARTIAL / LOSS)
                if hit_range:
                    final_result = "WIN"
                    lr_factor = 1.0  # Th??ng chu?n
                elif correct_direction:
                    final_result = "PARTIAL"
                    lr_factor = 0.5  # Th??ng nh?
                else:
                    final_result = "LOSS"
                    lr_factor = -1.0  # Ph?t

                # 4. Adaptive Weight Training (Heuristic Learning)
                lr = 0.05
                for ind, sig_val in p['raw_signals'].items():
                    # Ki?m tra indicator ?? c? ?ng h? h??ng d? ?o?n kh?ng
                    is_ind_correct = (sig_val > 0 and p['direction'] == "BULLISH") or \
                                     (sig_val < 0 and p['direction'] == "BEARISH")

                    if is_ind_correct:
                        self.weights[ind] *= (1 + (lr * lr_factor))

                    # Gi?i h?n weight tr?nh tri?t ti?u ho?c b?ng n?
                    self.weights[ind] = max(0.2, min(3.0, self.weights[ind]))

                # 5. Ghi Log JSONL ph?c v? Training
                log_entry = {
                    "time": datetime.fromtimestamp(p['target_time_sec']).isoformat(),
                    "symbol": self.symbol,
                    "predict_direction": p['direction'],
                    "range": {"min": p['range_min'], "max": p['range_max']},
                    "actual_price": round(current_price, 2),
                    "hit_range": hit_range,
                    "correct_direction": correct_direction,
                    "final_result": final_result,
                    "indicators_snapshot": p['indicators_snapshot'],
                    "volatility_atr": p['atr_snapshot']
                }
                with open(self.log_file, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
            else:
                still_pending.append(p)
        self.pending_predictions = still_pending


# ==========================================
# 2. POSITION MANAGER
# ==========================================
class PositionManager:
    def __init__(self):
        self.state = "NONE"
        self.entry_price = 0.0
        self.cooldown_counter = 0

    def open_position(self, side: str, price: float):
        self.state = "LONG" if side in ["BUY", "WEAK_BUY"] else "SHORT"
        self.entry_price = price

    def close_position(self, cooldown=10):
        self.state = "NONE"
        self.entry_price = 0.0
        self.cooldown_counter = cooldown


# ==========================================
# 3. ADVANCED SIGNAL ENGINE
# ==========================================
class AdvancedSignalEngine:
    def __init__(self, symbol: str, interval: str):
        self.symbol = symbol
        self.interval = interval
        self.pos_manager = PositionManager()
        self.scorer = AdaptiveScorer(symbol, interval)
        self.prev_smoothed_score = 0.0
        self.confirm_counter = 0
        self.hold_counter = 0
        self.last_candle_time = None
        self.current_direction = 0

    def _safe_get(self, series, default=0.0):
        """Ch?ng l?i NaN v? RSI = 0"""
        if series is None: return default
        val = series if isinstance(series, (float, int)) else series.iloc[-1]
        return float(val) if pd.notna(val) and val != 0 else default

    def generate_signal(self, df: pd.DataFrame):
        """
        Method c?t l?i: X? l? d? li?u, d? b?o bi?n ??ng v? ??a ra t?n hi?u giao d?ch.
        ?? t?i ?u ?? tr?nh l?i N/A v? ??ng b? ho?n to?n v?i UI.
        """
        # 1. KH?I T?O ??I T??NG PH?N H?I M?C ??NH (Ng?n l?i undefined/N/A)
        res = {
            "action": "HOLD",
            "score": 0.0,
            "confidence": 0.0,
            "reason": "Initializing system...",
            "prediction": {
                "direction": "UNKNOWN",
                "target_mid": 0.0,
                "range": {"min": 0.0, "max": 0.0}
            },
            "position": {"state": "NONE", "entry_price": 0.0},
            "indicators": {},
            "debug_info": {"hold_duration": 0, "current_threshold": 0.35}
        }

        # Ki?m tra d? li?u t?i thi?u ?? tr?nh crash
        if len(df) < 3:
            res["reason"] = "Waiting for data stream..."
            return res

        # 2. TR?CH XU?T D? LI?U C? B?N
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        current_price = float(curr['close'])
        current_time_sec = int(curr['timestamp'].timestamp())

        # 3. LOGIC D? B?O GI? (VOLATILITY PREDICTION)
        # Ph?n n?y ch?y ??c l?p, kh?ng c?n ??i n?n 50 ?? tr?nh hi?n N/A
        atr = self._safe_get(curr.get('ATR_14'), current_price * 0.002)
        ema9 = self._safe_get(curr.get('EMA_9'), current_price)
        ema9_prev = self._safe_get(prev.get('EMA_9'), current_price)

        # T?nh ?? d?c ?? d? b?o h??ng ?i trong 5 n?n t?i
        slope = ema9 - ema9_prev
        target_mid = current_price + (slope * 5)
        range_h = atr * 0.8  # ?? r?ng v?ng k? v?ng d?a tr?n bi?n ??ng th?c t?

        res["prediction"] = {
            "direction": "BULLISH" if slope > 0.01 else ("BEARISH" if slope < -0.01 else "SIDEWAY"),
            "target_mid": round(target_mid, 2),
            "range": {
                "min": round(target_mid - range_h, 2),
                "max": round(target_mid + range_h, 2)
            }
        }

        # 4. KI?M TRA ?I?U KI?N WARM-UP (Ch? ch?n Signal, kh?ng ch?n Prediction)
        if len(df) < 50:
            res["action"] = "HOLD"
            res["reason"] = f"Warming up indicators ({len(df)}/50)"
            return res

        # 5. C?P NH?T TR?NG TH?I H?C T?P (LEARNING)
        # Bot t? ??nh gi? n?n c? tr??c khi t?nh n?n m?i
        self.scorer.evaluate_and_learn(current_time_sec, current_price)

        # Gi?m th?i gian ch? gi?a c?c l?nh
        if self.pos_manager.cooldown_counter > 0:
            self.pos_manager.cooldown_counter -= 1

        # 6. T?NH TO?N T?N HI?U TH? (RAW SIGNALS)
        rsi_val = self._safe_get(curr.get('RSI_14'), 50)
        raw_sigs = {
            'EMA': 1.0 if ema9 > self._safe_get(curr.get('EMA_21')) else -1.0,
            'MACD': 1.0 if self._safe_get(curr.get('MACD_H')) > 0 else -1.0,
            'RSI': 1.0 if rsi_val < 40 else (-1.0 if rsi_val > 60 else 0.0),
            'BB': 1.0 if current_price <= self._safe_get(curr.get('BB_L')) else (
                -1.0 if current_price >= self._safe_get(curr.get('BB_U')) else 0.0),
            'VOL': 1.0 if self._safe_get(curr.get('OBV')) > self._safe_get(prev.get('OBV')) else -1.0
        }

        # 7. CH?M ?I?M TH?CH NGHI & L?M M??T (ADAPTIVE SCORING)
        weights = self.scorer.normalize_weights()
        raw_score = sum(raw_sigs[k] * weights[k] for k in raw_sigs.keys())

        # L?m m??t ?i?m s? ?? tr?nh t?n hi?u gi? (Whipsaw)
        smoothed_score = (0.35 * raw_score) + (0.65 * self.prev_smoothed_score)
        self.prev_smoothed_score = smoothed_score

        # C? ch? Anti-stuck: N?u ??ng ngo?i qu? l?u, h? th?p ti?u chu?n ?? t?m c? h?i
        self.hold_counter = self.hold_counter + 1 if self.pos_manager.state == "NONE" else 0
        decay = min(0.15, max(0, (self.hold_counter - 15) * 0.01))  # B?t ??u gi?m sau 15 n?n HOLD
        current_thresh = 0.38 - decay

        # 8. B? L?C SIDEWAY & X?C NH?N T?N HI?U
        action = "HOLD"
        reason = "No clear edge"

        bb_u = self._safe_get(curr.get('BB_U'))
        bb_l = self._safe_get(curr.get('BB_L'))
        bb_width = (bb_u - bb_l) / current_price
        is_sideway = bb_width < 0.0012  # Th? tr??ng qu? h?p, kh?ng n?n v?o l?nh

        if self.pos_manager.cooldown_counter == 0 and not is_sideway:
            if abs(smoothed_score) > current_thresh:
                self.confirm_counter += 1
                # C?n 2 n?n li?n ti?p ??ng thu?n ?? v?o l?nh
                if self.confirm_counter >= 2:
                    if smoothed_score > 0:
                        action = "BUY" if smoothed_score > 0.5 else "WEAK_BUY"
                    else:
                        action = "SELL" if smoothed_score < -0.5 else "WEAK_SELL"
                    reason = f"Trend confirmed (Score: {round(smoothed_score, 2)})"
            else:
                self.confirm_counter = 0
        elif is_sideway:
            reason = "Market Squeeze (Sideway)"

        # C?p nh?t tr?ng th?i l?nh v?o Manager
        if action != "HOLD" and self.pos_manager.state == "NONE":
            self.pos_manager.open_position(action, current_price)

        # 9. L?U D? ?O?N V? PH?N H?I UI
        indicators = {
            "RSI": round(rsi_val, 1),
            "EMA_9": round(ema9, 2),
            "EMA_21": round(self._safe_get(curr.get('EMA_21')), 2),
            "MACD": round(self._safe_get(curr.get('MACD_H')), 4),
            "BB_U": round(bb_u, 2),
            "BB_L": round(bb_l, 2)
        }

        # ??ng k? d? ?o?n v?o b? ch?m ?i?m (ch? ??ng k? khi c? n?n m?i)
        # L?u d? b?o ?? ch?m ?i?m (Ph?i ??m b?o c? ??y ?? key ?? kh?ng crash)
        candle_ts = int(curr['timestamp'].timestamp())
        if self.last_candle_time != candle_ts:
            self.scorer.register_prediction({
                "target_time_sec": candle_ts + (60 * 5),
                "start_price": current_price,
                "direction": res["prediction"]["direction"],
                "range_min": res["prediction"]["range"]["min"],
                "range_max": res["prediction"]["range"]["max"],
                "raw_signals": raw_sigs,
                "indicators_snapshot": indicators,  # TH?M D?NG N?Y ?? H?T L?I
                "atr_snapshot": round(atr, 2)  # TH?M D?NG N?Y ?? ?? D? LI?U
            })
            self.last_candle_time = candle_ts

        # G?p t?t c? v?o k?t qu? cu?i c?ng
        res.update({
            "action": action,
            "score": round(smoothed_score, 3),
            "confidence": round(min(1.0, abs(smoothed_score) * 1.8), 2),
            "reason": reason,
            "indicators": indicators,
            "position": {
                "state": self.pos_manager.state,
                "entry_price": self.pos_manager.entry_price
            },
            "debug_info": {
                "hold_duration": self.hold_counter,
                "current_threshold": round(current_thresh, 3),
                "raw_score": round(raw_score, 3)
            }
        })
        return res