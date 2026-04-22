import pandas as pd
import json
import logging
import os
from datetime import datetime
import numpy as np

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SignalEngine")


# ==========================================
# 1. ADAPTIVE SCORER & FEEDBACK LOOP
# ==========================================
class AdaptiveScorer:
    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.weights = {'EMA': 1.0, 'MACD': 1.0, 'RSI': 1.0, 'BB': 1.0, 'VOL': 1.0}
        self.pending_predictions = []
        self.recent_results = []  # Lưu kết quả để hiển thị lên bảng Tracker trên UI
        self.log_file = f"ml_training_data_{self.symbol}_{self.timeframe}.jsonl"

        # Ép tạo file ngay khi khởi động để tránh lỗi chưa có file
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                pass

    def normalize_weights(self):
        total = sum(self.weights.values())
        return {k: v / total for k, v in self.weights.items()}

    def register_prediction(self, pred_data: dict):
        self.pending_predictions.append(pred_data)

    def evaluate_and_learn(self, current_time_sec: int, current_price: float):
        still_pending = []
        for p in self.pending_predictions:
            if current_time_sec >= p['target_time_sec']:
                # 1. Kiểm tra Direction
                actual_move = current_price - p['start_price']
                correct_direction = (p['direction'] == "BULLISH" and actual_move > 0) or \
                                    (p['direction'] == "BEARISH" and actual_move < 0)

                # 2. Kiểm tra Hit Range
                hit_range = (p['range_min'] <= current_price <= p['range_max'])

                # 3. Phân loại kết quả (WIN / PARTIAL / LOSS)
                if hit_range:
                    final_result = "WIN"
                    lr_factor = 1.0
                elif correct_direction:
                    final_result = "PARTIAL"
                    lr_factor = 0.5
                else:
                    final_result = "LOSS"
                    lr_factor = -1.0

                # 4. Adaptive Weight Training
                lr = 0.05
                for ind, sig_val in p['raw_signals'].items():
                    is_ind_correct = (sig_val > 0 and p['direction'] == "BULLISH") or \
                                     (sig_val < 0 and p['direction'] == "BEARISH")
                    if is_ind_correct:
                        self.weights[ind] *= (1 + (lr * lr_factor))
                    self.weights[ind] = max(0.2, min(3.0, self.weights[ind]))

                # Cập nhật kết quả vào Tracker List cho UI (Giữ 5 kết quả mới nhất)
                tracker_entry = {
                    "time": datetime.fromtimestamp(p['target_time_sec']).strftime('%H:%M:%S'),
                    "type": p['direction'],
                    "target": f"{p['range_min']}-{p['range_max']}",
                    "actual": round(current_price, 2),
                    "result": final_result
                }
                self.recent_results.insert(0, tracker_entry)
                self.recent_results = self.recent_results[:5]

                # 5. Ghi Log JSONL
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
                try:
                    with open(self.log_file, "a") as f:
                        f.write(json.dumps(log_entry) + "\n")
                except Exception as e:
                    logger.error(f"Cannot write log: {e}")
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
        self.entry_price = float(price)

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

    def _safe_get(self, series, default=0.0):
        if series is None: return default
        val = series if isinstance(series, (float, int)) else series.iloc[-1]
        return float(val) if pd.notna(val) and val != 0 else default

    def generate_signal(self, df: pd.DataFrame):
        # 1. KHỞI TẠO ĐỐI TƯỢNG PHẢN HỒI (Giá trị mặc định để UI không crash)
        res = {
            "action": "HOLD",
            "score": 0.0,
            "confidence": 0.0,
            "reason": "Initializing...",
            "prediction": {"direction": "UNKNOWN", "target_mid": 0.0, "range": {"min": 0.0, "max": 0.0}},
            "position": {"state": "NONE", "entry_price": 0.0, "pnl": 0.0},
            "indicators": {},
            "tracker": [],
            "debug_info": {"hold_duration": 0, "current_threshold": 0.35}
        }

        if len(df) < 3:
            return res

        curr = df.iloc[-1]
        prev = df.iloc[-2]
        current_price = float(curr['close'])
        current_time_sec = int(curr['timestamp'].timestamp())

        # 2. DỰ BÁO GIÁ (Chạy độc lập để chart luôn có tia Target)
        atr = self._safe_get(curr.get('ATR_14'), current_price * 0.002)
        ema9 = self._safe_get(curr.get('EMA_9'), current_price)
        ema9_prev = self._safe_get(prev.get('EMA_9'), current_price)

        slope = ema9 - ema9_prev
        target_mid = current_price + (slope * 5)
        range_h = atr * 0.8

        pred_dir = "BULLISH" if slope > 0.01 else ("BEARISH" if slope < -0.01 else "SIDEWAY")

        res["prediction"] = {
            "direction": pred_dir,
            "target_mid": round(target_mid, 2),
            "range": {
                "min": round(target_mid - range_h, 2),
                "max": round(target_mid + range_h, 2)
            }
        }

        # 3. TÍNH PNL THỰC TẾ
        pnl = 0.0
        if self.pos_manager.state != "NONE" and self.pos_manager.entry_price > 0:
            if self.pos_manager.state in ["BUY", "LONG"]:
                pnl = ((current_price - self.pos_manager.entry_price) / self.pos_manager.entry_price) * 100
            elif self.pos_manager.state in ["SELL", "SHORT"]:
                pnl = ((self.pos_manager.entry_price - current_price) / self.pos_manager.entry_price) * 100
        res["position"]["pnl"] = round(pnl, 2)

        # 4. KIỂM TRA WARM-UP (Dưới 50 nến thì không xuất Signal Mua/Bán)
        if len(df) < 50:
            res["reason"] = f"Warming up ({len(df)}/50)"
            return res

        # 5. HỌC TẬP TỪ QUÁ KHỨ & CẬP NHẬT COOLDOWN
        try:
            self.scorer.evaluate_and_learn(current_time_sec, current_price)
        except Exception as e:
            logger.error(f"Scorer Evaluation Error: {e}")

        if self.pos_manager.cooldown_counter > 0:
            self.pos_manager.cooldown_counter -= 1

        # 6. TÍNH CHỈ BÁO & SCORING
        rsi_val = self._safe_get(curr.get('RSI_14'), 50)
        raw_sigs = {
            'EMA': 1.0 if ema9 > self._safe_get(curr.get('EMA_21')) else -1.0,
            'MACD': 1.0 if self._safe_get(curr.get('MACD_H')) > 0 else -1.0,
            'RSI': 1.0 if rsi_val < 40 else (-1.0 if rsi_val > 60 else 0.0),
            'BB': 1.0 if current_price <= self._safe_get(curr.get('BB_L')) else (
                -1.0 if current_price >= self._safe_get(curr.get('BB_U')) else 0.0),
            'VOL': 1.0 if self._safe_get(curr.get('OBV')) > self._safe_get(prev.get('OBV')) else -1.0
        }

        weights = self.scorer.normalize_weights()
        raw_score = sum(raw_sigs[k] * weights[k] for k in raw_sigs.keys())

        smoothed_score = (0.35 * raw_score) + (0.65 * self.prev_smoothed_score)
        self.prev_smoothed_score = smoothed_score

        self.hold_counter = self.hold_counter + 1 if self.pos_manager.state == "NONE" else 0
        decay = min(0.15, max(0, (self.hold_counter - 15) * 0.01))
        current_thresh = 0.38 - decay

        # 7. XUẤT TÍN HIỆU
        action = "HOLD"
        reason = "Monitoring..."

        bb_u = self._safe_get(curr.get('BB_U'))
        bb_l = self._safe_get(curr.get('BB_L'))
        bb_width = (bb_u - bb_l) / current_price
        is_sideway = bb_width < 0.0012

        if self.pos_manager.cooldown_counter == 0 and not is_sideway:
            if abs(smoothed_score) > current_thresh:
                self.confirm_counter += 1
                if self.confirm_counter >= 2:
                    action = "BUY" if smoothed_score > 0 else "SELL"
                    reason = f"Trend Confirmed (Score: {round(smoothed_score, 2)})"
            else:
                self.confirm_counter = 0
        elif is_sideway:
            reason = "Market Sideway (Squeeze)"

        if action != "HOLD" and self.pos_manager.state == "NONE":
            self.pos_manager.open_position(action, current_price)

        indicators = {
            "RSI": round(rsi_val, 1),
            "EMA_9": round(ema9, 2),
            "EMA_21": round(self._safe_get(curr.get('EMA_21')), 2),
            "MACD": round(self._safe_get(curr.get('MACD_H')), 4)
        }

        # 8. LƯU DỰ ĐOÁN (Chỉ lấy khi chuyển nến mới)
        if self.last_candle_time != current_time_sec:
            self.scorer.register_prediction({
                "target_time_sec": current_time_sec + (60 * 5),
                "start_price": current_price,
                "direction": pred_dir,
                "range_min": res["prediction"]["range"]["min"],
                "range_max": res["prediction"]["range"]["max"],
                "raw_signals": raw_sigs,
                "indicators_snapshot": indicators,
                "atr_snapshot": round(atr, 2)
            })
            self.last_candle_time = current_time_sec

        # 9. ĐÓNG GÓI CHUẨN TRẢ VỀ UI
        res.update({
            "action": action,
            "score": round(smoothed_score, 3),
            "confidence": round(min(1.0, abs(smoothed_score) * 1.8), 2),
            "reason": reason,
            "indicators": indicators,
            "tracker": self.scorer.recent_results,  # QUAN TRỌNG: Gửi list tracker qua UI
            "position": {
                "state": self.pos_manager.state,
                "entry_price": round(self.pos_manager.entry_price, 2),
                "pnl": round(pnl, 2)
            }
        })
        return res