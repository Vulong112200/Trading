import pandas as pd
import json
import logging
import os
from datetime import datetime
from typing import Dict, Optional
import numpy as np

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SignalEngine")


# ==========================================
# 1. ADAPTIVE SCORER & LEARNING SYSTEM
# ==========================================
class AdaptiveScorer:
    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.weights = {'EMA': 1.0, 'MACD': 1.0, 'RSI': 1.0, 'BB': 1.0, 'VWAP': 1.0, 'MTF': 1.5}
        self.pending_predictions = {}
        self.recent_results = []  # Lưu 10 kết quả cho Tracker UI
        self.ml_log_file = f"ml_training_data_{self.symbol}_{self.timeframe}.jsonl"
        self.stats = {"win": 0, "loss": 0, "partial": 0}

        self._load_and_train_from_history()

    def _load_and_train_from_history(self):
        if not os.path.exists(self.ml_log_file):
            with open(self.ml_log_file, "w") as f: pass
            return

        logger.info(f"[{self.symbol}-{self.timeframe}] Warming up model from ML history...")
        try:
            with open(self.ml_log_file, "r") as f:
                lines = f.readlines()

            valid_records = 0
            loaded_results = []

            for line in lines[-500:]:
                if not line.strip(): continue
                data = json.loads(line)

                res = data.get("result")
                direction = data.get("prediction", {}).get("direction", "")
                raw_sigs = data.get("raw_signals", {})

                if not raw_sigs or not res: continue

                # 1. Update stats
                if res == "WIN":
                    self.stats["win"] += 1
                elif res == "LOSS":
                    self.stats["loss"] += 1
                elif res == "PARTIAL":
                    self.stats["partial"] += 1

                # 2. Khôi phục Trọng số
                factor = 1.0 if res == "WIN" else (0.5 if res == "PARTIAL" else -1.0)
                for ind, val in raw_sigs.items():
                    if ind in self.weights:
                        is_correct = (val > 0 and "BULL" in direction) or (val < 0 and "BEAR" in direction)
                        if is_correct: self.weights[ind] *= (1 + (0.02 * factor))
                        self.weights[ind] = max(0.1, min(3.0, self.weights[ind]))

                # 3. Load 10 records cho UI Forward Testing
                time_str = data["time"]
                if "T" in time_str:
                    time_str = datetime.fromisoformat(time_str).strftime('%H:%M:%S')

                loaded_results.insert(0, {
                    "time": time_str,
                    "direction": direction,
                    "range": f"{data['prediction']['range']['min']:.2f} - {data['prediction']['range']['max']:.2f}",
                    "actual_price": data.get("actual_price", 0),
                    "final_result": res
                })
                valid_records += 1

            self.recent_results = loaded_results[:10]  # Lấy 10 dòng gần nhất

            if valid_records > 0:
                logger.info(f"Loaded {valid_records} ML records. WinRate: {self.get_winrate()}%")
        except Exception as e:
            logger.error(f"ML history load error: {e}")

    def get_weights(self):
        total = sum(self.weights.values())
        return {k: round(v / total, 3) for k, v in self.weights.items()}

    def register_prediction(self, target_time: int, payload: dict):
        self.pending_predictions[target_time] = payload

    def evaluate_and_learn(self, current_time_sec: int, closed_price: float):
        resolved = []
        for target_time, p in self.pending_predictions.items():
            if current_time_sec >= target_time:
                direction = p['direction']
                move = closed_price - p['start_price']
                correct_dir = (direction.startswith("BULL") and move > 0) or (direction.startswith("BEAR") and move < 0)
                hit_range = p['range']['min'] <= closed_price <= p['range']['max']

                if hit_range and correct_dir:
                    result = "WIN"
                elif correct_dir:
                    result = "PARTIAL"
                else:
                    result = "LOSS"

                if result == "WIN":
                    self.stats["win"] += 1
                elif result == "LOSS":
                    self.stats["loss"] += 1
                else:
                    self.stats["partial"] += 1

                # ML Learning từ Dự báo
                factor = 1.0 if result == "WIN" else (0.5 if result == "PARTIAL" else -1.0)
                for ind, val in p['raw_signals'].items():
                    is_correct = (val > 0 and direction.startswith("BULL")) or (
                                val < 0 and direction.startswith("BEAR"))
                    if is_correct: self.weights[ind] *= (1 + (0.05 * factor))
                    self.weights[ind] = max(0.1, min(3.0, self.weights[ind]))

                tracker_entry = {
                    "time": datetime.fromtimestamp(target_time).strftime('%H:%M:%S'),
                    "direction": direction,
                    "range": f"{p['range']['min']:.2f} - {p['range']['max']:.2f}",
                    "actual_price": round(closed_price, 2),
                    "final_result": result
                }
                self.recent_results.insert(0, tracker_entry)
                self.recent_results = self.recent_results[:10]  # Giữ 10 record

                log_entry = {
                    "time": datetime.fromtimestamp(target_time).isoformat(),
                    "symbol": self.symbol, "timeframe": self.timeframe,
                    "prediction": {"direction": direction, "range": p['range'], "confidence": p['confidence']},
                    "signal": p['signal'], "score": p['score'],
                    "raw_signals": p['raw_signals'],
                    "indicators_snapshot": p['indicators_snapshot'],
                    "actual_price": round(closed_price, 4), "result": result
                }
                try:
                    with open(self.ml_log_file, "a") as f:
                        f.write(json.dumps(log_entry) + "\n")
                except:
                    pass
                resolved.append(target_time)

        for k in resolved: del self.pending_predictions[k]

    def learn_from_real_trade(self, snapshot: dict, direction: str, result: str):
        """HỌC TỪ TRADE THẬT SỰ (Đóng lệnh chạm TP/SL)"""
        factor = 1.0 if result == "WIN" else -1.0
        for ind, val in snapshot.items():
            is_correct = (val > 0 and direction == "LONG") or (val < 0 and direction == "SHORT")
            if is_correct: self.weights[ind] *= (1 + (0.08 * factor))  # Trọng số học từ Trade thật cao hơn (8%)
            self.weights[ind] = max(0.1, min(3.0, self.weights[ind]))

    def get_winrate(self):
        total = self.stats["win"] + self.stats["loss"]
        return round((self.stats["win"] / total) * 100, 2) if total > 0 else 0.0


# ==========================================
# 2. REAL TRADE SIMULATOR (QUẢN LÝ VỐN CHUẨN)
# ==========================================
class TradeSimulator:
    def __init__(self, symbol: str, timeframe: str, capital=100.0, risk_pct=0.02):
        self.symbol = symbol
        self.timeframe = timeframe
        self.capital = capital
        self.risk_pct = risk_pct
        self.state = "NONE"
        self.trade = {}
        self.history = []
        self.cooldown = 0
        self.just_opened = False  # Fix Bug khớp lệnh ngay tức thì
        self.notifications = []

        self.trade_log_file = f"trade_history_{self.symbol}_{self.timeframe}.jsonl"
        self._load_history()

    def _load_history(self):
        if not os.path.exists(self.trade_log_file):
            with open(self.trade_log_file, "w") as f: pass
            return

        try:
            with open(self.trade_log_file, "r") as f:
                lines = f.readlines()
                for line in lines[-10:]:  # Load 10 lệnh lên UI
                    if not line.strip(): continue
                    data = json.loads(line)
                    self.history.insert(0, data)

                # Khôi phục vốn từ lệnh cuối cùng
                if self.history:
                    self.capital = self.history[0].get("capital_after", self.capital)
            logger.info(f"[{self.symbol}] Loaded trade history. Capital: {self.capital:.2f}$")
        except Exception as e:
            logger.error(f"Trade load error: {e}")

    def open_position(self, timestamp: str, side: str, price: float, atr: float, indicators_snap: dict):
        if self.state != "NONE" or self.cooldown > 0: return

        self.state = "LONG" if side == "BUY" else "SHORT"

        # Risk Management: Sizing
        sl_dist = max(atr * 1.5, price * 0.0005)  # Đảm bảo không chia cho 0
        tp_dist = atr * 3.0

        sl = price - sl_dist if self.state == "LONG" else price + sl_dist
        tp = price + tp_dist if self.state == "LONG" else price - tp_dist

        risk_usd = self.capital * self.risk_pct
        position_size = risk_usd / sl_dist
        volume_usd = position_size * price

        self.trade = {
            "entry_time": timestamp, "entry": price, "position": self.state,
            "sl": round(sl, 4), "tp": round(tp, 4),
            "size": position_size, "volume_usd": volume_usd,
            "snapshot": indicators_snap, "pnl_pct": 0.0, "profit_usd": 0.0
        }
        self.just_opened = True

        # Push Notification
        msg = f"🟢 OPEN {self.state}\nEntry: {price}\nVol: {volume_usd:.2f}$\nTP: {tp:.2f} | SL: {sl:.2f}"
        self.notifications.append(msg)

    def process_tick(self, current_price: float, high: float, low: float) -> Optional[dict]:
        if self.state == "NONE": return None

        # Delay execution: Không đóng lệnh trong cùng 1 tick mở lệnh (giả lập latency)
        if self.just_opened:
            self.just_opened = False
            return None

        t = self.trade
        is_closed = False
        res = ""

        # Live PnL & Profit USD
        mult = 1 if self.state == "LONG" else -1
        t['pnl_pct'] = ((current_price - t['entry']) / t['entry']) * 100 * mult
        t['profit_usd'] = (current_price - t['entry']) * t['size'] * mult

        # Check Hit TP/SL
        if self.state == "LONG":
            if low <= t['sl']:
                is_closed = True; res = "LOSS"; exit_p = t['sl']
            elif high >= t['tp']:
                is_closed = True; res = "WIN"; exit_p = t['tp']
        else:
            if high >= t['sl']:
                is_closed = True; res = "LOSS"; exit_p = t['sl']
            elif low <= t['tp']:
                is_closed = True; res = "WIN"; exit_p = t['tp']

        if is_closed:
            final_profit_usd = (exit_p - t['entry']) * t['size'] * mult
            final_pnl_pct = ((exit_p - t['entry']) / t['entry']) * 100 * mult
            capital_before = self.capital
            self.capital += final_profit_usd

            exit_time = datetime.now().isoformat()

            log_entry = {
                "symbol": self.symbol, "timeframe": self.timeframe,
                "entry_time": t["entry_time"], "exit_time": exit_time,
                "entry_price": t["entry"], "exit_price": round(exit_p, 4),
                "position": t["position"],
                "capital_before": round(capital_before, 2), "capital_after": round(self.capital, 2),
                "risk_pct": self.risk_pct, "position_size": round(t["size"], 6),
                "profit_usd": round(final_profit_usd, 2), "pnl": round(final_pnl_pct, 2),
                "result": res
            }

            # Ghi file
            try:
                with open(self.trade_log_file, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
            except:
                pass

            self.history.insert(0, log_entry)
            self.history = self.history[:10]  # UI cần 10 record

            # Push Notification
            icon = "🔴" if res == "LOSS" else "🔵"
            msg = f"{icon} CLOSED {res}\nPnL: {final_pnl_pct:.2f}%\nProfit: {final_profit_usd:.2f}$"
            self.notifications.append(msg)

            closed_trade_data = {"snapshot": t["snapshot"], "direction": self.state, "result": res}

            self.state = "NONE";
            self.trade = {};
            self.cooldown = 3
            return closed_trade_data

        return None

    def get_notifications(self):
        msgs = self.notifications.copy()
        self.notifications.clear()
        return msgs


# ==========================================
# 3. MASTER SIGNAL ENGINE
# ==========================================
class AdvancedSignalEngine:
    def __init__(self, symbol: str, interval: str):
        self.symbol = symbol;
        self.interval = interval
        self.scorer = AdaptiveScorer(symbol, interval)
        self.trade_sim = TradeSimulator(symbol, interval, capital=100.0)
        self.smoothed_score = 0.0
        self.confirm_counter = 0;
        self.current_dir = "HOLD"
        self.last_candle_time = None
        self.hold_counter = 0

        self.ui_state = {
            "action": "HOLD", "signal": "HOLD", "score": 0.0, "confidence": 0.0, "reason": "Initializing...",
            "prediction": {"direction": "UNKNOWN", "target_mid": 0.0, "mid_price": 0.0,
                           "range": {"min": 0.0, "max": 0.0}},
            "indicators": {}, "weights": self.scorer.get_weights(), "winrate": self.scorer.get_winrate(),
            "tracker": [], "notifications": []
        }

    def _safe_get(self, value, default=0.0):
        try:
            return float(value) if pd.notna(value) else default
        except:
            return default

    def _tf_to_sec(self, tf: str) -> int:
        unit = tf[-1];
        val = int(tf[:-1])
        if unit == 'm': return val * 60
        if unit == 'h': return val * 3600
        return 60

    def generate_signal(self, df: pd.DataFrame, mtf_context: dict):
        if df is None or len(df) < 5: return self.ui_state

        curr = df.iloc[-1];
        prev = df.iloc[-2]
        c_price = self._safe_get(curr.get('close'))
        c_high = self._safe_get(curr.get('high'), c_price)
        c_low = self._safe_get(curr.get('low'), c_price)
        c_time = int(curr['timestamp'].timestamp())
        c_time_str = datetime.fromtimestamp(c_time).strftime('%H:%M:%S')

        # 1. PROCESS TICK (REALTIME EXECUTION)
        closed_trade = self.trade_sim.process_tick(c_price, c_high, c_low)
        if closed_trade:
            # HỌC TỪ TRADE THẬT
            self.scorer.learn_from_real_trade(closed_trade["snapshot"], closed_trade["direction"],
                                              closed_trade["result"])

        atr = self._safe_get(curr.get('ATR_14'), c_price * 0.002)
        ema9 = self._safe_get(curr.get('EMA_9'), c_price)
        ema9_prev = self._safe_get(prev.get('EMA_9'), c_price)
        slope = ema9 - ema9_prev
        mid_price = c_price + (slope * 5)
        range_w = atr * 1.5
        p_dir = "BULLISH" if slope > 0 else "BEARISH"

        # 2. EVENT-DRIVEN: CHỈ CHẠY 1 LẦN KHI NẾN ĐÓNG
        if self.last_candle_time != c_time:
            self.scorer.evaluate_and_learn(c_time, float(prev['close']))

            mtf_score = 0.0
            for tf in ["5m", "15m", "1h"]:
                if tf in mtf_context and len(mtf_context[tf]) > 0:
                    mtf_c = mtf_context[tf].iloc[-1]
                    mtf_ema9 = self._safe_get(mtf_c.get('EMA_9'), c_price)
                    mtf_ema21 = self._safe_get(mtf_c.get('EMA_21'), c_price)
                    mtf_score += 1.0 if mtf_ema9 > mtf_ema21 else -1.0
            mtf_score = mtf_score / 3.0 if mtf_score != 0 else 0

            if self.trade_sim.cooldown > 0: self.trade_sim.cooldown -= 1

            macd_h = self._safe_get(curr.get('MACD_H'), 0.0)
            rsi_val = self._safe_get(curr.get('RSI_14'), 50.0)
            bb_u = self._safe_get(curr.get('BB_U'), c_price)
            bb_l = self._safe_get(curr.get('BB_L'), c_price)
            vwap = self._safe_get(curr.get('VWAP'), c_price)

            raw_sigs = {
                'EMA': 1.0 if ema9 > self._safe_get(curr.get('EMA_21'), c_price) else -1.0,
                'MACD': max(-1.0, min(1.0, macd_h / (atr * 0.5))),
                'RSI': max(-1.0, min(1.0, (rsi_val - 50) / 20)),
                'BB': 1.0 if c_price <= bb_l else (-1.0 if c_price >= bb_u else 0.0),
                'VWAP': 1.0 if c_price > vwap else -1.0, 'MTF': mtf_score
            }

            weights = self.scorer.get_weights()
            raw_score = sum(raw_sigs[k] * weights.get(k, 0.16) for k in raw_sigs.keys())
            self.smoothed_score = (0.35 * raw_score) + (0.65 * self.smoothed_score)

            self.hold_counter = self.hold_counter + 1 if self.trade_sim.state == "NONE" else 0
            decay = min(0.15, max(0, (self.hold_counter - 15) * 0.01))
            current_thresh = 0.38 - decay

            action = "HOLD"
            bb_width = (bb_u - bb_l) / c_price

            if bb_width > 0.0008 and self.trade_sim.cooldown == 0:
                if self.smoothed_score > current_thresh:
                    dir_cand = "BUY"
                elif self.smoothed_score < -current_thresh:
                    dir_cand = "SELL"
                else:
                    dir_cand = "HOLD"

                if dir_cand == self.current_dir and dir_cand != "HOLD":
                    self.confirm_counter += 1
                else:
                    self.confirm_counter = 1; self.current_dir = dir_cand

                if self.confirm_counter >= 2:
                    action = dir_cand
                    # MỞ LỆNH THẬT
                    self.trade_sim.open_position(c_time_str, action, c_price, atr, raw_sigs)

            conf = max(0.1, min(0.95, 1.0 - (atr / c_price) * 10))
            pred = {"direction": p_dir, "mid_price": round(mid_price, 2),
                    "range": {"min": round(mid_price - atr * 1.5, 2), "max": round(mid_price + atr * 1.5, 2)}}
            indicators_snap = {"EMA_9": round(ema9, 2), "EMA_21": round(self._safe_get(curr.get('EMA_21')), 2),
                               "RSI": round(rsi_val, 2), "MACD": round(macd_h, 4), "ATR": round(atr, 2)}

            self.scorer.register_prediction(c_time + (self._tf_to_sec(self.interval) * 5), {
                "direction": p_dir, "range": pred['range'], "confidence": conf, "start_price": c_price,
                "raw_signals": raw_sigs, "indicators_snapshot": indicators_snap, "atr_snapshot": atr,
                "signal": action, "score": self.smoothed_score
            })

            reason_txt = "Monitoring..."
            if action != "HOLD":
                reason_txt = "Trend Confirmed"
            elif self.trade_sim.cooldown > 0:
                reason_txt = f"Cooldown ({self.trade_sim.cooldown})"

            pending_list = [{"time": datetime.fromtimestamp(k).strftime('%H:%M:%S'), "direction": v['direction'],
                             "range": f"{v['range']['min']:.2f} - {v['range']['max']:.2f}", "actual_price": 0.0,
                             "final_result": "WAITING"} for k, v in list(self.scorer.pending_predictions.items())[-3:]]
            combined_tracker = pending_list + self.scorer.recent_results

            self.ui_state.update({
                "action": action, "signal": action, "score": round(self.smoothed_score, 3),
                "confidence": round(conf, 3),
                "prediction": pred, "indicators": indicators_snap, "reason": reason_txt, "weights": weights,
                "winrate": self.scorer.get_winrate(), "tracker": combined_tracker[:10]
            })
            self.last_candle_time = c_time

        # 3. GỘP STATE CẬP NHẬT REALTIME CHO FRONTEND
        res_out = self.ui_state.copy()
        res_out["notifications"] = self.trade_sim.get_notifications()
        res_out["trade_history"] = self.trade_sim.history[:10]

        t = self.trade_sim.trade
        res_out["trade"] = {
            "position_status": "OPEN" if self.trade_sim.state != "NONE" else "NONE",
            "entry": round(t.get('entry', 0.0), 2) if self.trade_sim.state != "NONE" else None,
            "tp": round(t.get('tp', 0.0), 2) if self.trade_sim.state != "NONE" else None,
            "sl": round(t.get('sl', 0.0), 2) if self.trade_sim.state != "NONE" else None,
            "pnl": round(t.get('pnl_pct', 0.0), 2),
            "profit_usd": round(t.get('profit_usd', 0.0), 2),
            "position_size": round(t.get('size', 0.0), 4),
            "capital": round(self.trade_sim.capital, 2)
        }
        return res_out