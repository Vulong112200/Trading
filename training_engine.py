import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


logger = logging.getLogger("TrainingEngine")


class AdaptiveTrainingEngine:
    def __init__(self, symbol: str, timeframe: str, update_interval: int = 50):
        self.symbol = symbol
        self.timeframe = timeframe
        self.update_interval = max(10, update_interval)
        self.learning_rate = 0.04
        self.weights = {
            "ema_cross": 1.0,
            "rsi": 1.0,
            "macd": 1.0,
            "bb": 1.0,
            "vwap": 1.0,
            "mtf": 1.0,
        }
        self._buffer: List[Dict] = []
        self.log_path = Path(f"ml_training_data_{symbol}_{timeframe}.jsonl")
        self.trade_log_path = Path(f"trade_log_{symbol}_{timeframe}.jsonl")

    def normalized_weights(self) -> Dict[str, float]:
        total = sum(max(v, 0.01) for v in self.weights.values())
        return {k: max(v, 0.01) / total for k, v in self.weights.items()}

    def register_result(self, indicator_snapshot: Dict[str, float], final_result: str):
        self._buffer.append({"snapshot": indicator_snapshot, "result": final_result})
        if len(self._buffer) >= self.update_interval:
            self._update_weights()
            self._buffer.clear()

    def _update_weights(self):
        for record in self._buffer:
            result = record["result"]
            direction_factor = 1.0 if result == "WIN" else -1.0
            for key, contribution in record["snapshot"].items():
                if key not in self.weights:
                    continue
                contrib_sign = 1.0 if contribution >= 0 else -1.0
                self.weights[key] += self.learning_rate * direction_factor * contrib_sign
                self.weights[key] = min(3.0, max(0.2, self.weights[key]))
        self.weights = self.normalized_weights()
        logger.info("Updated adaptive weights %s_%s: %s", self.symbol, self.timeframe, self.weights)

    def write_log(self, payload: Dict):
        entry = {"time": datetime.now(timezone.utc).isoformat(), **payload}
        try:
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=True) + "\n")
        except Exception as exc:
            logger.error("Failed writing training log: %s", exc)

    def write_trade_log(self, payload: Dict):
        entry = {"time": datetime.now(timezone.utc).isoformat(), **payload}
        try:
            with self.trade_log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=True) + "\n")
        except Exception as exc:
            logger.error("Failed writing trade log: %s", exc)
