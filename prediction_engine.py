# predict_engine.py
import logging
from typing import Dict


logger = logging.getLogger("PredictionEngine")


class PredictionEngine:
    def __init__(
        self,
        atr_multiplier: float = 1.2,
        smooth_period: int = 5,
        max_width_pct: float = 0.02,
    ):
        self.atr_multiplier = atr_multiplier
        self.max_width_pct = max_width_pct
        self.alpha = 2.0 / (smooth_period + 1.0)
        self._smoothed_mid = None
        self._smoothed_width = None

    @staticmethod
    def _ema_step(previous: float, value: float, alpha: float) -> float:
        if previous is None:
            return value
        return (alpha * value) + ((1.0 - alpha) * previous)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def predict(
        self,
        close_price: float,
        ema9: float,
        atr: float,
        bb_upper: float,
        bb_lower: float,
        momentum_score: float,
        direction: str,
    ) -> Dict:
        try:
            safe_close = max(close_price, 1e-9)
            safe_atr = max(atr, safe_close * 0.0005)
            safe_ema9 = ema9 if ema9 > 0 else close_price

            bb_width = max(0.0, bb_upper - bb_lower)
            base_width = safe_atr * self.atr_multiplier
            bb_adjusted = max(base_width, bb_width * 0.35)

            momentum_factor = 1.0 + (abs(momentum_score) * 0.25)
            raw_width = bb_adjusted * momentum_factor
            width_cap = safe_close * self.max_width_pct
            raw_width = self._clamp(raw_width, safe_atr * 0.7, max(width_cap, safe_atr * 1.2))

            self._smoothed_mid = self._ema_step(self._smoothed_mid, safe_ema9, self.alpha)
            self._smoothed_width = self._ema_step(self._smoothed_width, raw_width, self.alpha)

            mid_price = self._smoothed_mid
            range_width = max(self._smoothed_width, safe_atr * 0.5)

            if direction == "BULL":
                r_min = mid_price
                r_max = mid_price + range_width
            elif direction == "BEAR":
                r_min = mid_price - range_width
                r_max = mid_price
            else:
                r_min = mid_price - (range_width * 0.5)
                r_max = mid_price + (range_width * 0.5)

            confidence = self._clamp(1.0 - (safe_atr / safe_close) * 8.0, 0.05, 0.95)

            return {
                "direction": direction,
                "range": {"min": round(r_min, 4), "max": round(r_max, 4)},
                "mid_price": round(mid_price, 4),
                "range_width": round(range_width, 4),
                "confidence": round(confidence, 4),
            }
        except Exception as exc:
            logger.error("Prediction failed: %s", exc)
            return {
                "direction": "SIDEWAY",
                "range": {"min": round(close_price, 4), "max": round(close_price, 4)},
                "mid_price": round(close_price, 4),
                "range_width": 0.0,
                "confidence": 0.05,
            }
