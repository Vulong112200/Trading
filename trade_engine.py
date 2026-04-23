# trade_engine.py
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


logger = logging.getLogger("TradeEngine")


@dataclass
class Trade:
    entry_time: str
    entry_price: float
    position_type: str
    size: float
    leverage: float
    tp_price: float
    sl_price: float
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    result: Optional[str] = None
    status: str = "OPEN"


class TradeEngine:
    def __init__(
        self,
        initial_capital: float = 100.0,
        risk_per_trade: float = 0.01,
        tp_atr_mult: float = 1.8,
        sl_atr_mult: float = 1.2,
        entry_buffer_atr_mult: float = 0.1,
    ):
        self.capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.tp_atr_mult = tp_atr_mult
        self.sl_atr_mult = sl_atr_mult
        self.entry_buffer_atr_mult = entry_buffer_atr_mult

        self.current_trade: Optional[Trade] = None
        self.pending_order: Optional[Dict] = None
        self.trade_history: List[Dict] = []

    def set_capital(self, value: float):
        self.capital = max(1.0, float(value))

    def _build_order(self, timestamp: str, signal: str, close_price: float, atr: float) -> Dict:
        safe_atr = max(atr, close_price * 0.0005)
        entry_buffer = safe_atr * self.entry_buffer_atr_mult

        if signal == "BUY":
            entry = close_price + entry_buffer
            sl = entry - (safe_atr * self.sl_atr_mult)
            tp = entry + (safe_atr * self.tp_atr_mult)
            position_type = "LONG"
        else:
            entry = close_price - entry_buffer
            sl = entry + (safe_atr * self.sl_atr_mult)
            tp = entry - (safe_atr * self.tp_atr_mult)
            position_type = "SHORT"

        risk_usd = self.capital * self.risk_per_trade
        risk_per_unit = max(abs(entry - sl), 1e-9)
        size = max(risk_usd / risk_per_unit, 0.0)

        return {
            "requested_time": timestamp,
            "position_type": position_type,
            "entry_price": round(entry, 4),
            "tp_price": round(tp, 4),
            "sl_price": round(sl, 4),
            "size": round(size, 6),
            "leverage": 1.0,
        }

    def request_entry(self, timestamp: str, signal: str, close_price: float, atr: float):
        if self.current_trade or self.pending_order or signal not in {"BUY", "SELL"}:
            return
        self.pending_order = self._build_order(timestamp, signal, close_price, atr)

    def _touches_price(self, low: float, high: float, target_price: float) -> bool:
        return low <= target_price <= high

    def _open_from_pending(self, timestamp: str):
        order = self.pending_order
        if not order:
            return
        self.current_trade = Trade(
            entry_time=timestamp,
            entry_price=order["entry_price"],
            position_type=order["position_type"],
            size=order["size"],
            leverage=order["leverage"],
            tp_price=order["tp_price"],
            sl_price=order["sl_price"],
        )
        self.pending_order = None

    def _close_trade(self, timestamp: str, exit_price: float, result: str) -> Dict:
        trade = self.current_trade
        if not trade:
            return {"status": "NONE"}

        if trade.position_type == "LONG":
            pnl = (exit_price - trade.entry_price) * trade.size
        else:
            pnl = (trade.entry_price - exit_price) * trade.size

        trade.exit_time = timestamp
        trade.exit_price = round(exit_price, 4)
        trade.pnl = round(pnl, 4)
        trade.result = result
        trade.status = "CLOSED"

        self.capital += pnl
        closed = asdict(trade)
        self.trade_history.append(closed)
        self.current_trade = None
        return closed

    def on_candle(self, timestamp: str, high: float, low: float, close: float) -> Dict:
        try:
            if self.pending_order and self._touches_price(low, high, self.pending_order["entry_price"]):
                self._open_from_pending(timestamp)

            if not self.current_trade:
                return {
                    "position_status": "PENDING" if self.pending_order else "NONE",
                    "pending_order": self.pending_order,
                    "capital": round(self.capital, 4),
                }

            trade = self.current_trade
            if trade.position_type == "LONG":
                sl_hit = low <= trade.sl_price
                tp_hit = high >= trade.tp_price
                if sl_hit and tp_hit:
                    return self._close_trade(timestamp, trade.sl_price, "LOSS")
                if sl_hit:
                    return self._close_trade(timestamp, trade.sl_price, "LOSS")
                if tp_hit:
                    return self._close_trade(timestamp, trade.tp_price, "WIN")
            else:
                sl_hit = high >= trade.sl_price
                tp_hit = low <= trade.tp_price
                if sl_hit and tp_hit:
                    return self._close_trade(timestamp, trade.sl_price, "LOSS")
                if sl_hit:
                    return self._close_trade(timestamp, trade.sl_price, "LOSS")
                if tp_hit:
                    return self._close_trade(timestamp, trade.tp_price, "WIN")

            return {
                "position_status": "OPEN",
                "entry_time": trade.entry_time,
                "entry_price": trade.entry_price,
                "position_type": trade.position_type,
                "size": trade.size,
                "leverage": trade.leverage,
                "tp_price": trade.tp_price,
                "sl_price": trade.sl_price,
                "unrealized_pnl": round(
                    (close - trade.entry_price) * trade.size
                    if trade.position_type == "LONG"
                    else (trade.entry_price - close) * trade.size,
                    4,
                ),
                "capital": round(self.capital, 4),
            }
        except Exception as exc:
            logger.error("Trade engine error: %s", exc)
            return {"position_status": "ERROR", "error": str(exc), "capital": round(self.capital, 4)}
