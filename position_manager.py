# position_manager.py
class PositionManager:
    def __init__(self):
        self.state = "NONE"  # NONE, LONG, SHORT
        self.entry_price = 0.0
        self.bars_held = 0
        self.cooldown_counter = 0

    def update_state(self, current_price: float):
        """C?p nh?t th?i gian gi? l?nh v? PnL"""
        unrealized_pnl = 0.0
        if self.state != "NONE":
            self.bars_held += 1
            if self.state == "LONG":
                unrealized_pnl = (current_price - self.entry_price) / self.entry_price
            elif self.state == "SHORT":
                unrealized_pnl = (self.entry_price - current_price) / self.entry_price

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

        return unrealized_pnl

    def open_position(self, side: str, price: float):
        self.state = side
        self.entry_price = price
        self.bars_held = 0

    def close_position(self):
        self.state = "NONE"
        self.entry_price = 0.0
        self.bars_held = 0
        self.cooldown_counter = 5  # Cooldown 5 n?n kh?ng v?o l?nh