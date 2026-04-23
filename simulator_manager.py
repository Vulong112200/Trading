# -*- coding: utf-8 -*-
from datetime import datetime


class ManualTradeSimulator:
    def __init__(self):
        self.active_trades = []
        self.history = []

    def open_trade(self, trade_data):
        trade_data['status'] = 'OPEN'
        trade_data['entry_time'] = datetime.now().isoformat()
        self.active_trades.append(trade_data)

    def update_tick(self, current_price):
        for trade in self.active_trades[:]:
            hit_tp = (trade['position'] == 'LONG' and current_price >= trade['tp']) or \
                     (trade['position'] == 'SHORT' and current_price <= trade['tp'])

            hit_sl = (trade['position'] == 'LONG' and current_price <= trade['sl']) or \
                     (trade['position'] == 'SHORT' and current_price >= trade['sl'])

            if hit_tp or hit_sl:
                trade['status'] = 'CLOSED'
                trade['exit_price'] = current_price
                trade['exit_time'] = datetime.now().isoformat()
                trade['result'] = 'WIN' if hit_tp else 'LOSS'

                # Tính PnL (%)
                mult = 1 if trade['position'] == 'LONG' else -1
                trade['pnl'] = round(((current_price - trade['entry']) / trade['entry']) * 100 * mult, 2)

                # Chèn vào đầu lịch sử, giữ tối đa 50 lệnh gần nhất
                self.history.insert(0, trade)
                self.history = self.history[:50]

                self.active_trades.remove(trade)
                return trade
        return None