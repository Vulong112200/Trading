import requests
import pandas as pd
import asyncio
import json
import websockets


class DataLayer:
    def __init__(self, symbol="BTCUSDT", interval="1m"):
        self.symbol = symbol.upper()
        self.interval = interval
        self.base_url = "https://api4.binance.com/api/v3/klines"
        self.ws_url = f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@kline_{self.interval}"
        self.df_full = pd.DataFrame()

    def load_history(self, limit=1000):
        """Kh?i t?o d? li?u l?ch s? (Ch?y 1 l?n l?c b?t server)"""
        print(f"[DATA] ?ang t?i {limit} n?n l?ch s? cho {self.symbol} qua REST API...")
        params = {"symbol": self.symbol, "interval": self.interval, "limit": limit}
        try:
            res = requests.get(self.base_url, params=params, timeout=10)
            res.raise_for_status()
            data = res.json()

            columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades',
                       'taker_base', 'taker_quote', 'ignore']
            df = pd.DataFrame(data, columns=columns)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(
                float)

            self.df_full = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            print(f"[DATA] T?i th?nh c?ng {len(self.df_full)} n?n. S?n s?ng t?nh to?n!")
        except Exception as e:
            print(f"[ERROR] L?i t?i d? li?u l?ch s?: {e}")

    async def start_websocket(self):
        """K?t n?i WebSocket ?? update n?n real-time"""
        print(f"[WEBSOCKET] ?ang k?t n?i lu?ng gi? Real-time cho {self.symbol}...")
        async for websocket in websockets.connect(self.ws_url):
            try:
                print(f"[WEBSOCKET] K?t n?i th?nh c?ng! D? li?u ?ang ch?y (Ping m?i gi?y).")
                while True:
                    msg = await websocket.recv()
                    data = json.loads(msg)
                    kline = data['k']

                    tick_time = pd.to_datetime(kline['t'], unit='ms')
                    tick_data = {
                        'timestamp': tick_time,
                        'open': float(kline['o']),
                        'high': float(kline['h']),
                        'low': float(kline['l']),
                        'close': float(kline['c']),
                        'volume': float(kline['v'])
                    }

                    if not self.df_full.empty:
                        last_time = self.df_full.iloc[-1]['timestamp']
                        if tick_time == last_time:
                            # Update n?n hi?n t?i
                            for col in ['high', 'low', 'close', 'volume']:
                                self.df_full.at[self.df_full.index[-1], col] = tick_data[col]
                        else:
                            # Th?m n?n m?i
                            self.df_full = pd.concat([self.df_full, pd.DataFrame([tick_data])], ignore_index=True)
                            if len(self.df_full) > 2000:
                                self.df_full = self.df_full.iloc[-2000:].reset_index(drop=True)

            except websockets.ConnectionClosed:
                print("[WEBSOCKET] M?t k?t n?i, ?ang th? k?t n?i l?i...")
                await asyncio.sleep(2)
            except Exception as e:
                print(f"[WEBSOCKET] Error: {e}")
                await asyncio.sleep(2)

    def get_current_data(self):
        return self.df_full.copy()