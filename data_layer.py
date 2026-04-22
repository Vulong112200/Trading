# data_layer.py
import pandas as pd
import requests
import websockets
import json
import asyncio
import logging
from config import SYMBOLS, TIMEFRAMES, HISTORY_LIMIT, MAX_CACHE_SIZE

logger = logging.getLogger("DataLayer")


class DataLayer:
    def __init__(self):
        self.cache = {}
        self.on_tick_callback = None  # [S?A L?I] Kh?i t?o c?u n?i v?i main.py

    def bootstrap(self, symbol: str, interval: str):
        if symbol not in self.cache: self.cache[symbol] = {}
        logger.info(f"Bootstrapping {symbol} {interval}...")
        try:
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={HISTORY_LIMIT}"
            res = requests.get(url, timeout=10).json()
            df = pd.DataFrame(res,
                              columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'ct', 'qav', 'nt', 'tb',
                                       'tq', 'i'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            for col in ['open', 'high', 'low', 'close', 'volume']: df[col] = df[col].astype(float)
            self.cache[symbol][interval] = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        except Exception as e:
            logger.error(f"L?i REST API {symbol}: {e}")

    async def ws_loop(self):
        streams = [f"{s.lower()}@kline_{t}" for s in SYMBOLS for t in TIMEFRAMES]
        ws_url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"
        while True:
            try:
                async with websockets.connect(ws_url) as ws:
                    logger.info("Binance WS Connected!")
                    while True:
                        msg = json.loads(await ws.recv())['data']
                        await self._update_cache(msg)  # [S?A L?I] Ph?i c? await ? ??y
            except Exception as e:
                logger.error(f"WS Disconnected: {e}. Reconnecting...")
                await asyncio.sleep(5)

    async def _update_cache(self, msg):  # [S?A L?I] Th?m async def
        k, sym, interval = msg['k'], msg['s'], msg['k']['i']
        if sym not in self.cache or interval not in self.cache[sym]: return

        df = self.cache[sym][interval]
        tick_time = pd.to_datetime(k['t'], unit='ms', utc=True)
        last_idx = df.index[-1]

        if df.at[last_idx, 'timestamp'] == tick_time:
            df.at[last_idx, 'high'] = float(k['h'])
            df.at[last_idx, 'low'] = float(k['l'])
            df.at[last_idx, 'close'] = float(k['c'])
            df.at[last_idx, 'volume'] = float(k['v'])
        else:
            new_row = {'timestamp': tick_time, 'open': float(k['o']), 'high': float(k['h']), 'low': float(k['l']),
                       'close': float(k['c']), 'volume': float(k['v'])}
            self.cache[sym][interval] = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True).tail(MAX_CACHE_SIZE)

        # [S?A L?I QUAN TR?NG NH?T]: B?o cho main.py bi?t ?? ??y data xu?ng Frontend
        if self.on_tick_callback:
            await self.on_tick_callback(sym, interval, k['x'])


data_manager = DataLayer()