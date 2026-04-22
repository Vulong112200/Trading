import asyncio
import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

# Import c?c module b?n ?? t?o
from data_layer import data_manager
from indicator_layer import IndicatorLayer
from signal_engine import AdvancedSignalEngine
from config import SYMBOLS, TIMEFRAMES

# Thi?t l?p log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MainApp")

# Kh?i t?o c?c c?ng c? x? l?
indicator_layer = IndicatorLayer()

# T?o t? ?i?n qu?n l? Engine cho t?ng c?p ti?n v? timeframe
# C?u tr?c: engines["BTCUSDT_1m"] = AdvancedSignalEngine()
engines = {
    f"{sym}_{tf}": AdvancedSignalEngine(sym, tf) # Truy?n th?m sym v? tf v?o ??y
    for sym in SYMBOLS
    for tf in TIMEFRAMES
}


# --- CALLBACK X? L? D? LI?U T? WEBSOCKET BINANCE ---
async def on_market_tick(symbol: str, interval: str, is_closed: bool):
    """
    H?m n?y ???c g?i m?i khi Binance c? tick gi? m?i ho?c ??ng n?n.
    N? th?c hi?n: T?nh to?n -> Sinh t?n hi?u -> G?i xu?ng Frontend.
    """
    # 1. L?y d? li?u th? t? RAM Cache
    df_raw = data_manager.cache.get(symbol, {}).get(interval)
    if df_raw is None or df_raw.empty:
        return

    # 2. T?nh to?n ch? b?o (Indicator)
    df_ind = indicator_layer.apply_indicators(df_raw)

    # 3. L?y Engine t??ng ?ng ?? sinh t?n hi?u (Signal)
    engine_key = f"{symbol}_{interval}"
    engine = engines.get(engine_key)
    if not engine: return

    analysis = engine.generate_signal(df_ind)

    # 4. Chu?n b? d? li?u tick cho Chart
    last_row = df_ind.iloc[-1]
    # Quan tr?ng: Chuy?n timestamp sang gi?y (UNIX seconds)
    tick_time = int(last_row['timestamp'].timestamp())

    tick_payload = {
        "time": tick_time,
        "open": float(last_row['open']),
        "high": float(last_row['high']),
        "low": float(last_row['low']),
        "close": float(last_row['close']),
        "EMA_9": float(last_row.get('EMA_9', 0)),
        "EMA_21": float(last_row.get('EMA_21', 0))
    }

    # 5. ??ng g?i d? li?u g?i xu?ng c?c Client ?ang k?t n?i
    message = json.dumps({
        "type": "TICK",
        "symbol": symbol,
        "candle": tick_payload,
        "signal": analysis
    })

    # Broadcast t?i c?c tr?nh duy?t ?ang xem ??ng c?p ti?n n?y
    await connection_manager.broadcast_to_symbol(message, symbol, interval)


# G?n callback v?o data_manager
data_manager.on_tick_callback = on_market_tick


# --- QU?N L? K?T N?I WEBSOCKET FRONTEND ---
class ConnectionManager:
    def __init__(self):
        # L?u tr?: {websocket: {"symbol": "...", "interval": "..."}}
        self.active_connections = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[websocket] = {"symbol": "BTCUSDT", "interval": "1m"}

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            del self.active_connections[websocket]

    def update_subscription(self, websocket: WebSocket, symbol: str, interval: str):
        if websocket in self.active_connections:
            self.active_connections[websocket]["symbol"] = symbol
            self.active_connections[websocket]["interval"] = interval

    async def broadcast_to_symbol(self, message: str, symbol: str, interval: str):
        for ws, pref in list(self.active_connections.items()):
            if pref["symbol"] == symbol and pref["interval"] == interval:
                try:
                    await ws.send_text(message)
                except Exception:
                    self.disconnect(ws)


connection_manager = ConnectionManager()


# --- V?NG ??I ?NG D?NG (LIFESPAN) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[SYSTEM] Kh?i ??ng h? th?ng Quant Trading...")
    # 1. T?i d? li?u l?ch s? cho t?t c? c?c m? trong config
    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            data_manager.bootstrap(sym, tf)

    # 2. B?t ??u lu?ng WebSocket Binance ch?y ng?m
    asyncio.create_task(data_manager.ws_loop())
    yield
    logger.info("[SYSTEM] ?ang t?t h? th?ng.")


# --- KH?I T?O FASTAPI ---
app = FastAPI(lifespan=lifespan)

# Mount th? m?c static ?? ph?c v? file HTML/JS
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    """Tr? v? giao di?n ch?nh"""
    return FileResponse('static/index.html')


@app.websocket("/ws/frontend")
async def websocket_endpoint(websocket: WebSocket):
    """
    C?ng k?t n?i d?nh cho Tr?nh duy?t.
    Cho ph?p ??i c?p ti?n v? nh?n d? li?u real-time.
    """
    await connection_manager.connect(websocket)
    try:
        while True:
            # L?ng nghe y?u c?u ??i Symbol t? Frontend
            data = await websocket.receive_text()
            req = json.loads(data)

            if req.get("action") == "subscribe":
                symbol = req['symbol']
                interval = req['interval']

                # C?p nh?t l?a ch?n c?a Client
                connection_manager.update_subscription(websocket, symbol, interval)

                # Ki?m tra v? n?p cache n?u c?n
                if symbol not in data_manager.cache:
                    data_manager.bootstrap(symbol, interval)

                # G?i 200 n?n l?ch s? ?? Frontend v? l?i Chart
                df_raw = data_manager.cache.get(symbol, {}).get(interval)
                if df_raw is not None and not df_raw.empty:
                    df_ind = indicator_layer.apply_indicators(df_raw).tail(200)

                    # Chu?n b? n?n l?ch s?
                    history_data = []
                    for _, row in df_ind.iterrows():
                        history_data.append({
                            "time": int(row['timestamp'].timestamp()),
                            "open": row['open'], "high": row['high'],
                            "low": row['low'], "close": row['close'],
                            "EMA_9": row.get('EMA_9', 0),
                            "EMA_21": row.get('EMA_21', 0)
                        })

                    # S?p x?p v? x?a tr?ng l?p tr??c khi g?i
                    history_data = sorted(history_data, key=lambda x: x['time'])

                    await websocket.send_text(json.dumps({
                        "type": "FULL_LOAD",
                        "symbol": symbol,
                        "data": history_data
                    }))

    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"L?i WebSocket Client: {e}")
        connection_manager.disconnect(websocket)


if __name__ == "__main__":
    # Ch?y server t?i port 8000
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)