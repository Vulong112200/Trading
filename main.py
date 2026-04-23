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
async def on_market_tick(symbol: str, interval: str, is_closed: bool, tick_data: dict):
    """
    tick_data = {"time": int, "open": float, "high": float, "low": float, "close": float}
    """
    engine_key = f"{symbol}_{interval}"
    engine = engines.get(engine_key)
    if not engine: return

    # 1. CONTINUOUS EXECUTION: Qu?t r?u n?n m?i tick ?? check TP/SL ngay l?p t?c
    trade_status = engine.trade_sim.process_tick(
        current_price=tick_data['close'],
        high=tick_data['high'],
        low=tick_data['low']
    )

    # 2. EVENT-DRIVEN PREDICTION: CH? ch?y khi N?N ??NG C?A (1m, 5m, 1h...)
    if not is_closed:
        # N?u n?n ch?a ??ng, ch? g?i tr?ng th?i trade/gi? hi?n t?i xu?ng UI ?? c?p nh?t chart
        await connection_manager.broadcast_to_symbol(json.dumps({
            "type": "TICK_UPDATE",
            "candle": tick_data,
            "trade": trade_status
        }), symbol, interval)
        return

    # === T? ??Y TR? XU?NG CH? CH?Y 1 L?N/N?N ===

    # T?nh Indicator
    df_raw = data_manager.cache.get(symbol, {}).get(interval)
    df_ind = indicator_layer.apply_indicators(df_raw)

    # L?y Multi-Timeframe Context
    mtf_context = {}
    for tf in ["1m", "5m", "15m", "1h", "4h"]:  # L?y ?? c?c TF
        df_tf = data_manager.cache.get(symbol, {}).get(tf)
        if df_tf is not None and len(df_tf) > 0:
            mtf_context[tf] = indicator_layer.apply_indicators(df_tf)

    # Sinh T?n Hi?u & Ghi Log
    analysis = engine.generate_signal(df_ind, mtf_context)

    # Broadcast xu?ng UI
    await connection_manager.broadcast_to_symbol(json.dumps({
        "type": "CANDLE_CLOSE",
        "symbol": symbol,
        "candle": tick_data,
        "signal": analysis
    }), symbol, interval)


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
                capital = req.get("capital")

                # C?p nh?t l?a ch?n c?a Client
                connection_manager.update_subscription(websocket, symbol, interval)

                engine_key = f"{symbol}_{interval}"
                engine = engines.get(engine_key)
                if engine and capital is not None:
                    try:
                        engine.set_capital(float(capital))
                    except Exception:
                        logger.warning(f"Capital value invalid: {capital}")

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