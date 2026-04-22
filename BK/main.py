from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import asyncio
import uvicorn

from data_layer import DataLayer
from indicator_layer import IndicatorLayer
from signal_engine import SignalEngine

# Init layers
data_layer = DataLayer(symbol="BTCUSDT", interval="1m")
indicator_layer = IndicatorLayer()
signal_engine = SignalEngine()

# --- LIFESPAN: Qu?n l? v?ng ??i c?a App ---
# Ch?y tr??c khi server nh?n request v? d?n d?p khi t?t server
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[SYSTEM] Kh?i ??ng h? th?ng Quant Trading...")
    # 1. Bootstrapping: T?i 1000 n?n l?ch s?
    data_layer.load_history(limit=1000)
    # 2. B?t lu?ng ch?y ng?m ?? h?ng WebSocket Real-time
    asyncio.create_task(data_layer.start_websocket())
    yield
    print("[SYSTEM] T?t h? th?ng.")

app = FastAPI(lifespan=lifespan)

# Serve static files (Frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse('static/index.html')

@app.get("/api/market-data")
async def get_market_data():
    # 1. Fetch data real-time t? b? nh? (R?t nhanh, ko b? delay m?ng)
    df = data_layer.get_current_data()

    # KHI L?I HO?C CH?A K?P LOAD DATA
    if df.empty:
        return {"status": "error", "message": "?ang ??ng b? d? li?u t? Binance..."}

    # 2. Add Indicators
    df_ind = indicator_layer.apply_indicators(df)

    # 3. Generate Signal
    signal = signal_engine.generate_signal(df_ind)

    # 4. Format Chart Data
    # Tr?ch xu?t 200 n?n cu?i ?? giao di?n load kh?ng b? lag
    df_chart = df_ind.tail(500).copy()
    chart_data = df_chart[['timestamp', 'open', 'high', 'low', 'close', 'EMA_9', 'EMA_21', 'BB_upper', 'BB_lower']].copy()

    # [?? S?A] D?ng h?m int(x.timestamp()) ?? ??m b?o lu?n l?y ??ng UNIX seconds
    chart_data['time'] = chart_data['timestamp'].apply(lambda x: int(x.timestamp()))

    return {
        "status": "success",
        "chart_data": chart_data.drop('timestamp', axis=1).to_dict(orient="records"),
        "analysis": signal

    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)