# indicator_layer.py
import pandas as pd
import pandas_ta as ta
import logging

logger = logging.getLogger("IndicatorLayer")


class IndicatorLayer:
    @staticmethod
    def safe_assign(df: pd.DataFrame, col_name: str, data):
        """G?n an to?n, t? ??ng detect pandas object hay numpy array, kh?ng d?ng hardcode column"""
        if data is None: return
        if isinstance(data, (pd.Series, pd.DataFrame)):
            if not data.empty: df[col_name] = data.values
        elif hasattr(data, '__len__'):
            if len(data) > 0: df[col_name] = data

    @staticmethod
    def apply_indicators(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty: return df
        df = df.copy()

        # ?p ki?u an to?n
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns: df[col] = df[col].astype(float)

        # 1. Trend
        try:
            df['EMA_9'] = ta.ema(df['close'], length=9)
            df['EMA_21'] = ta.ema(df['close'], length=21)
            df['EMA_50'] = ta.ema(df['close'], length=50)
            df['SMA_50'] = ta.sma(df['close'], length=50)
            df['SMA_200'] = ta.sma(df['close'], length=200)
        except Exception as e:
            logger.warning(f"L?i Trend: {e}")

        # 2. MACD
        try:
            macd = ta.macd(df['close'])
            if macd is not None and not macd.empty:
                IndicatorLayer.safe_assign(df, 'MACD', macd.iloc[:, 0])
                IndicatorLayer.safe_assign(df, 'MACD_H', macd.iloc[:, 1])
                IndicatorLayer.safe_assign(df, 'MACD_S', macd.iloc[:, 2])
                df['MACD_Hist'] = df['MACD_H']
        except Exception as e:
            logger.warning(f"L?i MACD: {e}")

        # 3. Momentum
        try:
            df['RSI'] = ta.rsi(df['close'], length=14)
            df['RSI_14'] = df['RSI']
            stoch = ta.stoch(df['high'], df['low'], df['close'])
            if stoch is not None and not stoch.empty:
                IndicatorLayer.safe_assign(df, 'STOCH_K', stoch.iloc[:, 0])
                IndicatorLayer.safe_assign(df, 'STOCH_D', stoch.iloc[:, 1])
        except Exception as e:
            logger.warning(f"L?i Momentum: {e}")

        # 4. Volatility (BB & ATR)
        try:
            bb = ta.bbands(df['close'], length=20, std=2)
            if bb is not None and not bb.empty:
                IndicatorLayer.safe_assign(df, 'BB_L', bb.iloc[:, 0])
                IndicatorLayer.safe_assign(df, 'BB_U', bb.iloc[:, 2])
                df['BB_Lower'] = df['BB_L']
                df['BB_Upper'] = df['BB_U']
            df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            df['ATR_14'] = df['ATR']
        except Exception as e:
            logger.warning(f"L?i Volatility: {e}")

        # 5. Volume
        try:
            df['VOL_MA'] = ta.sma(df['volume'], length=20)
            df['OBV'] = ta.obv(df['close'], df['volume'])

            temp_df = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(temp_df['timestamp']):
                temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'], unit='ms', utc=True)
            temp_df.set_index('timestamp', inplace=True)
            vwap = ta.vwap(temp_df['high'], temp_df['low'], temp_df['close'], temp_df['volume'])
            IndicatorLayer.safe_assign(df, 'VWAP', vwap)
        except Exception as e:
            logger.warning(f"L?i Volume: {e}")

        # X? l? NaN tri?t ?? tr??c khi tr? v? JSON
        return df.bfill().ffill().fillna(0.0)