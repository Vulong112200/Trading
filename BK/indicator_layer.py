import pandas as pd
import pandas_ta as ta
import logging

# Thi?t l?p logger
logger = logging.getLogger(__name__)


class IndicatorLayer:

    @staticmethod
    def safe_assign(df: pd.DataFrame, col_name: str, data):
        """Helper function: G?n d? li?u an to?n, h? tr? c? Pandas Series v? Numpy Array."""
        if data is None:
            return

        # X? l? n?u data l? Pandas Series ho?c DataFrame
        if isinstance(data, (pd.Series, pd.DataFrame)):
            if not data.empty:
                # ?p v? .values ?? tr?nh l?i l?ch Index (g?y ra gi? tr? NaN)
                df[col_name] = data.values

        # X? l? n?u data l? Numpy Array ho?c List
        elif hasattr(data, '__len__'):
            if len(data) > 0:
                df[col_name] = data
        else:
            logger.warning(f"Kh?ng th? g?n {col_name}: ??nh d?ng d? li?u kh?ng h?p l?.")

    @staticmethod
    def apply_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        T?nh to?n Indicators an to?n. Ch?ng crash to?n b? API.
        S? d?ng iloc ?? tr?nh KeyError t? vi?c pandas-ta ??i t?n c?t.
        """
        # 1. KI?M TRA DATA ??U V?O
        if df is None or df.empty:
            logger.warning("DataFrame r?ng. B? qua t?nh to?n Indicator.")
            return df

        # 2. CHU?N HO? KI?U D? LI?U
        # ??m b?o c?c c?t t?nh to?n ph?i l? float ?? tr?nh l?i Math c?a numpy
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(float)

        # 3. KH?I T?O SCHEMA M?C ??NH (B?O V? API TR? V?)
        # ??m b?o API /api/market-data lu?n c? c?c key n?y, d? indicator c? l?i
        required_columns = [
            'EMA_9', 'EMA_21', 'EMA_50', 'SMA_50', 'SMA_200',
            'MACD', 'MACD_signal', 'MACD_hist',
            'RSI_14', 'STOCH_k', 'STOCH_d',
            'BB_lower', 'BB_middle', 'BB_upper',
            'VOL_MA', 'OBV'
        ]
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0.0

        # 4. T?NH TO?N T?NG BLOCK INDICATOR (C? L?P L?I)

        # --- T?n hi?u xu h??ng (Trend) ---
        try:
            df['EMA_9'] = ta.ema(df['close'], length=9)
            df['EMA_21'] = ta.ema(df['close'], length=21)
            df['EMA_50'] = ta.ema(df['close'], length=50)
            df['SMA_50'] = ta.sma(df['close'], length=50)
            df['SMA_200'] = ta.sma(df['close'], length=200)
        except Exception as e:
            logger.warning(f"[Indicator] L?i t?nh to?n EMA/SMA: {e}")

        # --- MACD ---
        try:
            macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
            if macd is not None and not macd.empty:
                # D?ng iloc: 0=MACD, 1=Histogram, 2=Signal
                IndicatorLayer.safe_assign(df, 'MACD', macd.iloc[:, 0])
                IndicatorLayer.safe_assign(df, 'MACD_hist', macd.iloc[:, 1])
                IndicatorLayer.safe_assign(df, 'MACD_signal', macd.iloc[:, 2])
        except Exception as e:
            logger.warning(f"[Indicator] L?i t?nh to?n MACD: {e}")

        # --- RSI ---
        try:
            df['RSI_14'] = ta.rsi(df['close'], length=14)
        except Exception as e:
            logger.warning(f"[Indicator] L?i t?nh to?n RSI: {e}")

        # --- Stochastic ---
        try:
            stoch = ta.stoch(df['high'], df['low'], df['close'])
            if stoch is not None and not stoch.empty:
                # D?ng iloc: 0=Stoch %K, 1=Stoch %D
                IndicatorLayer.safe_assign(df, 'STOCH_k', stoch.iloc[:, 0])
                IndicatorLayer.safe_assign(df, 'STOCH_d', stoch.iloc[:, 1])
        except Exception as e:
            logger.warning(f"[Indicator] L?i t?nh to?n STOCH: {e}")

        # --- Bollinger Bands (Ngu?n g?c l?i KeyError) ---
        try:
            bbands = ta.bbands(df['close'], length=20, std=2)
            if bbands is not None and not bbands.empty:
                # D?ng iloc: 0=Lower, 1=Middle (SMA), 2=Upper
                IndicatorLayer.safe_assign(df, 'BB_lower', bbands.iloc[:, 0])
                IndicatorLayer.safe_assign(df, 'BB_middle', bbands.iloc[:, 1])
                IndicatorLayer.safe_assign(df, 'BB_upper', bbands.iloc[:, 2])
        except Exception as e:
            logger.warning(f"[Indicator] L?i t?nh to?n Bollinger Bands: {e}")

        # --- Volume ---
        try:
            df['VOL_MA'] = ta.sma(df['volume'], length=20)
            df['OBV'] = ta.obv(df['close'], df['volume'])
        except Exception as e:
            logger.warning(f"[Indicator] L?i t?nh to?n Volume (MA/OBV): {e}")

        # --- ATR (Average True Range) - ?o ?? bi?n ??ng ---
        try:
            df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        except Exception as e:
            logger.warning(f"[Indicator] L?i t?nh ATR: {e}")

        # --- VWAP (Volume Weighted Average Price) ---
        try:
            temp_df = df.copy()

            # ??m b?o timestamp l? datetime
            if not pd.api.types.is_datetime64_any_dtype(temp_df['timestamp']):
                temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'], unit='ms')

            temp_df.set_index('timestamp', inplace=True)

            vwap_result = ta.vwap(
                temp_df['high'],
                temp_df['low'],
                temp_df['close'],
                temp_df['volume']
            )

            # ? Normalize v? pandas Series
            if vwap_result is not None:
                if isinstance(vwap_result, pd.DataFrame):
                    vwap_series = vwap_result.iloc[:, 0]

                elif isinstance(vwap_result, pd.Series):
                    vwap_series = vwap_result

                else:
                    # numpy array Å® convert sang Series
                    vwap_series = pd.Series(vwap_result, index=temp_df.index)

                # ? G?n an to?n
                if not vwap_series.empty:
                    IndicatorLayer.safe_assign(df, 'VWAP', vwap_series.values)

        except Exception as e:
            logger.warning(f"L?i t?nh VWAP: {e}")

        # --- ADX (Average Directional Index) - S?c m?nh xu h??ng ---
        try:
            adx = ta.adx(df['high'], df['low'], df['close'], length=14)
            if adx is not None and not adx.empty:
                IndicatorLayer.safe_assign(df, 'ADX_14', adx.iloc[:, 0])
        except Exception as e:
            logger.warning(f"[Indicator] L?i t?nh ADX: {e}")

        # 5. X? L? D? LI?U R?NG (NaN/Null Handling)
        # Bfill (Backfill): L?y gi? tr? ph?a sau l?p l?n tr??c (R?t h?u ?ch ?? fill nh?ng n?n ??u ti?n b? thi?u c?a SMA200)
        # Ffill (Forward fill): L?y gi? tr? tr??c l?p ra sau (Tr??ng h?p n?n hi?n t?i indicator fail)
        # Fillna(0.0): N?u ho?n to?n r?ng, cho b?ng 0 ?? an to?n parse sang JSON
        df = df.bfill().ffill().fillna(0.0)

        return df