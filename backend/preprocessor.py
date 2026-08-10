"""
Preprocessor Module
Computes technical indicators, creates lag features, and constructs sliding window datasets.
"""

from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class StockPreprocessor:
    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators on historical OHLCV data:
        SMA (20, 50), EMA (20), RSI (14), MACD, Volatility, and price lags.
        """
        df_feat = df.copy()

        # Moving Averages
        df_feat["SMA_20"] = df_feat["Close"].rolling(window=20, min_periods=1).mean()
        df_feat["SMA_50"] = df_feat["Close"].rolling(window=50, min_periods=1).mean()
        df_feat["EMA_20"] = df_feat["Close"].ewm(span=20, adjust=False).mean()

        # Returns and Volatility
        df_feat["Daily_Return"] = df_feat["Close"].pct_change().fillna(0.0)
        df_feat["Volatility_20"] = (
            df_feat["Daily_Return"].rolling(window=20, min_periods=1).std().fillna(0.0)
        )

        # RSI (14)
        delta = df_feat["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        df_feat["RSI_14"] = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

        # MACD
        ema_12 = df_feat["Close"].ewm(span=12, adjust=False).mean()
        ema_26 = df_feat["Close"].ewm(span=26, adjust=False).mean()
        df_feat["MACD"] = ema_12 - ema_26
        df_feat["MACD_Signal"] = df_feat["MACD"].ewm(span=9, adjust=False).mean()

        # Price spreads
        df_feat["High_Low_Pct"] = ((df_feat["High"] - df_feat["Low"]) / (df_feat["Close"] + 1e-9)) * 100.0
        df_feat["Close_Open_Pct"] = ((df_feat["Close"] - df_feat["Open"]) / (df_feat["Open"] + 1e-9)) * 100.0

        # Lags
        df_feat["Close_Lag_1"] = df_feat["Close"].shift(1)
        df_feat["Close_Lag_2"] = df_feat["Close"].shift(2)
        df_feat["Close_Lag_3"] = df_feat["Close"].shift(3)
        df_feat["Close_Lag_5"] = df_feat["Close"].shift(5)

        df_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_feat.bfill(inplace=True)
        df_feat.ffill(inplace=True)

        return df_feat

    def prepare_lstm_data(
        self, df: pd.DataFrame, train_split: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
        """
        Prepare 3D sliding window sequences for LSTM training and testing.
        """
        close_prices = df["Close"].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(close_prices)

        X, y = [], []
        window = min(self.window_size, max(5, len(scaled_data) // 4))

        for i in range(window, len(scaled_data)):
            X.append(scaled_data[i - window : i, 0])
            y.append(scaled_data[i, 0])

        X = np.array(X)
        y = np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        split_idx = int(len(X) * train_split)
        if split_idx == 0 or split_idx >= len(X):
            split_idx = max(1, len(X) - 10)

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        return X_train, y_train, X_test, y_test, self.scaler

    def prepare_tabular_data(
        self, df_with_features: pd.DataFrame, train_split: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare tabular feature matrix and next-day target for Linear Regression and Random Forest.
        """
        df = df_with_features.copy()
        df["Target_Next_Close"] = df["Close"].shift(-1)
        df_clean = df.dropna().copy()

        feature_cols = [
            "Close", "Open", "High", "Low", "Volume",
            "SMA_20", "SMA_50", "EMA_20", "RSI_14", "MACD", "MACD_Signal",
            "Daily_Return", "Volatility_20", "High_Low_Pct", "Close_Open_Pct",
            "Close_Lag_1", "Close_Lag_2", "Close_Lag_3", "Close_Lag_5",
        ]

        available_features = [c for c in feature_cols if c in df_clean.columns]
        X = df_clean[available_features]
        y = df_clean["Target_Next_Close"]

        split_idx = int(len(X) * train_split)
        if split_idx == 0 or split_idx >= len(X):
            split_idx = max(1, len(X) - 10)

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        return X_train, y_train, X_test, y_test, available_features
