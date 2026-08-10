"""
Data Fetcher Module
Handles fetching historical stock data, multi-timeframe analytics, and company metadata using yfinance.
"""

from typing import Dict, Any, Tuple, List
import datetime
import pandas as pd
import numpy as np
import yfinance as yf


class StockDataFetcher:
    """
    Fetches and validates historical stock price data and metadata.
    """

    SUPPORTED_PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]
    SUPPORTED_INTERVALS = ["1h", "1d", "1wk", "1mo"]

    def __init__(self, cache_enabled: bool = True):
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, Tuple[datetime.datetime, Any]] = {}

    def _get_from_cache(self, key: str, max_age_seconds: int = 180):
        if not self.cache_enabled:
            return None
        if key in self._cache:
            ts, val = self._cache[key]
            if (datetime.datetime.now() - ts).total_seconds() < max_age_seconds:
                return val
        return None

    def _set_cache(self, key: str, val: Any):
        if self.cache_enabled:
            # Keep cache size small (< 20 items)
            if len(self._cache) > 20:
                self._cache.clear()
            self._cache[key] = (datetime.datetime.now(), val)

    def fetch_history(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch OHLCV data for a given ticker symbol and interval.
        """
        ticker = str(ticker).replace(" ", "").upper()
        if not ticker:
            raise ValueError("Ticker symbol cannot be empty.")

        if period not in self.SUPPORTED_PERIODS:
            period = "1y"
        if interval not in self.SUPPORTED_INTERVALS:
            interval = "1d"

        # yfinance restriction: 1h interval is only available for periods <= 730d (approx 2y)
        if interval == "1h" and period in ["2y", "5y", "max"]:
            period = "1mo"

        cache_key = f"hist_{ticker}_{period}_{interval}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached.copy()

        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval, auto_adjust=True)

            if df is None or df.empty:
                raise ValueError(f"No historical price data found for ticker '{ticker}'.")

            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing expected column '{col}' in downloaded data.")

            df = df[required_cols].copy()
            df.dropna(inplace=True)

            if isinstance(df.index, pd.DatetimeIndex):
                if df.index.tz is not None:
                    df.index = df.index.tz_convert(None)
                if interval == "1h":
                    df.index = pd.to_datetime(df.index.strftime("%Y-%m-%d %H:%M"))
                else:
                    df.index = pd.to_datetime(df.index.strftime("%Y-%m-%d"))

            df.sort_index(ascending=True, inplace=True)

            if len(df) < 5:
                raise ValueError(f"Insufficient historical data ({len(df)} bars) for ticker '{ticker}'.")

            self._set_cache(cache_key, df)
            return df

        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            raise ValueError(f"Failed to fetch data for ticker '{ticker}': {str(e)}")

    def fetch_multi_timeframe_stats(self, ticker: str) -> Dict[str, Any]:
        """
        Compute multi-timeframe analytics:
        - Bottom Level: Hour-to-Hour intraday movement & volatility (1h interval)
        - Mid Level: Month-to-Month returns for the past 12 months
        - High Level: Year-to-Year (YoY) annual returns
        """
        ticker = str(ticker).replace(" ", "").upper()
        cache_key = f"tf_{ticker}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        stock = yf.Ticker(ticker)

        # 1. Hourly (Bottom-Level)
        hourly_data = []
        hourly_volatility = 0.0
        hourly_trend = "NEUTRAL"
        try:
            df_hourly = stock.history(period="5d", interval="1h", auto_adjust=True)
            if not df_hourly.empty and len(df_hourly) >= 3:
                recent_hourly = df_hourly.tail(8).copy()
                for idx, row in recent_hourly.iterrows():
                    time_label = idx.strftime("%b %d, %H:%M") if isinstance(idx, pd.Timestamp) else str(idx)
                    hourly_data.append({
                        "timestamp": time_label,
                        "open": round(float(row["Open"]), 2),
                        "close": round(float(row["Close"]), 2),
                        "high": round(float(row["High"]), 2),
                        "low": round(float(row["Low"]), 2),
                        "volume": int(row["Volume"]),
                        "change_pct": round(float((row["Close"] - row["Open"]) / (row["Open"] + 1e-9) * 100.0), 2),
                    })
                hourly_returns = df_hourly["Close"].pct_change().dropna()
                hourly_volatility = round(float(hourly_returns.std() * 100.0), 2)

                if len(hourly_data) >= 2:
                    last_h = hourly_data[-1]["close"]
                    prev_h = hourly_data[-2]["close"]
                    if last_h > prev_h:
                        hourly_trend = "BULLISH"
                    elif last_h < prev_h:
                        hourly_trend = "BEARISH"
        except Exception:
            hourly_data = []

        # 2. Month-to-Month (Mid-Level)
        monthly_returns = []
        try:
            df_monthly = stock.history(period="2y", interval="1mo", auto_adjust=True)
            if not df_monthly.empty and len(df_monthly) >= 2:
                df_monthly["Monthly_Return"] = df_monthly["Close"].pct_change() * 100.0
                recent_months = df_monthly.tail(12)
                for idx, row in recent_months.iterrows():
                    month_label = idx.strftime("%b %Y") if isinstance(idx, pd.Timestamp) else str(idx)
                    ret_val = row["Monthly_Return"]
                    monthly_returns.append({
                        "period": month_label,
                        "close": round(float(row["Close"]), 2),
                        "return_pct": round(float(ret_val), 2) if not np.isnan(ret_val) else 0.0,
                    })
        except Exception:
            monthly_returns = []

        # 3. Year-to-Year (High-Level)
        yearly_returns = []
        try:
            df_daily = stock.history(period="5y", interval="1d", auto_adjust=True)
            if not df_daily.empty and len(df_daily) >= 50:
                df_daily["Year"] = df_daily.index.year
                years = sorted(df_daily["Year"].unique())
                for y in years:
                    df_year = df_daily[df_daily["Year"] == y]
                    if len(df_year) >= 5:
                        start_price = float(df_year["Open"].iloc[0])
                        end_price = float(df_year["Close"].iloc[-1])
                        yoy_pct = ((end_price - start_price) / (start_price + 1e-9)) * 100.0
                        yearly_returns.append({
                            "year": str(y),
                            "start_price": round(start_price, 2),
                            "end_price": round(end_price, 2),
                            "return_pct": round(yoy_pct, 2),
                            "high": round(float(df_year["High"].max()), 2),
                            "low": round(float(df_year["Low"].min()), 2),
                        })
        except Exception:
            yearly_returns = []

        result = {
            "symbol": ticker,
            "hourly": {
                "trend": hourly_trend,
                "volatility_pct": hourly_volatility,
                "recent_bars": hourly_data,
            },
            "monthly": monthly_returns,
            "yearly": yearly_returns,
        }
        self._set_cache(cache_key, result)
        return result

    def fetch_company_info(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch company metadata, current quote, and profile information.
        """
        ticker = str(ticker).replace(" ", "").upper()
        cache_key = f"info_{ticker}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}

            fast_history = stock.history(period="5d", auto_adjust=True)
            latest_close = None
            prev_close = None
            day_high = None
            day_low = None
            day_volume = None

            if not fast_history.empty and len(fast_history) >= 1:
                latest_close = float(fast_history["Close"].iloc[-1])
                day_high = float(fast_history["High"].iloc[-1])
                day_low = float(fast_history["Low"].iloc[-1])
                day_volume = int(fast_history["Volume"].iloc[-1])
                if len(fast_history) >= 2:
                    prev_close = float(fast_history["Close"].iloc[-2])
                else:
                    prev_close = float(info.get("previousClose", latest_close))

            current_price = info.get("currentPrice") or latest_close or 0.0
            previous_close = info.get("previousClose") or prev_close or current_price
            price_change = current_price - previous_close if previous_close else 0.0
            price_change_pct = (price_change / previous_close * 100.0) if previous_close else 0.0

            result = {
                "symbol": ticker,
                "name": info.get("shortName") or info.get("longName") or ticker,
                "currency": info.get("currency", "USD"),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "current_price": round(current_price, 2),
                "previous_close": round(previous_close, 2),
                "change": round(price_change, 2),
                "change_percent": round(price_change_pct, 2),
                "day_high": round(info.get("dayHigh") or day_high or current_price, 2),
                "day_low": round(info.get("dayLow") or day_low or current_price, 2),
                "fifty_two_week_high": round(info.get("fiftyTwoWeekHigh", 0.0), 2),
                "fifty_two_week_low": round(info.get("fiftyTwoWeekLow", 0.0), 2),
                "market_cap": info.get("marketCap", 0),
                "volume": info.get("volume") or day_volume or 0,
                "summary": info.get("longBusinessSummary", "No company summary available."),
                "website": info.get("website", ""),
            }
            self._set_cache(cache_key, result)
            return result

        except Exception as e:
            return {
                "symbol": ticker,
                "name": ticker,
                "currency": "USD",
                "sector": "N/A",
                "industry": "N/A",
                "current_price": 0.0,
                "previous_close": 0.0,
                "change": 0.0,
                "change_percent": 0.0,
                "day_high": 0.0,
                "day_low": 0.0,
                "fifty_two_week_high": 0.0,
                "fifty_two_week_low": 0.0,
                "market_cap": 0,
                "volume": 0,
                "summary": f"Could not retrieve full profile: {str(e)}",
                "website": "",
            }
