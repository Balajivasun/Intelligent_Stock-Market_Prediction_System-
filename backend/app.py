"""
Main Flask REST API Server
Orchestrates data fetching, multi-timeframe analytics, model predictions, and dashboard UI serving.
"""

import os
import sys
from typing import Dict, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np

from backend.data_fetcher import StockDataFetcher
from backend.preprocessor import StockPreprocessor
from backend.utils import calculate_metrics, generate_technical_explanations
from backend.sentiment import StockSentimentAnalyzer
from backend.models.linear_model import LinearStockModel
from backend.models.rf_model import RandomForestStockModel
from backend.models.lstm_model import LSTMStockModel

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR)

fetcher = StockDataFetcher()
preprocessor = StockPreprocessor(window_size=60)
sentiment_analyzer = StockSentimentAnalyzer()


# --- Static Frontend Serving Routes ---

@app.route("/")
def serve_dashboard():
    """Serve the main frontend dashboard."""
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/css/<path:filename>")
def serve_css(filename):
    return send_from_directory(os.path.join(FRONTEND_DIR, "css"), filename)


@app.route("/js/<path:filename>")
def serve_js(filename):
    return send_from_directory(os.path.join(FRONTEND_DIR, "js"), filename)


# --- REST API Routes ---

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check route."""
    return jsonify({"status": "healthy", "service": "Intelligent Stock Market Prediction API"})


@app.route("/api/stock/info", methods=["GET"])
def get_stock_info():
    """Fetch company profile, quote, and stats."""
    raw_ticker = request.args.get("ticker", "AAPL")
    ticker = raw_ticker.replace(" ", "").upper()
    try:
        info = fetcher.fetch_company_info(ticker)
        return jsonify({"status": "success", "data": info})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/stock/history", methods=["GET"])
def get_stock_history():
    """Fetch historical OHLCV and indicators for charts."""
    raw_ticker = request.args.get("ticker", "AAPL")
    ticker = raw_ticker.replace(" ", "").upper()
    period = request.args.get("period", "1y").strip().lower()
    interval = request.args.get("interval", "1d").strip().lower()

    try:
        df = fetcher.fetch_history(ticker, period=period, interval=interval)
        df_feat = preprocessor.add_technical_indicators(df)

        dates = [
            d.strftime("%Y-%m-%d %H:%M") if interval == "1h" else d.strftime("%Y-%m-%d")
            for d in df_feat.index
        ]
        history_data = {
            "dates": dates,
            "close": [round(float(v), 2) for v in df_feat["Close"]],
            "open": [round(float(v), 2) for v in df_feat["Open"]],
            "high": [round(float(v), 2) for v in df_feat["High"]],
            "low": [round(float(v), 2) for v in df_feat["Low"]],
            "volume": [int(v) for v in df_feat["Volume"]],
            "sma_20": [round(float(v), 2) if not np.isnan(v) else None for v in df_feat["SMA_20"]],
            "sma_50": [round(float(v), 2) if not np.isnan(v) else None for v in df_feat["SMA_50"]],
            "ema_20": [round(float(v), 2) if not np.isnan(v) else None for v in df_feat["EMA_20"]],
            "rsi_14": [round(float(v), 2) if not np.isnan(v) else None for v in df_feat["RSI_14"]],
            "macd": [round(float(v), 2) if not np.isnan(v) else None for v in df_feat["MACD"]],
            "macd_signal": [round(float(v), 2) if not np.isnan(v) else None for v in df_feat["MACD_Signal"]],
        }

        return jsonify({
            "status": "success",
            "symbol": ticker,
            "period": period,
            "interval": interval,
            "data": history_data,
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/stock/timeframe-analysis", methods=["GET"])
def get_timeframe_analysis():
    """
    Fetch multi-timeframe comparative stats:
    - Hourly (Bottom-level intraday bars, hourly volatility & trend)
    - Monthly (Month-to-month return breakdown)
    - Yearly (Year-to-year annual performance)
    """
    raw_ticker = request.args.get("ticker", "AAPL")
    ticker = raw_ticker.replace(" ", "").upper()
    try:
        stats = fetcher.fetch_multi_timeframe_stats(ticker)
        return jsonify({"status": "success", "data": stats})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/stock/sentiment", methods=["GET"])
def get_stock_sentiment():
    """Fetch news articles and compute NLP sentiment polarity score."""
    raw_ticker = request.args.get("ticker", "AAPL")
    ticker = raw_ticker.replace(" ", "").upper()
    try:
        sentiment_data = sentiment_analyzer.analyze_ticker_news(ticker, max_articles=8)
        return jsonify({"status": "success", "data": sentiment_data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/stock/explain", methods=["GET"])
def get_stock_explanation():
    """Provide plain-English technical indicator signal decomposition."""
    raw_ticker = request.args.get("ticker", "AAPL")
    ticker = raw_ticker.replace(" ", "").upper()
    try:
        df = fetcher.fetch_history(ticker, period="1y")
        df_feat = preprocessor.add_technical_indicators(df)
        explanation = generate_technical_explanations(df_feat)
        return jsonify({"status": "success", "symbol": ticker, "data": explanation})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/stock/predict", methods=["POST"])
def predict_stock():
    """Train models and generate next-day and multi-day stock price predictions."""
    body = request.get_json(silent=True) or {}
    raw_ticker = body.get("ticker", "AAPL")
    ticker = str(raw_ticker).replace(" ", "").upper()
    period = body.get("period", "2y").strip().lower()
    model_type = body.get("model_type", "all").strip().lower()

    try:
        df = fetcher.fetch_history(ticker, period=period)
        if len(df) < 70:
            df = fetcher.fetch_history(ticker, period="2y")

        df_feat = preprocessor.add_technical_indicators(df)
        latest_actual_close = float(df_feat["Close"].iloc[-1])

        results = {}
        comparison_table = []
        next_day_target = latest_actual_close
        five_day_forecast = []
        selected_model_name = "LSTM Deep Learning"

        X_train_tab, y_train_tab, X_test_tab, y_test_tab, feat_cols = preprocessor.prepare_tabular_data(df_feat)
        latest_tab_row = df_feat[feat_cols].iloc[[-1]].copy()

        X_train_seq, y_train_seq, X_test_seq, y_test_seq, scaler = preprocessor.prepare_lstm_data(df_feat)
        scaled_close = scaler.transform(df_feat["Close"].values.reshape(-1, 1))
        latest_seq_scaled = scaled_close[-60:].flatten()

        last_date = df.index[-1]
        future_dates = []
        curr_date = last_date
        while len(future_dates) < 5:
            curr_date = curr_date + pd.Timedelta(days=1)
            if curr_date.weekday() < 5:
                future_dates.append(curr_date.strftime("%Y-%m-%d"))

        # 1. Linear Regression
        if model_type in ["linear", "all"]:
            lin_model = LinearStockModel()
            lin_eval = lin_model.train_and_evaluate(X_train_tab, y_train_tab, X_test_tab, y_test_tab)
            lin_forecast = lin_model.predict_next_days(latest_tab_row, days=5)
            results["linear"] = {**lin_eval, "forecast_5d": lin_forecast}
            comparison_table.append({
                "model": "Linear Regression",
                "rmse": lin_eval["metrics"]["rmse"],
                "mae": lin_eval["metrics"]["mae"],
                "mape": lin_eval["metrics"]["mape"],
                "directional_accuracy": lin_eval["metrics"]["directional_accuracy"],
                "next_day_pred": lin_forecast[0] if lin_forecast else latest_actual_close,
            })
            if model_type == "linear":
                selected_model_name = "Linear Regression"
                next_day_target = lin_forecast[0]
                five_day_forecast = lin_forecast

        # 2. Random Forest Regressor
        if model_type in ["rf", "all"]:
            rf_model = RandomForestStockModel(n_estimators=80, random_state=42)
            rf_eval = rf_model.train_and_evaluate(X_train_tab, y_train_tab, X_test_tab, y_test_tab)
            rf_forecast = rf_model.predict_next_days(latest_tab_row, days=5)
            results["rf"] = {**rf_eval, "forecast_5d": rf_forecast}
            comparison_table.append({
                "model": "Random Forest",
                "rmse": rf_eval["metrics"]["rmse"],
                "mae": rf_eval["metrics"]["mae"],
                "mape": rf_eval["metrics"]["mape"],
                "directional_accuracy": rf_eval["metrics"]["directional_accuracy"],
                "next_day_pred": rf_forecast[0] if rf_forecast else latest_actual_close,
            })
            if model_type == "rf":
                selected_model_name = "Random Forest Regressor"
                next_day_target = rf_forecast[0]
                five_day_forecast = rf_forecast

        # 3. LSTM Deep Learning
        if model_type in ["lstm", "all"]:
            lstm = LSTMStockModel(hidden_dim=64, num_layers=2, epochs=25, lr=0.006)
            lstm_eval = lstm.train_and_evaluate(X_train_seq, y_train_seq, X_test_seq, y_test_seq, scaler)
            lstm_forecast = lstm.predict_next_days(latest_seq_scaled, days=5)
            results["lstm"] = {**lstm_eval, "forecast_5d": lstm_forecast}
            comparison_table.append({
                "model": "LSTM Neural Network",
                "rmse": lstm_eval["metrics"]["rmse"],
                "mae": lstm_eval["metrics"]["mae"],
                "mape": lstm_eval["metrics"]["mape"],
                "directional_accuracy": lstm_eval["metrics"]["directional_accuracy"],
                "next_day_pred": lstm_forecast[0] if lstm_forecast else latest_actual_close,
            })
            if model_type == "lstm":
                selected_model_name = "LSTM Deep Learning"
                next_day_target = lstm_forecast[0]
                five_day_forecast = lstm_forecast

        pred_change = next_day_target - latest_actual_close
        pred_change_pct = (pred_change / latest_actual_close) * 100.0 if latest_actual_close else 0.0

        if pred_change_pct > 0.5:
            signal = "BULLISH (UPWARD)"
            signal_color = "positive"
        elif pred_change_pct < -0.5:
            signal = "BEARISH (DOWNWARD)"
            signal_color = "negative"
        else:
            signal = "NEUTRAL (CONSOLIDATION)"
            signal_color = "neutral"

        tech_exp = generate_technical_explanations(df_feat)

        response_data = {
            "symbol": ticker,
            "period": period,
            "model_type": model_type,
            "selected_model_name": selected_model_name,
            "current_price": round(latest_actual_close, 2),
            "next_day_predicted_price": round(next_day_target, 2),
            "predicted_change": round(pred_change, 2),
            "predicted_change_percent": round(pred_change_pct, 2),
            "signal": signal,
            "signal_color": signal_color,
            "future_dates": future_dates,
            "future_forecast": five_day_forecast,
            "comparison_table": comparison_table,
            "models": results,
            "technical_indicators": tech_exp,
        }

        return jsonify({"status": "success", "data": response_data})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# --- Global Error Handlers & Fallback Routing ---

@app.errorhandler(404)
def handle_404(e):
    if request.path.startswith("/api/"):
        return jsonify({"status": "error", "message": "API endpoint not found"}), 404
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.errorhandler(500)
def handle_500(e):
    if request.path.startswith("/api/"):
        return jsonify({"status": "error", "message": "Internal Server Error"}), 500
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:path>")
def serve_fallback(path):
    if path.startswith("api/"):
        return jsonify({"status": "error", "message": "API endpoint not found"}), 404
    if os.path.exists(os.path.join(FRONTEND_DIR, path)):
        return send_from_directory(FRONTEND_DIR, path)
    return send_from_directory(FRONTEND_DIR, "index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Intelligent Stock Market Prediction Server at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
