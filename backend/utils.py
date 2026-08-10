"""
Utility and Evaluation Module
Calculates regression metrics and indicator signals.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, prev_prices: np.ndarray = None) -> Dict[str, float]:
    """Calculate RMSE, MAE, MAPE, R2, and Directional Accuracy."""
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {
            "rmse": 0.0,
            "mae": 0.0,
            "mape": 0.0,
            "r2": 0.0,
            "directional_accuracy": 50.0,
        }

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    non_zero = y_true != 0
    mape = float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100.0) if np.any(non_zero) else 0.0

    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = 0.0

    dir_acc = 50.0
    if len(y_true) >= 2:
        if prev_prices is not None and len(prev_prices) == len(y_true):
            actual_dir = np.sign(y_true - prev_prices)
            pred_dir = np.sign(y_pred - prev_prices)
        else:
            actual_dir = np.sign(np.diff(y_true))
            pred_dir = np.sign(y_pred[1:] - y_true[:-1])

        correct_dir = np.sum(actual_dir == pred_dir)
        dir_acc = float((correct_dir / len(actual_dir)) * 100.0)

    return {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "mape": round(mape, 2),
        "r2": round(r2, 4),
        "directional_accuracy": round(dir_acc, 2),
    }


def generate_technical_explanations(df_features: pd.DataFrame) -> Dict[str, Any]:
    """Generate plain-English explanations of technical indicators."""
    if df_features.empty:
        return {"summary": "Insufficient data for technical analysis.", "signals": []}

    latest = df_features.iloc[-1]
    close = float(latest.get("Close", 0.0))
    sma_20 = float(latest.get("SMA_20", close))
    sma_50 = float(latest.get("SMA_50", close))
    rsi = float(latest.get("RSI_14", 50.0))
    macd = float(latest.get("MACD", 0.0))
    macd_signal = float(latest.get("MACD_Signal", 0.0))
    volatility = float(latest.get("Volatility_20", 0.0)) * 100.0

    signals: List[Dict[str, str]] = []
    bullish_points = 0
    bearish_points = 0

    # RSI
    if rsi >= 70:
        signals.append({
            "indicator": "RSI (14-Day)",
            "value": f"{rsi:.1f}",
            "status": "Overbought",
            "type": "bearish",
            "explanation": f"RSI is at {rsi:.1f} (>70), indicating potential near-term resistance or overbought conditions.",
        })
        bearish_points += 1
    elif rsi <= 30:
        signals.append({
            "indicator": "RSI (14-Day)",
            "value": f"{rsi:.1f}",
            "status": "Oversold",
            "type": "bullish",
            "explanation": f"RSI is at {rsi:.1f} (<30), signaling oversold conditions and potential mean-reversion rebound.",
        })
        bullish_points += 1
    else:
        signals.append({
            "indicator": "RSI (14-Day)",
            "value": f"{rsi:.1f}",
            "status": "Neutral Momentum",
            "type": "neutral",
            "explanation": f"RSI is at {rsi:.1f}, reflecting balanced market momentum.",
        })

    # SMA 50
    if close >= sma_50:
        signals.append({
            "indicator": "50-Day SMA Trend",
            "value": f"{sma_50:.2f}",
            "status": "Bullish Trend",
            "type": "bullish",
            "explanation": f"Current price ({close:.2f}) is above the 50-day SMA ({sma_50:.2f}), confirming upward trend.",
        })
        bullish_points += 1
    else:
        signals.append({
            "indicator": "50-Day SMA Trend",
            "value": f"{sma_50:.2f}",
            "status": "Bearish Pressure",
            "type": "bearish",
            "explanation": f"Current price ({close:.2f}) is below the 50-day SMA ({sma_50:.2f}), indicating downward pressure.",
        })
        bearish_points += 1

    # MACD
    if macd >= macd_signal:
        signals.append({
            "indicator": "MACD Crossover",
            "value": f"{macd:.2f} vs {macd_signal:.2f}",
            "status": "Positive Momentum",
            "type": "bullish",
            "explanation": "MACD line is above the Signal line, indicating positive momentum.",
        })
        bullish_points += 1
    else:
        signals.append({
            "indicator": "MACD Crossover",
            "value": f"{macd:.2f} vs {macd_signal:.2f}",
            "status": "Negative Momentum",
            "type": "bearish",
            "explanation": "MACD line is below the Signal line, indicating negative momentum.",
        })
        bearish_points += 1

    signals.append({
        "indicator": "20-Day Volatility",
        "value": f"{volatility:.2f}%",
        "status": "High Volatility" if volatility > 3.0 else "Stable Volatility",
        "type": "neutral",
        "explanation": f"20-day historical standard deviation of returns is {volatility:.2f}%.",
    })

    if bullish_points > bearish_points:
        consensus = "BULLISH"
        summary = f"Technical indicators lean Bullish ({bullish_points} positive vs {bearish_points} negative signals)."
    elif bearish_points > bullish_points:
        consensus = "BEARISH"
        summary = f"Technical indicators lean Bearish ({bearish_points} negative vs {bullish_points} positive signals)."
    else:
        consensus = "NEUTRAL"
        summary = "Technical indicators are balanced with mixed signals."

    return {
        "consensus": consensus,
        "summary": summary,
        "bullish_count": bullish_points,
        "bearish_count": bearish_points,
        "signals": signals,
        "current_values": {
            "close": close,
            "sma_20": round(sma_20, 2),
            "sma_50": round(sma_50, 2),
            "rsi_14": round(rsi, 2),
            "macd": round(macd, 2),
            "macd_signal": round(macd_signal, 2),
            "volatility_pct": round(volatility, 2),
        },
    }
