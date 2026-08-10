"""
Linear Regression Baseline Model
"""

from typing import Dict, Any, List
import pandas as pd
from sklearn.linear_model import LinearRegression
from backend.utils import calculate_metrics


class LinearStockModel:
    def __init__(self):
        self.model = LinearRegression()
        self.feature_names: List[str] = []
        self.is_trained = False

    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Any]:
        self.feature_names = list(X_train.columns)
        self.model.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.model.predict(X_test)
        prev_test_prices = X_test["Close"].values if "Close" in X_test.columns else None

        metrics = calculate_metrics(y_test.values, y_pred, prev_prices=prev_test_prices)

        coefficients = [
            {"feature": feat, "weight": round(float(coef), 4)}
            for feat, coef in zip(self.feature_names, self.model.coef_)
        ]
        coefficients = sorted(coefficients, key=lambda x: abs(x["weight"]), reverse=True)

        return {
            "model_name": "Linear Regression (Baseline)",
            "metrics": metrics,
            "y_test": [round(float(v), 2) for v in y_test.values],
            "y_pred": [round(float(v), 2) for v in y_pred],
            "coefficients": coefficients[:8],
            "intercept": round(float(self.model.intercept_), 4),
        }

    def predict_next_days(
        self, latest_features: pd.DataFrame, days: int = 5
    ) -> List[float]:
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")

        forecasts = []
        curr_row = latest_features.copy()

        for _ in range(days):
            pred = float(self.model.predict(curr_row)[0])
            forecasts.append(round(pred, 2))

            if "Close_Lag_5" in curr_row.columns and "Close_Lag_3" in curr_row.columns:
                curr_row["Close_Lag_5"] = curr_row["Close_Lag_3"]
            if "Close_Lag_3" in curr_row.columns and "Close_Lag_2" in curr_row.columns:
                curr_row["Close_Lag_3"] = curr_row["Close_Lag_2"]
            if "Close_Lag_2" in curr_row.columns and "Close_Lag_1" in curr_row.columns:
                curr_row["Close_Lag_2"] = curr_row["Close_Lag_1"]
            if "Close_Lag_1" in curr_row.columns and "Close" in curr_row.columns:
                curr_row["Close_Lag_1"] = curr_row["Close"]
            if "Close" in curr_row.columns:
                curr_row["Close"] = pred

        return forecasts
