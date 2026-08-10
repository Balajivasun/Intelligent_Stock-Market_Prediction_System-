"""
Random Forest Regressor Model
"""

from typing import Dict, Any, List
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from backend.utils import calculate_metrics


class RandomForestStockModel:
    def __init__(self, n_estimators: int = 30, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=8,
            min_samples_split=5,
            random_state=random_state,
            n_jobs=1,
        )
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

        importances = self.model.feature_importances_
        feature_importance_list = [
            {"feature": feat, "importance": round(float(imp) * 100.0, 2)}
            for feat, imp in zip(self.feature_names, importances)
        ]
        feature_importance_list = sorted(
            feature_importance_list, key=lambda x: x["importance"], reverse=True
        )

        return {
            "model_name": "Random Forest Regressor",
            "metrics": metrics,
            "y_test": [round(float(v), 2) for v in y_test.values],
            "y_pred": [round(float(v), 2) for v in y_pred],
            "feature_importances": feature_importance_list,
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
