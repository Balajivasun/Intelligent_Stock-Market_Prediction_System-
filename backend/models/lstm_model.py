"""
PyTorch LSTM Deep Learning Model
"""

from typing import Dict, Any, List
import gc
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from backend.utils import calculate_metrics

# Restrict PyTorch to a single thread to maintain ultra-low memory on free-tier hosting
torch.set_num_threads(1)


class StockLSTMNetwork(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 32, num_layers: int = 2, dropout: float = 0.1):
        super(StockLSTMNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        return self.fc(last_time_step)


class LSTMStockModel:
    def __init__(self, hidden_dim: int = 32, num_layers: int = 2, epochs: int = 12, lr: float = 0.008):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.model: StockLSTMNetwork = None
        self.scaler: MinMaxScaler = None
        self.is_trained = False

    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        scaler: MinMaxScaler,
    ) -> Dict[str, Any]:
        self.scaler = scaler

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

        input_dim = X_train.shape[2] if len(X_train.shape) == 3 else 1
        self.model = StockLSTMNetwork(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=0.1,
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        batch_size = 64
        dataset_size = len(X_train)
        self.model.train()

        loss_history = []
        for _ in range(self.epochs):
            permutation = torch.randperm(dataset_size)
            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, dataset_size, batch_size):
                indices = permutation[i : i + batch_size]
                batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            loss_history.append(epoch_loss / max(1, num_batches))

        self.is_trained = True
        self.model.eval()

        with torch.no_grad():
            preds_scaled = self.model(X_test_tensor).numpy()

        y_test_inv = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_inv = self.scaler.inverse_transform(preds_scaled).flatten()
        prev_test_prices = self.scaler.inverse_transform(X_test[:, -1, 0].reshape(-1, 1)).flatten()

        metrics = calculate_metrics(y_test_inv, y_pred_inv, prev_prices=prev_test_prices)

        # Free temporary tensors
        del X_train_tensor, y_train_tensor, X_test_tensor
        gc.collect()

        return {
            "model_name": "LSTM Neural Network (Deep Learning)",
            "metrics": metrics,
            "y_test": [round(float(v), 2) for v in y_test_inv],
            "y_pred": [round(float(v), 2) for v in y_pred_inv],
            "training_loss": [round(float(l), 5) for l in loss_history],
            "architecture": f"2-Layer LSTM (Hidden Dim={self.hidden_dim}, Seq Len={X_train.shape[1]})",
        }

    def predict_next_days(self, last_sequence_scaled: np.ndarray, days: int = 5) -> List[float]:
        if not self.is_trained or self.model is None:
            raise ValueError("LSTM model has not been trained yet.")

        self.model.eval()
        curr_seq = list(last_sequence_scaled.flatten())
        window_size = len(curr_seq)
        forecasts = []

        with torch.no_grad():
            for _ in range(days):
                input_tensor = torch.tensor(curr_seq[-window_size:], dtype=torch.float32).view(1, window_size, 1)
                pred_scaled = self.model(input_tensor).item()

                pred_price = self.scaler.inverse_transform([[pred_scaled]])[0][0]
                forecasts.append(round(float(pred_price), 2))
                curr_seq.append(pred_scaled)

        gc.collect()
        return forecasts
