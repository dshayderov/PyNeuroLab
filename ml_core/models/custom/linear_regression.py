from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Any
from ml_core.base_model import BaseModel
from ml_core import metrics


class LinearRegression(BaseModel):
    """
    Линейная регрессия, совместимая с BaseModel и Trainer.
    Поддерживает batch/mini-batch обучение (через Trainer),
    L2-регуляризацию и историю.
    """

    def __init__(
        self,
        lr: float = 1e-2,
        l2: float = 0.0,
        verbose: bool = False
    ):
        self.lr = float(lr)
        self.l2 = float(l2)
        self.verbose = verbose

        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0

        self.history = {"loss": []}

    # -----------------------------
    # BaseModel API
    # -----------------------------

    def forward(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if self.weights is None:
            raise RuntimeError("Model is not trained yet.")
        return X @ self.weights + self.bias

    def loss_fn(self, y_pred, y_true):
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        mse = np.mean((y_pred - y_true) ** 2)
        if self.l2 > 0:
            mse += self.l2 * np.sum(self.weights ** 2)
        return float(mse)

    def train_step(self, X, y) -> float:
        """Один шаг градиентного спуска для Trainer."""
        X = np.asarray(X)
        y = np.asarray(y)

        # Инициализация
        if self.weights is None:
            self.weights = np.zeros(X.shape[1])
            self.bias = 0.0

        # Forward
        preds = self.forward(X)
        loss = self.loss_fn(preds, y)

        # Градиенты
        m = len(X)
        err = preds - y

        dw = (2 / m) * X.T @ err
        db = (2 / m) * np.sum(err)

        if self.l2 > 0:
            dw += 2 * self.l2 * self.weights

        # Обновление
        self.weights -= self.lr * dw
        self.bias -= self.lr * db

        return loss

    # -----------------------------
    # Дополнительный standalone-fit
    # -----------------------------
    def fit(self, X, y, epochs=1000):
        """Классическое обучение без Trainer."""
        X = np.asarray(X)
        y = np.asarray(y)

        if self.weights is None:
            self.weights = np.zeros(X.shape[1])

        for _ in range(epochs):
            loss = self.train_step(X, y)
            self.history["loss"].append(loss)

        return self

    # -----------------------------
    # Прочее
    # -----------------------------
    def predict(self, X):
        return self.forward(X)

    def evaluate(self, X, y):
        preds = self.predict(X)
        return {
            "mse": metrics.mean_squared_error(y, preds),
            "rmse": metrics.root_mean_squared_error(y, preds),
            "r2": metrics.r2_score(y, preds),
        }
