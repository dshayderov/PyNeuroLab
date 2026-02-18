from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Any
from ml_core.base_model import BaseModel
from ml_core.metrics import accuracy, precision, recall, f1_score


class LogisticRegression(BaseModel):
    """
    Бинарная логистическая регрессия, полностью совместимая с BaseModel и Trainer.

    Trainer использует:
        - forward()
        - loss_fn()
        - train_step()

    fit() реализован отдельно — для стандартного обучения без Trainer.
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

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def forward(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if self.weights is None:
            raise RuntimeError("Model is not trained yet.")
        logits = X @ self.weights + self.bias
        return self.sigmoid(logits)

    def loss_fn(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Бинарная кросс-энтропия + L2-регуляризация.
        """
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        y_true = np.asarray(y_true)

        ce = -np.mean(
            y_true * np.log(y_pred) +
            (1 - y_true) * np.log(1 - y_pred)
        )

        if self.l2 > 0 and self.weights is not None:
            ce += self.l2 * np.sum(self.weights ** 2)

        return float(ce)

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Один шаг обучения — используется Trainer'ом.
        """
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)

        n_samples, n_features = X.shape

        # Инициализация параметров
        if self.weights is None:
            self.weights = np.zeros(n_features)
            self.bias = 0.0

        # Forward
        logits = X @ self.weights + self.bias
        probs = self.sigmoid(logits)

        # Loss
        loss = self.loss_fn(probs, y)

        # Градиенты
        error = probs - y
        m = len(X)

        dw = (X.T @ error) / m
        db = np.sum(error) / m

        if self.l2 > 0:
            dw += 2 * self.l2 * self.weights

        # Обновление
        self.weights -= self.lr * dw
        self.bias -= self.lr * db

        return loss

    # -----------------------------
    # Standalone-fit (как sklearn)
    # -----------------------------
    def fit(self, X, y, epochs=1000):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)

        n_samples, n_features = X.shape
        if self.weights is None:
            self.weights = np.zeros(n_features)

        for _ in range(epochs):
            loss = self.train_step(X, y)
            self.history["loss"].append(loss)

            if self.verbose:
                print(f"loss={loss:.6f}")

        return self

    # -----------------------------
    # Pred / evaluate
    # -----------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        y = np.asarray(y)
        y_pred = self.predict(X)
        return {
            "accuracy": float(accuracy(y, y_pred)),
            "precision": float(precision(y, y_pred)),
            "recall": float(recall(y, y_pred)),
            "f1": float(f1_score(y, y_pred)),
        }
