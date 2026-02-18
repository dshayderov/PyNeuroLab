import os
import pickle
import numpy as np
import random
from typing import Optional, Dict, Any, Callable


class Trainer:
    """
    Универсальный тренер для моделей, реализующих API BaseModel.
    Поддерживает:
    - обучение по эпохам
    - mini-batch обучение
    - логирование истории
    - early stopping
    - чекпоинты
    - возобновление обучения
    """

    def __init__(
        self,
        model,
        metric_fn: Optional[Callable] = None,
        val_metric_fn: Optional[Callable] = None,
        early_stopping: Optional[Dict[str, Any]] = None,
        verbose: bool = True
    ):
        self.model = model
        self.metric_fn = metric_fn
        self.val_metric_fn = val_metric_fn
        self.early_stopping = early_stopping
        self.verbose = verbose

        self.history = {
            "loss": [],
            "metric": [],
            "val_loss": [],
            "val_metric": []
        }

        # состояние для возобновления
        self._trainer_state: Dict[str, Any] = {
            "epoch": 0,
            "rng_state": None,
        }

    # ----------------------------------------------------------------------
    # -------------------------  CHECKPOINT API  ---------------------------
    # ----------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Сохраняет состояние модели и тренера."""
        checkpoint = {
            "model_state": self.model.save_state(),
            "trainer_state": self._trainer_state,
            "history": self.history,
        }

        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)

        if self.verbose:
            print(f"[Trainer] Checkpoint saved → {path}")

    def load_checkpoint(self, path: str) -> None:
        """Загружает состояние модели и тренера."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint found at {path}")

        with open(path, "rb") as f:
            checkpoint = pickle.load(f)

        self.model.load_state(checkpoint["model_state"])
        self._trainer_state = checkpoint["trainer_state"]
        self.history = checkpoint["history"]

        # восстановим RNG для корректного продолжения
        if self._trainer_state.get("rng_state") is not None:
            random.setstate(self._trainer_state["rng_state"])

        if self.verbose:
            print(f"[Trainer] Checkpoint loaded ← {path}")

    # ----------------------------------------------------------------------
    # -------------------------    TRAIN LOOP   ----------------------------
    # ----------------------------------------------------------------------

    def train(
        self,
        X,
        y,
        epochs: int,
        batch_size: Optional[int] = None,
        X_val=None,
        y_val=None,
        shuffle: bool = True,
        resume: bool = False
    ):
        """
        Запускает обучение.
        Если resume=True — продолжается с последнего сохранённого состояния.
        """

        # -----------------------------------------------------
        # восстановление состояния
        # -----------------------------------------------------
        start_epoch = 0
        if resume:
            start_epoch = self._trainer_state.get("epoch", 0)
            if self.verbose:
                print(f"[Trainer] Resuming training from epoch {start_epoch}")

        # -----------------------------------------------------
        # гиперпараметры
        # -----------------------------------------------------
        use_batches = batch_size is not None

        # -----------------------------------------------------
        # сквозной цикл по эпохам
        # -----------------------------------------------------
        for epoch in range(start_epoch, epochs):
            # сохранить состояние RNG (нужно для корректного резюма)
            self._trainer_state["rng_state"] = random.getstate()

            # перемешивание данных
            if shuffle:
                indices = np.random.permutation(len(X))
                X, y = X[indices], y[indices]

            # mini-batch разбиение
            if use_batches:
                batches = range(0, len(X), batch_size)
            else:
                batches = [0]

            epoch_losses = []

            # -----------------------------------------------------
            # цикл по батчам
            # -----------------------------------------------------
            for i in batches:
                if use_batches:
                    X_batch = X[i : i + batch_size]
                    y_batch = y[i : i + batch_size]
                else:
                    X_batch = X
                    y_batch = y

                loss = self.model.train_step(X_batch, y_batch)
                epoch_losses.append(loss)

            # -----------------------------------------------------
            # логирование
            # -----------------------------------------------------
            train_loss = float(np.mean(epoch_losses))
            self.history["loss"].append(train_loss)

            # метрики
            if self.metric_fn is not None:
                pred = self.model.predict(X)
                metric_value = float(self.metric_fn(y, pred))
                self.history["metric"].append(metric_value)
            else:
                metric_value = None

            # валидация
            if X_val is not None and y_val is not None:
                val_pred = self.model.predict(X_val)
                val_loss = float(self.model.loss_fn(val_pred, y_val))
                self.history["val_loss"].append(val_loss)

                if self.val_metric_fn is not None:
                    val_metric_value = float(self.val_metric_fn(y_val, val_pred))
                    self.history["val_metric"].append(val_metric_value)
                else:
                    val_metric_value = None
            else:
                val_loss = None
                val_metric_value = None

            # вывод
            if self.verbose:
                msg = (
                    f"Epoch {epoch+1}/{epochs} — "
                    f"loss: {train_loss:.4f}"
                )
                if metric_value is not None:
                    msg += f", metric: {metric_value:.4f}"
                if val_loss is not None:
                    msg += f", val_loss: {val_loss:.4f}"
                if val_metric_value is not None:
                    msg += f", val_metric: {val_metric_value:.4f}"

                print(msg)

            # -----------------------------------------------------
            # обновить состояние для resume
            # -----------------------------------------------------
            self._trainer_state["epoch"] = epoch + 1

            # -----------------------------------------------------
            # early stopping
            # -----------------------------------------------------
            if self.early_stopping is not None:
                if self._check_early_stopping():
                    if self.verbose:
                        print("[Trainer] Early stopping triggered.")
                    break

    # ----------------------------------------------------------------------
    # ----------------------   EARLY STOPPING  -----------------------------
    # ----------------------------------------------------------------------

    def _check_early_stopping(self) -> bool:
        cfg = self.early_stopping
        if cfg is None:
            return False

        patience = cfg.get("patience", 5)
        min_delta = cfg.get("min_delta", 1e-4)

        val_loss_history = self.history["val_loss"]
        if len(val_loss_history) < patience + 1:
            return False

        recent = val_loss_history[-patience - 1 :]

        # проверяем: улучшение < min_delta
        return all(
            recent[i] - recent[i + 1] < min_delta
            for i in range(len(recent) - 1)
        )
