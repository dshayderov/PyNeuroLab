from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseModel(ABC):
    """
    Универсальный базовый класс для всех типов моделей:
    - классические ML модели (LinearRegression, LogisticRegression)
    - нейросети (PyTorch)
    - модели transformers (BERT, GPT, LLaMA)
    - генеративные модели (LLaMA, DeepSeek)
    - мультимодальные модели (LLaVA, CLIP)

    Цель — единый API, позволящий Trainer работать с любой моделью.
    """

    # ============================================================
    # === Основной интерфейс модели ===
    # ============================================================

    @abstractmethod
    def forward(self, X: Any, **kwargs) -> Any:
        """
        Прямой проход модели.
        Обязателен для любой модели.
        """
        raise NotImplementedError

    def predict(self, X: Any, **kwargs) -> Any:
        """
        Универсальное предсказание.
        ML-модели могут переопределить.
        Генеративные модели используют generate().
        """
        return self.forward(X, **kwargs)

    def generate(self, X: Optional[Any] = None, **kwargs) -> Any:
        """
        Для моделей, которые могут генерировать текст/изображения.
        Для негениративных моделей — ошибка.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support generate().")

    # ============================================================
    # === Обучение ===
    # ============================================================

    def fit(self, X: Any, y: Any = None, **kwargs) -> "BaseModel":
        """
        Универсальный метод обучения.
        Не обязателен — Trainer может обучать модель без fit().
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement fit().")

    # ------- Требуется для Trainer (пошаговое обучение) --------

    def train_step(self, X: Any, y: Any, **kwargs) -> float:
        """
        Один шаг обучения (forward + backward + update).
        Возвращает loss.

        ML-модели обязаны переопределить.
        PyTorch-модели также должны переопределить.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement train_step().")

    def loss_fn(self, y_pred: Any, y_true: Any) -> float:
        """
        Чистая функция потерь (без градиентов и обновлений).
        Обязательна для Trainer.

        ML-модели — MSE или BCE.
        Нейросети — свой loss.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement loss_fn().")

    # ============================================================
    # === Параметры модели ===
    # ============================================================

    def get_params(self) -> Dict[str, Any]:
        """
        Возвращает гиперпараметры модели.
        Для совместимости с GridSearch-like подходами.
        """
        params = {}
        for key, val in self.__dict__.items():
            if isinstance(val, (int, float, str, bool, type(None))):
                params[key] = val
        return params

    def set_params(self, **params) -> None:
        """
        Позволяет массово обновлять гиперпараметры.
        """
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)

    # ============================================================
    # === Сохранение и загрузка состояния (для Trainer resume) ===
    # ============================================================

    def save_state(self) -> Dict[str, Any]:
        """
        Возвращает состояние модели (веса + параметры), пригодное
        для сериализации.

        ML-модели → словарь с numpy массивами
        PyTorch-модели → state_dict()
        HF-модели → model.save_pretrained()
        """
        return {
            "class": self.__class__.__name__,
            "state": self.__dict__
        }

    def load_state(self, state_dict: Dict[str, Any]) -> None:
        """
        Восстановление состояния модели.
        Используется Trainer для паузы/возобновления.
        """
        self.__dict__.update(state_dict)

    # ============================================================
    # === Информация о модели (UI) ===
    # ============================================================

    def info(self) -> str:
        """Красивое строковое описание модели."""
        params = self.get_params()
        s = f"{self.__class__.__name__}("
        if params:
            s += ", ".join(f"{k}={v}" for k, v in params.items())
        s += ")"
        return s
