import numpy as np
from pyn_utils.data_utils import normalize

# === Этап 2, Тема 7: NumPy: массивы, операции, broadcasting ===

class ArrayProcessor:
    """
    Класс для базовой обработки и трансформации массивов данных
    (предназначен для дальнейшего использования при обучении моделей).
    """

    def normalize(self, data: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Нормализация данных (min-max)."""

        return np.asarray(normalize(data, scale=scale))


    def standardize(self, data: np.ndarray) -> np.ndarray:
        """Стандартизация данных."""
        
        mean = np.mean(data)
        std = np.std(data)

        return np.asarray((data - mean) / std)

    def combine_arrays(self, arrays: list[np.ndarray], axis=0) -> np.ndarray:
        """Объединяет несколько массивов."""
        
        if axis == 1:
            rows = [array.shape[0] for array in arrays]
            if all(row == rows[0] for row in rows):
                combined = np.hstack(arrays)
            else:
                raise ValueError("Массивы должны иметь совместимые размеры")
        else:
            columns = [array.shape[1] for array in arrays]
            if all(column == columns[0] for column in columns):
                combined = np.vstack(arrays)
            else:
                raise ValueError("Массивы должны иметь совместимые размеры")

        return np.asarray(combined)

    def apply_mask(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Применяет булеву маску к данным."""
        
        if data.shape == mask.shape:
            return np.asarray(data[mask])
        else:
            raise ValueError("Маска дожна быть того же размера, что и данные")

    def scale_broadcast(self, data: np.ndarray, factors: np.ndarray) -> np.ndarray:
        """Масштабирует данные с использованием broadcasting."""
        
        return np.asarray(data * factors)