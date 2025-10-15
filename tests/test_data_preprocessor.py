import sys
sys.path.append('./')
import pandas as pd
import numpy as np
from pyn_utils import FileHandler, DataPreprocessor


def test_data_preprocessor():
    # --- Подготовка данных ---
    df = pd.DataFrame({
        "A": [1, 2, np.nan, 4],
        "B": [10, 20, 30, 40],
        "C": ["x", "y", "z", "w"]
    })

    file_handler = FileHandler()
    preprocessor = DataPreprocessor(file_handler)

    # --- Проверка загрузки ---
    # эмулируем загрузку (не читаем файл, чтобы не зависеть от файловой системы)
    preprocessor.df = df.copy()
    assert isinstance(preprocessor.df, pd.DataFrame)

    # --- Проверка очистки ---
    cleaned = preprocessor.clean(method='fill', fill_value=0)
    assert not cleaned.df.isnull().values.any(), "После fill не должно быть NaN"

    dropped = preprocessor.clean(method='drop')
    assert dropped.df.isnull().sum().sum() == 0, "После drop не должно быть NaN"

    interpolated = preprocessor.clean(method='interpolate')
    assert isinstance(interpolated.df, pd.DataFrame), "interpolate должен вернуть DataFrame"

    # --- Проверка нормализации ---
    df_for_norm = pd.DataFrame({
        "num1": [1, 2, 3],
        "num2": [10, 20, 30]
    })
    preprocessor.df = df_for_norm.copy()

    normed = preprocessor.normalize(method='minmax')
    for col in ["num1", "num2"]:
        assert np.isclose(normed.df[col].min(), 0), f"min {col} должен быть 0"
        assert np.isclose(normed.df[col].max(), 1), f"max {col} должен быть 1"

    standardized = preprocessor.normalize(method='zscore')
    for col in ["num1", "num2"]:
        mean = standardized.df[col].mean()
        std = standardized.df[col].std(ddof=0)
        assert np.isclose(mean, 0, atol=1e-6), f"mean {col} должен быть ~0"
        assert np.isclose(std, 1, atol=1e-6), f"std {col} должен быть ~1"

    # --- Проверка экспорта ---
    # эмулируем сохранение без записи файла
    class DummyFileHandler(FileHandler):
        def save(self, df, path, **kwargs):
            self.saved_df = df
            self.saved_path = path
            self.saved_kwargs = kwargs
            return path

    dummy = DummyFileHandler()
    preprocessor = DataPreprocessor(dummy)
    preprocessor.df = pd.DataFrame({"x": [1, 2, 3]})
    result_path = preprocessor.export("output.csv", index=False)
    assert result_path == "output.csv"
    assert isinstance(dummy.saved_df, pd.DataFrame)
    assert dummy.saved_kwargs == {"index": False}

    print("✅ Все тесты пройдены успешно!")


if __name__ == "__main__":
    test_data_preprocessor()
