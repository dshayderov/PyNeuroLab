import pandas as pd
import numpy as np

class DataFrameProcessor:
    """
    Класс для обработки и анализа табличных данных с помощью pandas.
    Используется для подготовки и агрегации данных в обучающих выборках.
    """

    # === Базовые методы ===
    def from_dict(self, data: dict) -> pd.DataFrame:
        """Создает DataFrame из словаря."""

        return pd.DataFrame(data)

    def filter_rows(self, df: pd.DataFrame, condition: callable) -> pd.DataFrame:
        """Фильтрует строки DataFrame по условию."""

        return pd.DataFrame(df[condition])

    def add_column(self, df: pd.DataFrame, name: str, values) -> pd.DataFrame:
        """Добавляет новый столбец в DataFrame."""

        df[name] = values

        return pd.DataFrame(df)

    def group_and_aggregate(self, df: pd.DataFrame, by: str, agg_func: dict) -> pd.DataFrame:
        """Группирует и агрегирует данные."""

        return pd.DataFrame(df.groupby(by).agg(agg_func))

    def sort_by_column(self, df: pd.DataFrame, column: str, ascending: bool = True) -> pd.DataFrame:
        """Сортирует DataFrame по заданной колонке."""

        return pd.DataFrame(df.sort_values(column, ascending=ascending))

    def summarize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Возвращает краткую статистику DataFrame (аналог df.describe())."""

        return pd.DataFrame({
            'count': df.count(),
            'mean': df.mean(),
            'std': df.std(),
            'min': df.min(),
            '25%': df.quantile(0.25),
            '50%': df.median(),
            '75%': df.quantile(0.75),
            'max': df.max()
        })

    # === Методы расширенной обработки данных ===
    def detect_outliers(self, df: pd.DataFrame, method: str = "zscore", threshold: float = 3.0) -> pd.DataFrame:
        """Определяет выбросы по методу Z-score или IQR."""

    def encode_categorical(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """Преобразует категориальные признаки в числовые (one-hot encoding)."""

    def split_features_labels(self, df: pd.DataFrame, target_col: str):
        """Разделяет DataFrame на признаки (X) и целевую переменную (y)."""

    def balance_dataset(self, df: pd.DataFrame, target_col: str, strategy: str = "undersample") -> pd.DataFrame:
        """Балансирует набор данных по классам (undersampling/oversampling)."""

    def normalize_columns(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Применяет нормализацию (min-max) только к выбранным столбцам."""
