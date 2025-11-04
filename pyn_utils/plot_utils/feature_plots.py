import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_distribution(df: pd.DataFrame, column: str,
                      kind: str = "hist",
                      figsize: tuple = (8, 5),
                      palette: str = "pastel",
                      **kwargs):
    """
    Построить распределение значений колонки `column` DataFrame `df`.
    kind: один из {"hist", "kde", "box", "violin"}.
    figsize: размер фигуры.
    palette: палитра цветов.
    **kwargs: дополнительные аргументы Seaborn/Matplotlib.
    """

    # Выбор типа графика
    if kind == "hist":
        func = sns.histplot
    elif kind == "kde":
        func = sns.kdeplot
    elif kind == "box":
        func = sns.boxplot
    elif kind == "violin":
        func = sns.violinplot
    else:
        raise ValueError(f"Задан неправильный тип графика: {kind}")
    
    # Построение распределения
    plt.figure(figsize=(figsize[0], figsize[1]))
    func(data=df, x=column, palette=palette, **kwargs)
    plt.tight_layout()
    plt.show()
    

def plot_correlation_matrix(df: pd.DataFrame,
                            method: str = "pearson",
                            annot: bool = True,
                            mask_upper_triangle: bool = False,
                            figsize: tuple = (10, 8),
                            cmap: str = "vlag",
                            **kwargs):
    """
    Построить тепловую карту корреляционной матрицы числовых столбцов `df`.
    method: корреляция (pearson, spearman, kendall).
    annot: показывать ли значения в клетках.
    mask_upper_triangle: если True — замаскировать верхнюю треугольную часть.
    figsize: размер фигуры.
    cmap: цветовая карта.
    **kwargs: дополнительные параметры Seaborn.
    """

    # Выбираем только числовые столбцы
    numeric_df = df.select_dtypes(include=['number'])
    
    # Вычисляем корреляционную матрицу
    correlation_matrix = numeric_df.corr(method=method)

    # Проверяем маску для верхнего треугольника
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool)) if mask_upper_triangle else None

    # Построение тепловой карты
    plt.figure(figsize=(figsize[0], figsize[1]))
    sns.heatmap(correlation_matrix, annot=annot, cmap=cmap, mask=mask **kwargs)
    plt.tight_layout()
    plt.show()


def plot_pairwise(df: pd.DataFrame,
                  columns: list[str] = None,
                  hue: str = None,
                  kind: str = "scatter",
                  diag_kind: str = "hist",
                  **kwargs):
    """
    Построить pair-плот для столбцов `columns` DataFrame `df`.  
    hue: категориальная колонка для окраски точек.
    kind: вид графика для off-diagonal (scatter, kde).
    diag_kind: вид графика для диагонали (hist, kde).
    figsize: размер фигуры.
    **kwargs: дополнительные параметры Seaborn.
    """
    
    # Построение pair-plot
    if columns is not None:
        df = df[columns]
    sns.pairplot(df, hue=hue, kind=kind, diag_kind=diag_kind, **kwargs)
    plt.show()


def plot_feature_importance(scores,
                             title: str = "Важность признаков",
                             figsize: tuple = (8, 6),
                             palette: str = "deep",
                             **kwargs):
    """
    Построить бар-график важности признаков.
    scores: pd.Series с индексами = названия признаков и значениями = важности,
            или dict {feature: score}.
    title: заголовок графика.
    figsize: размер фигуры.
    palette: цветовая палитра.
    **kwargs: дополнительные параметры Matplotlib.
    """
    
    # Преобразование scores в Series
    data_series = pd.Series(scores).sort_values(ascending=True)
    
    # Построение бар-графика важности признаков
    plt.figure(figsize=figsize)
    sns.barplot(x=data_series.values, y=data_series.index, palette=palette, **kwargs)
    plt.title(title)
    plt.xlabel("Важность")
    plt.ylabel("Признак")
    plt.tight_layout()
    plt.show()
    
