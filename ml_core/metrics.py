"""
metrics.py — базовые метрики для оценки моделей
(регрессия и классификация)
"""

import numpy as np


# === РЕГРЕССИЯ ===

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error (MAE)

    Формула:
        MAE = (1/n) * Σ |y_i - ŷ_i|
    """
    # Подсказка:
    # 1. Преобразуй входы в numpy-массивы.
    # 2. Возьми абсолютное значение разности.
    # 3. Найди среднее.
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = np.mean(np.abs(y_true - y_pred))

    return float(mae)


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error (MSE)

    Формула:
        MSE = (1/n) * Σ (y_i - ŷ_i)²
    """
    # Подсказка:
    # 1. Вычисли (y_true - y_pred)²
    # 2. Найди среднее значение.
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mse = np.mean(np.power(y_true - y_pred, 2))

    return float(mse)



def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error (RMSE)

    Формула:
        RMSE = sqrt(MSE)
    """
    # Подсказка:
    # 1. Используй mean_squared_error из этого же файла.
    # 2. Возьми квадратный корень.
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return float(rmse)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Коэффициент детерминации (R²)

    Формула:
        R² = 1 - [Σ (y_i - ŷ_i)²] / [Σ (y_i - ȳ)²]
    """
    # Подсказка:
    # 1. Найди среднее y_true (ȳ).
    # 2. Вычисли числитель и знаменатель.
    # 3. Верни 1 - (числитель / знаменатель)
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    y_mean = np.mean(y_true)
    numerator = np.sum(np.power(y_true - y_pred, 2))
    denominator = np.sum(np.power(y_true - y_mean, 2))
    r_2 = 1 - numerator / denominator

    return float(r_2)


# === КЛАССИФИКАЦИЯ ===

def confusion_components(y_true, y_pred):
    """
    Вычисляет TP, TN, FP, FN для бинарной классификации.

    Параметры:
    ----------
    y_true : array-like
        Истинные метки классов (0 или 1).
    y_pred : array-like
        Предсказанные метки классов (0 или 1).

    Возвращает:
    -----------
    tp, tn, fp, fn : int
        Количество истинно положительных, истинно отрицательных,
        ложно положительных и ложно отрицательных.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return tp, tn, fp, fn


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    """
    # Подсказка:
    # 1. Сравни y_true и y_pred.
    # 2. Посчитай долю совпадений.
    
    tp, tn, fp, fn = confusion_components(y_true, y_pred)
    acc = (tp + tn) / (tp + tn + fp + fn)

    return float(acc)


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Precision = TP / (TP + FP)
    """
    # Подсказка:
    # 1. Вычисли TP и FP.
    # 2. Защити от деления на ноль.
    
    tp, _, fp, _ = confusion_components(y_true, y_pred)
    if (tp + fp) == 0:
        prec = 0
    else:
        prec = tp / (tp + fp)

    return float(prec)


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Recall = TP / (TP + FN)
    """
    # Подсказка:
    # 1. Вычисли TP и FN.
    # 2. Защити от деления на ноль.
    
    tp, _, _, fn = confusion_components(y_true, y_pred)
    if (tp + fn) == 0:
        rec = 0
    else:
        rec = tp / (tp + fn)

    return float(rec)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    # Подсказка:
    # 1. Используй precision() и recall() из этого же файла.
    # 2. Гармоническое среднее, избегай деления на 0.
    
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)

    if (prec + rec) == 0:
        f1 = 0
    else:
        f1 = 2 * (prec * rec) / (prec + rec)

    return float(f1)


def confusion_matrix(y_true, y_pred):
    """
    Confusion Matrix:
        [[TN, FP],
         [FN, TP]]
    """
    # Подсказка:
    # 1. Преобразуй всё в numpy.
    # 2. Для каждой пары (y_true[i], y_pred[i]) —
    #    увеличивай соответствующий элемент матрицы.
    
    tp, tn, fp, fn = confusion_components(y_true, y_pred)
    conf_matrix = np.array([[tn, fp], [fn, tp]])

    return conf_matrix
