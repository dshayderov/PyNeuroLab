from ml_core.models.custom.logistic_regression import LogisticRegression
from neuroml.data_loader import NeuroDataLoader
from neuroml.feature_extraction import extract_features_from_samples
import numpy as np
import pandas as pd
from ml_core.trainer import Trainer


def eeg_classification_preset(data_path: str, target_column: str, sampling_rate: float,
                            num_channels: int = 3, signal_length: int = 256):
    """
    Готовый пайплайн для классификации ЭЭГ сигналов.

    1. Загружает данные.
    2. Извлекает признаки из каждого канала.
    3. Обучает модель LogisticRegression.
    
    :param data_path: Путь к CSV с данными.
    :param target_column: Название целевого столбца.
    :param sampling_rate: Частота дискретизации.
    :param num_channels: Количество каналов в данных.
    :param signal_length: Длина сигнала для каждого канала.
    :return: Кортеж (X_features, y), где X_features - извлеченные признаки, y - целевая переменная.
    """
    print("--- Запуск пайплайна классификации ЭЭГ ---")

    # 1. Загрузка данных
    print(f"Шаг 1: Загрузка данных из {data_path}...")
    X_raw, y = NeuroDataLoader.load_eeg_from_csv(
        data_path,
        target_column=target_column,
        num_channels=num_channels,
        signal_length=signal_length
    )
    print(f"Загружены данные: X_raw.shape={X_raw.shape}, y.shape={y.shape}")


    # 2. Извлечение признаков
    print("Шаг 2: Извлечение признаков...")
    X_features = extract_features_from_samples(X_raw, sampling_rate=sampling_rate)
    print(f"Извлечено признаков: {X_features.shape[1]} для каждого семпла ({X_features.shape[0]} семплов).")

    # 3. Обучение модели
    print("Шаг 3: Обучение модели LogisticRegression...")
    model = LogisticRegression(lr=0.01, l2=0.01)
    
    # Для Trainer, как было изменено ранее, параметры обучения передаются в метод train
    trainer = Trainer(model, verbose=True) # early_stopping можно добавить из конфига
    
    # Разделение данных на обучающую и тестовую выборки (для простоты здесь не делается)
    # В реальном приложении нужно использовать sklearn.model_selection.train_test_split
    
    trainer.train(X_features, y, epochs=100, batch_size=32, shuffle=True)
    
    print("Шаг 4: Оценка модели...")
    y_pred = model.predict(X_features)
    evaluation_metrics = model.evaluate(X_features, y)
    print(f"Метрики модели на обучающей выборке: {evaluation_metrics}")

    print("--- Пайплайн завершен ---")
    
    return X_features, y

def run_eeg_demo():
    """
    Запускает демонстрацию пайплайна классификации ЭЭГ с использованием
    синтетических данных.
    """
    data_path = 'datasets/eeg_sample.csv'
    target_column = 'label'
    sampling_rate = 128 # Должно соответствовать частоте дискретизации, использованной при генерации данных
    num_channels = 3 # Количество каналов, как в синтетическом CSV
    signal_length = 256 # Длина сигнала, как в синтетическом CSV

    # Запускаем пайплайн
    X_features, y = eeg_classification_preset(data_path, target_column, sampling_rate, num_channels, signal_length)

    print("\nДемонстрация работы 'eeg_classification_preset' завершена.")
    print(f"Размерность извлеченных признаков X: {X_features.shape}")
    print(f"Размерность целевой переменной y: {y.shape}")

if __name__ == '__main__':
    run_eeg_demo()
