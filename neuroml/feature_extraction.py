import numpy as np
from pyn_utils.signal_utils import SignalProcessor

def extract_signal_features(signal: np.ndarray, sampling_rate: float) -> np.ndarray:
    """
    Извлекает полный вектор признаков из одного сигнала (канала).

    Использует SignalProcessor для извлечения временных, частотных
    и мощностных признаков.

    :param signal: Одномерный массив NumPy с данными сигнала.
    :param sampling_rate: Частота дискретизации.
    :return: Вектор признаков NumPy.
    """
    if not isinstance(signal, np.ndarray) or signal.ndim != 1:
        raise ValueError("Сигнал должен быть одномерным массивом NumPy.")

    processor = SignalProcessor(signal, sampling_rate=sampling_rate)
    
    # feature_vector() объединяет все типы признаков
    features = processor.feature_vector()
    
    return features

def extract_features_from_samples(X_samples: np.ndarray, sampling_rate: float) -> np.ndarray:
    """
    Применяет извлечение признаков ко всем семплам в наборе данных.
    Может принимать как 2D (семплы x временные отсчеты) так и 3D (семплы x каналы x временные отсчеты) данные.

    :param X_samples: NumPy массив с данными. Может быть 2D или 3D.
    :param sampling_rate: Частота дискретизации.
    :return: 2D массив (семплы x признаки).
    """
    features_list = []
    if X_samples.ndim == 2:
        # 2D массив: каждый семпл - это один сигнал
        for sample_signal in X_samples:
            features = extract_signal_features(sample_signal, sampling_rate)
            features_list.append(features)
    elif X_samples.ndim == 3:
        # 3D массив: семплы x каналы x временные отсчеты
        for sample_multi_channel in X_samples: # Итерация по семплам
            sample_features = []
            for channel_signal in sample_multi_channel: # Итерация по каналам
                features = extract_signal_features(channel_signal, sampling_rate)
                sample_features.append(features)
            features_list.append(np.concatenate(sample_features)) # Объединяем признаки всех каналов для одного семпла
    else:
        raise ValueError(f"Неподдерживаемая размерность входных данных X_samples: {X_samples.ndim}. Ожидается 2D или 3D.")
        
    return np.array(features_list)
