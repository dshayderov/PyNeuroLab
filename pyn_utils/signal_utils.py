import numpy as np
from pyn_utils import normalize, summary, combine_results


class SignalProcessor:
    """Класс для обработки и анализа сигналов"""

    def __init__(self, signal):
        """ Инициализация сигнала с проверкой типа"""
        
        if not isinstance(signal, (list, np.ndarray)):
            raise TypeError("Сигнал должен быть списком или массивом NumPy")
        if len(signal) == 0:
            raise ValueError("Сигнал не может быть пустым")
        self.signal = signal

    def normalize(self, scale=1):
        """Нормализация сигнала"""
        
        return normalize(self.signal, scale=scale)

    def moving_average(self, window=3):
        """Сглаживание сигнала скользящим средним"""
        
        return [sum(self.signal[i:i+window]) / window for i in range(len(self.signal) - window + 1)]

    def summary(self):
        """Минимум, максимум и среднее значения сигнала"""
        
        return summary(self.signal)
