import sys
sys.path.append('./')
import numpy as np
from pyn_utils import SignalProcessor


# === Тест 1. Инициализация ===
signal = [1, 2, 3, 4, 5]
processor = SignalProcessor(signal)
assert isinstance(processor.signal, list) or isinstance(processor.signal, np.ndarray)


# === Тест 2. Ошибка при пустом сигнале ===
try:
    SignalProcessor([])
except ValueError as e:
    assert "не может быть пустым" in str(e)


# === Тест 3. Ошибка при неверном типе ===
try:
    SignalProcessor("12345")
except TypeError as e:
    assert "должен быть списком" in str(e)


# === Тест 4. Нормализация ===
norm_signal = processor.normalize(scale=2)
assert np.isclose(max(norm_signal), 2.0, atol=1e-6)


# === Тест 5. Скользящее среднее ===
smoothed = processor.moving_average(window=3)
expected = np.convolve(signal, np.ones(3)/3, mode='valid')
assert np.allclose(smoothed, expected)


# === Тест 6. Проверка summary ===
summary = processor.summary()
assert isinstance(summary, dict)
assert set(summary.keys()) == {"min", "max", "mean"}
assert summary["min"] == 1
assert summary["max"] == 5
assert np.isclose(summary["mean"], 3.0)


print("✅ Все тесты пройдены успешно!")
