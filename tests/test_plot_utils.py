import sys
sys.path.append('./')
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Без вывода окон
from pyn_utils.plot_utils import generate_signal, plot_signal, compare_signals

# --- Тест 1: генерация сигнала ---
t, s = generate_signal(freq=2, duration=1, sampling_rate=100)
assert isinstance(t, np.ndarray)
assert isinstance(s, np.ndarray)
assert len(t) == len(s) == 100
assert np.allclose(s[:5], np.sin(2 * np.pi * 2 * t[:5]))

# --- Тест 2: проверка амплитуды сигнала ---
assert np.isclose(np.max(s), 1, atol=1e-2)
assert np.isclose(np.min(s), -1, atol=1e-2)

# --- Тест 3: проверка некорректных данных в compare_signals ---
try:
    compare_signals(t[:50], s, s)
except ValueError as e:
    assert str(e) == "Длины сигналов не совпадают с вектором времени"
else:
    raise AssertionError("Ожидалось исключение ValueError при разных длинах сигналов")

# --- Тест 4: базовая проверка plot_signal ---
# Здесь проверяем, что функция выполняется без ошибок
plot_signal(t, s, title="Тестовый сигнал")

# --- Тест 5: базовая проверка compare_signals ---
compare_signals(t, s, s * 0.5, labels=("Оригинал", "Уменьшенный"))

print("✅ Все тесты пройдены успешно!")
