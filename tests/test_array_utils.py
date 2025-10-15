import sys
sys.path.append('./')
import numpy as np
from pyn_utils.data_utils import ArrayProcessor


# === Тесты для ArrayProcessor ===
processor = ArrayProcessor()

# --- 1. Тест normalize ---
data = np.array([1, 2, 3, 4, 5])
norm_data = processor.normalize(data)
assert np.allclose(norm_data, [0.0, 0.25, 0.5, 0.75, 1.0]), "Ошибка в normalize() без scale"

norm_scaled = processor.normalize(data, scale=10)
assert np.allclose(norm_scaled, [0.0, 2.5, 5.0, 7.5, 10.0]), "Ошибка в normalize() со scale"

# --- 2. Тест standardize ---
data = np.array([1, 2, 3, 4, 5])
std_data = processor.standardize(data)
assert np.isclose(np.mean(std_data), 0, atol=1e-7), "Среднее стандартизированных данных должно быть 0"
assert np.isclose(np.std(std_data), 1, atol=1e-7), "Ст. отклонение стандартизированных данных должно быть 1"

# --- 3. Тест combine_arrays ---
a1 = np.array([[1, 2], [3, 4]])
a2 = np.array([[5, 6], [7, 8]])

combined_v = processor.combine_arrays([a1, a2], axis=0)
assert combined_v.shape == (4, 2), "Ошибка в combine_arrays(axis=0)"

combined_h = processor.combine_arrays([a1, a2], axis=1)
assert combined_h.shape == (2, 4), "Ошибка в combine_arrays(axis=1)"

# --- 4. Тест apply_mask ---
data = np.array([10, 20, 30, 40, 50])
mask = np.array([True, False, True, False, True])
masked = processor.apply_mask(data, mask)
assert np.array_equal(masked, np.array([10, 30, 50])), "Ошибка в apply_mask()"

# Проверка ошибки при несовпадении размеров
try:
    processor.apply_mask(np.array([1, 2, 3]), np.array([True, False]))
except ValueError:
    pass
else:
    raise AssertionError("apply_mask() должен вызывать ValueError при несовпадении размеров")

# --- 5. Тест scale_broadcast ---
data = np.array([[1, 2, 3], [4, 5, 6]])
factors = np.array([10, 100, 1000])
scaled = processor.scale_broadcast(data, factors)
expected = np.array([[10, 200, 3000], [40, 500, 6000]])
assert np.array_equal(scaled, expected), "Ошибка в scale_broadcast()"

print("✅ Все тесты пройдены успешно!")
