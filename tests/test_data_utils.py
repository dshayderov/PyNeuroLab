import sys
sys.path.append('./')
from pyn_utils import (
    unique_elements, merge_dicts, count_occurrences,  # 1-е задание
    normalize, filter_data, combine_results, summary   # 2-е задание
)

# --- Тесты для unique_elements ---
assert unique_elements([1, 2, 2, 3, 3, 3]) == [1, 2, 3]
assert unique_elements(["a", "a", "b"]) == ["a", "b"]

# --- Тесты для merge_dicts ---
assert merge_dicts({"x": 1, "y": 2}, {"y": 10, "z": 3}) == {"x": 1, "y": 10, "z": 3}

# --- Тесты для count_occurrences ---
assert count_occurrences([1, 2, 2, 3, 3, 3]) == {1: 1, 2: 2, 3: 3}
assert count_occurrences(["a", "b", "b"]) == {"a": 1, "b": 2}


# === Этап 1, Тема 2 ===
# --- Тесты для normalize ---
data = [1, 2, 3]
assert normalize(data) == [0.0, 0.5, 1.0]
assert normalize(data, scale=10) == [0.0, 5.0, 10.0]

# --- Тесты для filter_data ---
assert filter_data([1, 2, 3, 4], lambda x: x % 2 == 0) == [2, 4]
assert filter_data(["apple", "banana", "pear"], lambda s: "a" in s) == ["apple", "banana", "pear"]

# --- Тесты для combine_results ---
r1 = [1, 2, 3]
r2 = [2, 4, 6]
assert combine_results(r1, r2, w1=0.3, w2=0.7) == [1.7, 3.4, 5.1]
assert combine_results(r1, r2) == [1.5, 3.0, 4.5]  # без весов

# --- Тесты для summary ---
s = summary([1, 2, 3])
assert s == {"min": 1, "max": 3, "mean": 2.0}
s = summary([2.345, 2.678, 2.999], round_digits=3)
assert s == {"min": 2.345, "max": 2.999, "mean": 2.674}

print("✅ Все тесты пройдены успешно!")
