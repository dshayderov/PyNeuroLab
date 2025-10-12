import sys
sys.path.append('./')
from pyn_utils import read_text, write_text, read_json, write_json, Timer
import os
import json
import time


# === Этап 1, Тема 3 ===
# --- Тесты для read_text ---
test_file = "test_read.txt"
with open(test_file, "w", encoding="utf-8") as f:
    f.write("Hello, PyNeuroLab!")

assert read_text(test_file) == "Hello, PyNeuroLab!"
os.remove(test_file)

# Проверка: если файл отсутствует
assert read_text("no_such_file.txt") == ""


# --- Тесты для write_text ---
test_file = "test_write.txt"
written = write_text(test_file, "abc", mode="w")
assert written == 3
assert read_text(test_file) == "abc"

written = write_text(test_file, "XYZ", mode="a")
assert written == 3
assert read_text(test_file) == "abcXYZ"
os.remove(test_file)


# --- Тесты для read_json и write_json ---
test_json = "test_data.json"
data = {"a": 1, "b": 2}
write_json(test_json, data)

read_back = read_json(test_json)
assert read_back == data

# Проверка на несуществующий файл
assert read_json("missing.json") == {}

# Проверка на повреждённый JSON
with open(test_json, "w", encoding="utf-8") as f:
    f.write("{ broken json ")
assert read_json(test_json) == {}
os.remove(test_json)


# --- Тесты для Timer ---
start = time.time()
with Timer("Sleep test"):
    time.sleep(0.1)
end = time.time()
assert end - start >= 0.1

print("✅ Все тесты по теме 3 пройдены успешно!")
