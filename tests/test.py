import sys, os, json, time
sys.path.append('./')
from pyn_utils import read_json, write_json, normalize, Timer

# --- Тест 1: запись и чтение JSON ---
data = {"name": "Peter", "age": 32}
write_json("data_test.json", data)
assert os.path.exists("data_test.json")

read_data = read_json("data_test.json")
assert read_data == data

# --- Тест 2: проверка нормализации ---
assert normalize([1, 2, 3], scale=6) == [0.0, 3.0, 6.0]

# --- Тест 3: проверка работы Timer ---
start = time.time()
with Timer("Проверка задержки"):
    time.sleep(0.1)
elapsed = time.time() - start
assert elapsed >= 0.1  # должно сработать не мгновенно

os.remove("data_test.json")
print("✅ Все интеграционные тесты пройдены успешно!")
