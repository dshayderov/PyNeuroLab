import sys
sys.path.append('./')
import os
import io
import sqlite3
import json
import pandas as pd
from pyn_utils.file_utils import FileHandler


def test_filehandler():
    # === Подготовка тестовой среды ===
    fh = FileHandler()
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "city": ["London", "Berlin", "Paris"]
    })

    test_dir = "tests/temp"
    os.makedirs(test_dir, exist_ok=True)

    csv_path = os.path.join(test_dir, "test_data.csv")
    json_path = os.path.join(test_dir, "test_data.json")
    excel_path = os.path.join(test_dir, "test_data.xlsx")
    sql_path = os.path.join(test_dir, "test.db")

    # === Тест detect_format_from_path ===
    assert fh.detect_format_from_path(csv_path) == ".csv"
    assert fh.detect_format_from_path("data.xlsx") == ".xlsx"
    assert fh.detect_format_from_path("data.txt") is None

    # === Тест save ===
    assert fh.save(df, csv_path) is True
    assert os.path.exists(csv_path)

    assert fh.save(df, json_path) is True
    assert os.path.exists(json_path)

    assert fh.save(df, excel_path) is True
    assert os.path.exists(excel_path)

    # Проверка ошибок: некорректный формат
    try:
        fh.save(df, os.path.join(test_dir, "badfile.txt"))
    except ValueError as e:
        assert "Не удалось определить формат" in str(e)

    # === Тест read_csv ===
    df_csv = fh.read_csv(csv_path)
    assert isinstance(df_csv, pd.DataFrame)
    assert len(df_csv) == 3
    assert list(df_csv.columns) == list(df.columns)

    # === Тест read_json ===
    df_json = fh.read_json(json_path)
    assert isinstance(df_json, pd.DataFrame)
    assert set(df_json.columns) == set(df.columns)

    # Проверка ошибки JSON
    bad_json_path = os.path.join(test_dir, "bad.json")
    with open(bad_json_path, "w", encoding="utf-8") as f:
        f.write("{invalid_json: true,}")  # специально повреждённый JSON

    try:
        fh.read_json(bad_json_path)
    except ValueError as e:
        assert "Ошибка при чтении JSON" in str(e)


    # === Тест read_excel ===
    df_excel = fh.read_excel(excel_path)
    assert isinstance(df_excel, pd.DataFrame)
    assert len(df_excel) == 3

    # === Тест read_sql ===
    con = sqlite3.connect(sql_path)
    df.to_sql("people", con, index=False, if_exists="replace")
    df_sql = fh.read_sql("SELECT * FROM people", con)
    con.close()
    assert isinstance(df_sql, pd.DataFrame)
    assert len(df_sql) == 3
    assert set(df_sql.columns) == set(df.columns)

    # === Тест read_csv — ошибка при отсутствии файла ===
    try:
        fh.read_csv(os.path.join(test_dir, "no_such_file.csv"))
    except FileNotFoundError:
        pass
    else:
        assert False, "FileNotFoundError не был вызван"

    # === Тест read_excel — ошибка при неправильном пути ===
    try:
        fh.read_excel(os.path.join(test_dir, "no_such_file.xlsx"))
    except FileNotFoundError:
        pass
    else:
        assert False, "FileNotFoundError не был вызван"

    # === Тест read_sql — отсутствие соединения ===
    try:
        fh.read_sql("SELECT * FROM people", None)
    except ValueError as e:
        assert "Необходимо указать соединение" in str(e)

    # === Очистка временных файлов ===
    for f in (csv_path, json_path, excel_path, bad_json_path, sql_path):
        if os.path.exists(f):
            os.remove(f)
    os.rmdir(test_dir)

    print("✅ Все тесты пройдены успешно!")


if __name__ == "__main__":
    test_filehandler()
