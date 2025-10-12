# === Этап 1, Тема 3: Работа с файлами и контекстными менеджерами ===

import json


def read_text(path):
    """
    Читает содержимое текстового файла и возвращает его как строку.
    Если файл не найден — возвращает пустую строку.
    """
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        content = ""

    return content


def write_text(path, text, mode="w"):
    """
    Записывает текст в файл.
    При mode="a" добавляет в конец файла.
    Возвращает количество записанных символов.
    """
    
    if mode=="a":
        with open(path, "a", encoding="utf-8") as f:
            wr_num = f.write(text)
    
    else:
        with open(path, "w", encoding="utf-8") as f:
            wr_num = f.write(text)

    return wr_num


def read_json(path):
    """
    Считывает JSON-файл и возвращает Python-объект.
    Если файл не существует или повреждён — возвращает пустой словарь {}.
    """
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    return data


def write_json(path, data):
    """
    Сохраняет объект data в JSON-файл с отступами.
    """
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    return True

