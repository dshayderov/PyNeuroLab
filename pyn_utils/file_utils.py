# === Этап 1, Тема 3: Работа с файлами и контекстными менеджерами ===

import json
from typing import Optional
import pandas as pd
import os


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

class FileHandler:
    """
    Класс для чтения и записи табличных данных в форматах CSV, Excel и JSON.
    Поддерживаемые форматы: .csv, .xlsx, .xls, .json
    """

    def __init__(self, default_encoding: str = "utf-8"):
        """
        Инициализация хэндлера.

        Параметры:
            default_encoding (str): кодировка по умолчанию для чтения и записи.
        """
        self.default_encoding = default_encoding

    # ------------------------------------------------------------------ #
    # Чтение
    # ------------------------------------------------------------------ #

    def read_csv(
        self,
        path: str,
        sep: str = ",",
        encoding: Optional[str] = None,
        usecols: Optional[list] = None,
        nrows: Optional[int] = None,
        dtype: Optional[dict] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Читает CSV-файл и возвращает DataFrame.

        Обработка:
        - FileNotFoundError
        - PermissionError
        - EmptyDataError
        """
        if encoding is None:
            encoding = self.default_encoding

        try:
            return pd.read_csv(
                path,
                sep=sep,
                encoding=encoding,
                usecols=usecols,
                nrows=nrows,
                dtype=dtype,
                **kwargs
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл '{path}' не найден")
        except PermissionError:
            raise PermissionError(f"Нет доступа к файлу '{path}'")
        except pd.errors.EmptyDataError:
            raise pd.errors.EmptyDataError(f"Файл '{path}' пуст")
        except ValueError as e:
            raise ValueError(f"Ошибка при чтении CSV: {e}")

    def read_excel(
        self,
        path: str,
        sheet_name: Optional[str] = None,
        engine: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Читает Excel-файл (.xlsx, .xls) и возвращает DataFrame.
        Если sheet_name не указан — читается первый лист.
        """
        try:
            if sheet_name is None:
                sheet_name = 0
            return pd.read_excel(path, sheet_name=sheet_name, engine=engine, **kwargs)
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл '{path}' не найден")
        except PermissionError:
            raise PermissionError(f"Нет доступа к файлу '{path}'")
        except pd.errors.EmptyDataError:
            raise pd.errors.EmptyDataError(f"Файл '{path}' пуст")
        except ValueError as e:
            raise ValueError(f"Ошибка при чтении Excel: {e}")

    def read_json(
        self,
        path: str,
        orient: Optional[str] = "records",
        lines: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        """
        Читает JSON-файл и возвращает DataFrame.

        Поддерживает:
        - разные ориентации (orient);
        - построчный JSON (lines=True);
        - обработку ошибок парсинга JSON.
        """
        try:
            return pd.read_json(path, orient=orient, lines=lines, **kwargs)
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл '{path}' не найден")
        except json.JSONDecodeError as e:
            raise ValueError(f"Ошибка парсинга JSON: {e}")
        except PermissionError:
            raise PermissionError(f"Нет доступа к файлу '{path}'")
        except pd.errors.EmptyDataError:
            raise pd.errors.EmptyDataError(f"Файл '{path}' пуст")
        except ValueError as e:
            raise ValueError(f"Ошибка при чтении JSON: {e}")

    # ------------------------------------------------------------------ #
    # Запись / Сохранение
    # ------------------------------------------------------------------ #

    def save(
        self,
        df: pd.DataFrame,
        path: str,
        format: Optional[str] = None,
        index: bool = False,
        encoding: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Сохраняет DataFrame в указанный путь.

        Формат:
            - определяется из расширения, если format=None
            - поддерживаются csv, xlsx, xls, json

        Возвращает:
            True — при успешном сохранении, иначе возбуждает исключение.
        """
        if format is None:
            ext = self.detect_format_from_path(path)
            if ext is None:
                raise ValueError("Не удалось определить формат по расширению")
            format = ext.lstrip(".")
        else:
            format = format.lower().lstrip(".")

        if encoding is None:
            encoding = self.default_encoding

        try:
            if format == "csv":
                df.to_csv(path, index=index, encoding=encoding, **kwargs)
            elif format in ("xlsx", "xls"):
                df.to_excel(path, index=index, **kwargs)
            elif format == "json":
                df.to_json(path, orient="records", force_ascii=False, **kwargs)
            else:
                raise ValueError(f"Неподдерживаемый формат файла: '{format}'")

            return True
        except Exception as e:
            raise ValueError(f"Ошибка при сохранении файла '{path}': {e}")

    # ------------------------------------------------------------------ #
    # Утилиты
    # ------------------------------------------------------------------ #

    def detect_format_from_path(self, path: str) -> Optional[str]:
        """
        Определяет формат файла по расширению.
        Возвращает расширение в нижнем регистре без точки или None.
        """
        extension = os.path.splitext(path)[1].lower()
        if extension in (".csv", ".xlsx", ".xls", ".json", '.sql'):
            return extension
        return None

    def read_sql(self, sql: str, con, **kwargs) -> pd.DataFrame:
        """
        Читает данные из базы данных через SQL-запрос.

        Параметры:
            sql (str): SQL-запрос.
            con: SQLAlchemy engine или соединение.

        Возвращает DataFrame.
        """
        if con is None:
            raise ValueError("Необходимо указать соединение с базой данных (con)")

        try:
            return pd.read_sql(sql, con, **kwargs)
        except Exception as e:
            raise ValueError(f"Ошибка при выполнении SQL-запроса: {e}")