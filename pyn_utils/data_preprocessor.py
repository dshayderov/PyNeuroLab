from pyn_utils import ArrayProcessor


class DataPreprocessor:
    def __init__(self, file_handler):
        """
        Класс для загрузки, очистки и нормализации данных.

        :param file_handler: экземпляр FileHandler (для чтения/записи файлов)
        """
        self.file_handler = file_handler
        self.df = None

    def load(self, path: str):
        """
        Загружает данные из файла (csv, excel, json, sql).
        Определяет формат по расширению и использует FileHandler.

        :param path: путь к файлу данных
        :return: self
        """

        format = self.file_handler.detect_format_from_path(path)
        if format == '.csv':
            self.df = self.file_handler.read_csv(path)
        elif format in (".xlsx", ".xls"):
            self.df = self.file_handler.read_excel(path)
        elif format == '.json':
            self.df = self.file_handler.read_json(path)
        elif format == '.sql':
            self.df = self.file_handler.read_sql(path)
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {format}")
        return self

    def clean(self, method='drop', fill_value=None, interpolate_method='linear'):
        """
        Обработка пропусков.

        :param method: способ обработки ('drop', 'fill', 'interpolate')
        :param fill_value: значение для заполнения (если method='fill')
        :param interpolate_method: метод интерполяции ('linear', 'polynomial', ...)
        :return: self
        """

        if self.df is None:
            raise ValueError("Данные не загружены")

        if method == 'drop':
            self.df = self.df.dropna()
        elif method == 'fill':
            self.df = self.df.fillna(fill_value)
        elif method == 'interpolate':
            self.df = self.df.interpolate(method=interpolate_method)
        else:
            raise ValueError(f"Неизвестный способ обработки: {method}")
        return self

    def normalize(self, method='minmax', columns=None):
        """
        Масштабирование или стандартизация числовых данных.

        :param method: 'minmax' или 'zscore'
        :param columns: список колонок для нормализации (по умолчанию — все числовые)
        :return: self
        """

        if self.df is None:
            raise ValueError("Данные не загружены")

        # Выбираем только нужные колонки
        if columns is None:
            columns = self.df.select_dtypes(include='number').columns

        scaler = ArrayProcessor()
        for column in columns:
            values = self.df[column].to_numpy()
            if method not in ("minmax", "zscore"):
                raise ValueError("Метод должен быть 'minmax' или 'zscore'")
            elif method == 'minmax':
                self.df[column] = scaler.normalize(values)
            else:
                self.df[column] = scaler.standardize(values)
        
        return self

    def export(self, output_path: str, **kwargs):
        """
        Сохраняет очищенные данные в новый файл.
        Формат определяется по расширению файла.

        :param output_path: путь для сохранения
        :param kwargs: дополнительные аргументы для FileHandler.save()
        """

        if self.df is None:
            raise ValueError("Данные отсутствуют")

        self.file_handler.save(self.df, output_path, **kwargs)
        
        return output_path
