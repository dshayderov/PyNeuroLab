import pandas as pd
import numpy as np
from typing import Tuple, Optional, List

class NeuroDataLoader:
    """
    Класс для загрузки и базовой подготовки нейроданных.
    """
    @staticmethod
    def load_eeg_from_csv(path: str, target_column: str,
                          num_channels: Optional[int] = None,
                          signal_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Загружает данные ЭЭГ из CSV файла и возвращает их в виде 3D NumPy массива.

        Предполагается, что каждая строка - это временной срез одного семпла,
        а столбцы содержат конкатенированные временные отсчеты всех каналов,
        за которыми следует целевая переменная.

        :param path: Путь к CSV файлу.
        :param target_column: Название столбца с метками классов.
        :param num_channels: Ожидаемое количество каналов. Если None, будет попытка определить.
        :param signal_length: Ожидаемая длина сигнала для каждого канала. Если None, будет попытка определить.
        :return: Кортеж (X, y), где X - 3D NumPy массив (samples, channels, time_points),
                 y - 1D NumPy массив целевой переменной.
        """
        try:
            data = pd.read_csv(path)
            y = data[target_column].to_numpy()
            X_df = data.drop(columns=[target_column])

            if num_channels is None or signal_length is None:
                # Попытка определить num_channels и signal_length из названий колонок
                # Предполагаем формат 'chX_tY'
                channel_cols = [col for col in X_df.columns if '_t' in col]
                if not channel_cols:
                    raise ValueError("Не удалось определить количество каналов и длину сигнала из названий колонок. "
                                     "Укажите num_channels и signal_length явно.")
                
                # Извлекаем названия каналов (например, 'ch1', 'ch2')
                unique_channels = sorted(list(set([col.split('_t')[0] for col in channel_cols])))
                if num_channels is None:
                    num_channels = len(unique_channels)
                
                # Извлекаем временные индексы (например, 't0', 't1')
                # Должны быть одинаковыми для всех каналов
                if signal_length is None:
                    # Для первого канала определяем его длину
                    first_channel_cols = [col for col in channel_cols if col.startswith(unique_channels[0] + '_t')]
                    signal_length = len(first_channel_cols)
                
                if num_channels * signal_length != len(X_df.columns):
                    raise ValueError(f"Несоответствие: (num_channels={num_channels} * signal_length={signal_length}) "
                                     f"должно быть равно количеству колонок признаков ({len(X_df.columns)}). "
                                     f"Проверьте num_channels, signal_length или формат CSV.")

            # Преобразуем в 3D массив
            X = X_df.to_numpy().reshape(X_df.shape[0], num_channels, signal_length)
            
            print(f"Данные успешно загружены: {X.shape[0]} семплов, {X.shape[1]} каналов, {X.shape[2]} временных отсчетов.")
            return X, y
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл не найден по пути: {path}")
        except KeyError as e:
            raise KeyError(f"Целевой столбец '{target_column}' не найден в файле: {e}")
        except ValueError as e:
            raise ValueError(f"Ошибка при обработке данных CSV: {e}")

