"""
Пакет PyNeuroLab — основной пакет проекта.

Здесь можно описать общие настройки, инициализацию логов,
импорт ключевых функций и классов для упрощённого доступа из других модулей.
"""

# --- 1. Импорт нужных модулей пакета (при необходимости) ---
from .file_utils import read_text, write_text, read_json, write_json
from .data_utils import normalize, filter_data, combine_results, summary, unique_elements, merge_dicts, count_occurrences
from .timing import Timer
from .signal_utils import SignalProcessor
from .plot_utils import generate_signal, plot_signal, compare_signals


# --- 2. Определение списка доступных при импорте элементов ---
# Если указать __all__, то при выполнении "from pyneurolab import *"
# будут импортированы только эти имена.
__all__ = ['read_text', 'write_text', 'read_json', 'write_json', 
           'normalize', 'filter_data', 'combine_results', 'summary', 
           'unique_elements', 'merge_dicts', 'count_occurrences',
           'Timer', 'SignalProcessor',
           'generate_signal', 'plot_signal', 'compare_signals']