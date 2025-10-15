import numpy as np


# === Этап 1, Тема 1: Структуры данных ===
def unique_elements(data):
    """Возвращает список уникальных элементов исходного списка"""
    
    seen = set()
    uniq = []
    for i in data:
        if i not in seen:
            uniq.append(i)
            seen.add(i)
    return uniq


def merge_dicts(dict1, dict2):
    """Объединяет два словаря в один"""
    
    combined = dict1.copy()
    combined.update(dict2)

    return combined


def count_occurrences(data):
    """Возвращает словарь: элемент -> количество вхождений"""
    
    return {x: data.count(x) for x in data}


# === Этап 1, Тема 2: Функции в Python ===

def normalize(data, *, scale=1):
    """
    Нормализует список чисел к диапазону [0, 1] и масштабирует на scale.

    Пример:
        normalize([1, 2, 3], scale=10) → [0, 5, 10]
    """

    min_val = min(data)
    max_val = max(data)
    
    if min_val == max_val:
        return [0] * len(data)
    else:
        return [(x - min_val) / (max_val - min_val) * scale for x in data]


def filter_data(data, func):
    """
    Фильтрует список с помощью переданной функции.

    Пример:
        filter_data([1, 2, 3, 4], lambda x: x % 2 == 0) → [2, 4]
    """
    
    return [x for x in data if func(x)]


def combine_results(*results, **weights):
    """
    Комбинирует несколько списков чисел одинаковой длины с весовыми коэффициентами.

    Пример:
        combine_results([1, 2, 3], [2, 4, 6], w1=0.3, w2=0.7)
        → [1.7, 3.4, 5.1]
    """
    
    lengths = [len(sublist) for sublist in results]

    if all(length == lengths[0] for length in lengths):
        tuples = list(zip(*results))
        weis = list(weights.values())
        n = len(results)

        if n == len(weights):
            res = [sum([x * weis[ind] for ind, x in enumerate(val)]) for val in tuples]
        else:
            res = [sum(val) / n for val in tuples]

        return res
    
    else:
        print("Ошибка: списки должны быть одинаковой длины")


def summary(data, *, round_digits=2):
    """
    Возвращает словарь с минимумом, максимумом и средним значением списка.

    Пример:
        summary([1, 2, 3]) → {'min': 1, 'max': 3, 'mean': 2.0}
    """
    
    return {
        'min': round(min(data), round_digits),
        'max': round(max(data), round_digits),
        'mean': round(sum(data) / len(data), round_digits)
    }

