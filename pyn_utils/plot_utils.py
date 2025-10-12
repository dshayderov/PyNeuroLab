import numpy as np
import matplotlib.pyplot as plt


def generate_signal(freq=1, duration=1, sampling_rate=100):
    """
    Генерирует синусоидальный сигнал.
    freq — частота (Гц)
    duration — длительность (сек)
    sampling_rate — частота дискретизации
    Возвращает: (t, signal)
    """
    
    t = np.linspace(0, duration, sampling_rate)
    signal = np.sin(2 * np.pi * freq * t)

    return t, signal


def plot_signal(t, signal, title="Сигнал", xlabel="Время (с)", ylabel="Амплитуда"):
    """
    Строит график сигнала.
    """
    
    x_data = np.asarray(t)
    y_data = np.asarray(signal)

    plt.plot(x_data, y_data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


def compare_signals(t, s1, s2, labels=("Сигнал 1", "Сигнал 2")):
    """
    Отображает два сигнала на одном графике для сравнения.
    """
    
    x_data = np.asarray(t)
    y1_data = np.asarray(s1)
    y2_data = np.asarray(s2)

    if len(y1_data) == len(x_data) and len(y2_data) == len(x_data):
        plt.plot(x_data, y1_data, 'r', label=labels[0])
        plt.plot(x_data, y2_data, 'g', label=labels[1])
        plt.xlabel("Время (с)")
        plt.ylabel("Амплитуда")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    else:
        raise ValueError("Длины сигналов не совпадают с вектором времени")
