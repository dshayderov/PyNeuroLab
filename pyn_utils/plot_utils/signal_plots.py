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


def plot_amplitude_spectrum(freqs, amplitudes, title="Амплитудный спектр"):
    """
    Строит амплитудный спектр сигнала.
    """

    freqs = np.asarray(freqs)
    amplitudes = np.asarray(amplitudes)

    plot_signal(freqs, amplitudes, title=title, xlabel="Частота (Гц)", ylabel="Амплитуда")


def plot_phase_spectrum(freqs, phases, title="Фазовый спектр"):
    """
    Строит фазовый спектр сигнала.
    """

    freqs = np.asarray(freqs)
    phases = np.asarray(phases)

    plot_signal(freqs, phases, title=title, xlabel="Частота (Гц)", ylabel="Фаза (рад)")


def plot_frequency_bands(freqs, amplitudes, bands=None):
    """
    Отображает амплитудный спектр с выделением частотных диапазонов.
    bands — словарь, например:
        {
            "low": (0, 200),
            "mid": (200, 2000),
            "high": (2000, 5000)
        }
    """

    freqs = np.asarray(freqs)
    amplitudes = np.asarray(amplitudes)

    plt.figure(figsize=(8, 4))
    plt.plot(freqs, amplitudes, label="Амплитудный спектр", color="black")

    if bands:
        colors = plt.cm.tab10(np.linspace(0, 1, len(bands)))
        for (name, (f_min, f_max)), color in zip(bands.items(), colors):
            mask = (freqs >= f_min) & (freqs <= f_max)
            plt.fill_between(freqs[mask], amplitudes[mask], color=color, alpha=0.3, label=name)

    plt.xlabel("Частота (Гц)")
    plt.ylabel("Амплитуда")
    plt.title("Амплитудный спектр с выделением диапазонов")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_dominant_frequencies(freqs, amplitudes, n=3):
    """
    Отмечает на графике n частот с наибольшей амплитудой.
    """

    freqs = np.asarray(freqs)
    amplitudes = np.asarray(amplitudes)

    top_indices = np.argsort(amplitudes)[-n:]
    top_freqs = freqs[top_indices]
    top_amps = amplitudes[top_indices]

    plt.figure(figsize=(8, 4))
    plt.plot(freqs, amplitudes, label="Амплитудный спектр", color="gray")
    plt.scatter(top_freqs, top_amps, color="red", zorder=5, label=f"Топ-{n} частот")

    for f, a in zip(top_freqs, top_amps):
        plt.text(f, a, f"{f:.1f} Гц", fontsize=9, ha="center", va="bottom")

    plt.xlabel("Частота (Гц)")
    plt.ylabel("Амплитуда")
    plt.title(f"Топ-{n} доминирующих частот")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_comparison_in_time_domain(t, original, reconstructed):
    """
    Сравнивает исходный и восстановленный сигнал во временной области.
    """
    
    compare_signals(t, original, reconstructed, labels=("Исходный сигнал", "Восстановленный сигнал"))
