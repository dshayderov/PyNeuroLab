"""
visualizer_demo.py — демонстрация возможностей пакета plot_utils.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from pyn_utils.plot_utils import signal_plots, export_utils


def demo_signal_plots():
    """Демонстрация работы функций из signal_plots."""
    print("=== Демонстрация signal_plots ===")

    # Генерация простого синусоидального сигнала
    t, s1 = signal_plots.generate_signal(freq=2, duration=2, sampling_rate=200)
    _, s2 = signal_plots.generate_signal(freq=4, duration=2, sampling_rate=200)

    # Отображение одного сигнала
    signal_plots.plot_signal(t, s1, title="Синусоидальный сигнал (2 Гц)")

    # Сравнение двух сигналов
    signal_plots.compare_signals(t, s1, s2, labels=("2 Гц", "4 Гц"))

    # Амплитудный спектр
    freqs = np.linspace(0, 10, len(s1))
    amplitudes = np.abs(np.fft.fft(s1))[:len(freqs)]
    signal_plots.plot_amplitude_spectrum(freqs, amplitudes)

    # Частотные диапазоны
    bands = {"low": (0, 3), "mid": (3, 6), "high": (6, 10)}
    signal_plots.plot_frequency_bands(freqs, amplitudes, bands=bands)

    # Доминирующие частоты
    signal_plots.plot_dominant_frequencies(freqs, amplitudes, n=3)


def demo_export_utils():
    """Демонстрация сохранения графиков и отчётов."""
    print("=== Демонстрация export_utils ===")

    # Создание каталога экспорта
    export_dir = export_utils.create_export_dir()

    # Создание matplotlib-графика
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y)
    ax.set_title("Matplotlib-график")
    export_utils.save_matplotlib_plot(fig, "matplotlib_demo", export_dir=export_dir)
    plt.close(fig)

    # Создание plotly-графика
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x, y=np.cos(x), mode="lines", name="cos(x)"))
    fig2.update_layout(title="Plotly-график")
    export_utils.save_plotly_figure(fig2, "plotly_demo", export_dir=export_dir)

    # Экспорт отчёта
    summary = {
        "author": "PyNeuroLab Demo",
        "modules_tested": ["signal_plots", "export_utils"],
        "timestamp": export_utils.get_timestamp(),
    }
    export_utils.export_summary_report(summary, filename="report", export_dir=export_dir)

    print(f"Все файлы сохранены в: {export_dir}")


if __name__ == "__main__":
    demo_signal_plots()
    demo_export_utils()
