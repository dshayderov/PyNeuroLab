import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def plot_interactive_signal(t, signal, title="Interactive Signal", template="plotly_white"):
    """
    Интерактивное отображение временного сигнала.

    Аргументы:
        t: массив времени
        signal: массив амплитуд
        title: заголовок графика
        template: стиль оформления (plotly_white, plotly_dark и др.)

    Возвращает:
        fig: объект plotly.graph_objects.Figure
    """
    fig = px.line(
        x=t,
        y=signal,
        labels={"x": "Time (s)", "y": "Amplitude"},
        title=title,
        template=template
    )
    fig.update_traces(line=dict(width=2))
    return fig


def plot_interactive_spectrum(freqs, amplitudes, title="Amplitude Spectrum", template="plotly_dark"):
    """
    Интерактивный амплитудный спектр сигнала.

    Аргументы:
        freqs: массив частот
        amplitudes: массив амплитуд
        title: заголовок
        template: стиль оформления

    Возвращает:
        fig: объект Figure
    """
    fig = px.area(
        x=freqs,
        y=amplitudes,
        labels={"x": "Frequency (Hz)", "y": "Amplitude"},
        title=title,
        template=template
    )
    fig.update_traces(line=dict(width=2, color="cyan"))
    fig.update_layout(yaxis=dict(rangemode="tozero"))
    return fig


def plot_3d_features(df: pd.DataFrame, x: str, y: str, z: str, color: str = None, title="3D Feature Plot"):
    """
    Интерактивный 3D scatter plot признаков.

    Аргументы:
        df: DataFrame с данными
        x, y, z: имена столбцов для осей
        color: столбец для окраски точек
        title: заголовок

    Возвращает:
        fig: объект Figure
    """
    fig = px.scatter_3d(
        df,
        x=x, y=y, z=z,
        color=color,
        title=title,
        labels={x: x, y: y, z: z}
    )
    fig.update_traces(marker=dict(size=5, opacity=0.8))
    return fig


def plot_interactive_learning_curve(train_scores, val_scores, title="Learning Curve", template="plotly_white"):
    """
    Интерактивная кривая обучения модели.

    Аргументы:
        train_scores: список или массив значений метрики для обучающей выборки
        val_scores: список или массив значений метрики для валидационной выборки
        title: заголовок графика
        template: стиль оформления

    Возвращает:
        fig: объект Figure
    """
    epochs = list(range(1, len(train_scores) + 1))
    df = pd.DataFrame({
        "Epoch": epochs,
        "Train": train_scores,
        "Validation": val_scores
    })

    fig = px.line(
        df,
        x="Epoch",
        y=["Train", "Validation"],
        title=title,
        labels={"value": "Score", "Epoch": "Epoch", "variable": "Dataset"},
        template=template
    )
    fig.update_traces(mode="lines+markers")
    return fig


def plot_heatmap_interactive(df: pd.DataFrame, title="Interactive Heatmap", template="plotly_white"):
    """
    Интерактивная тепловая карта DataFrame (например, корреляций).

    Аргументы:
        df: DataFrame
        title: заголовок
        template: стиль оформления

    Возвращает:
        fig: объект Figure
    """
    fig = px.imshow(
        df,
        color_continuous_scale="RdBu_r",
        title=title,
        template=template
    )
    fig.update_layout(xaxis_title="", yaxis_title="")
    return fig


def demo():
    """
    Демонстрация работы функций модуля.
    """
    # Пример 1: сигнал
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 10 * t)
    plot_interactive_signal(t, signal).show()

    # Пример 2: спектр
    freqs = np.linspace(0, 100, 500)
    amps = np.abs(np.sin(freqs / 10))
    plot_interactive_spectrum(freqs, amps).show()

    # Пример 3: 3D scatter
    df = px.data.iris()
    plot_3d_features(df, "sepal_length", "sepal_width", "petal_length", color="species").show()

    # Пример 4: кривая обучения
    train = np.random.rand(10)
    val = np.random.rand(10)
    plot_interactive_learning_curve(train, val).show()

    # Пример 5: интерактивная тепловая карта
    corr = df.corr(numeric_only=True)
    plot_heatmap_interactive(corr).show()


if __name__ == "__main__":
    demo()
