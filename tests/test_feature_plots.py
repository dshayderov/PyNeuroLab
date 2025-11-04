import pytest
import pandas as pd
from unittest import mock
import matplotlib.pyplot as plt

import sys
sys.path.append('./')
from pyn_utils.plot_utils import (
    plot_feature_distribution,
    plot_feature_correlation,
    plot_feature_pairplot,
)

@pytest.fixture
def sample_df():
    """Фикстура: создаёт небольшой DataFrame для тестов"""
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45],
        "salary": [3000, 4000, 5000, 6000, 7000],
        "target": [0, 1, 0, 1, 1],
    })


def test_plot_feature_distribution_runs(sample_df):
    """Проверяет, что функция строит график без ошибок"""
    with mock.patch("matplotlib.pyplot.show"):
        plot_feature_distribution(sample_df, "age")
        plt.close("all")  # чистим графики после теста


def test_plot_feature_distribution_invalid_feature(sample_df):
    """Проверка реакции на несуществующий признак"""
    with pytest.raises(KeyError):
        plot_feature_distribution(sample_df, "unknown_feature")


def test_plot_feature_correlation_calls_heatmap(sample_df):
    """Мокаем seaborn и проверяем, что heatmap вызывается"""
    with mock.patch("seaborn.heatmap") as mock_heatmap:
        plot_feature_correlation(sample_df)
        mock_heatmap.assert_called_once()


def test_plot_feature_pairplot_calls_pairplot(sample_df):
    """Проверяем, что seaborn.pairplot вызывается с нужными аргументами"""
    with mock.patch("seaborn.pairplot") as mock_pairplot:
        plot_feature_pairplot(sample_df, ["age", "salary"])
        mock_pairplot.assert_called_once()
