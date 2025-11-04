import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Отключаем GUI-режим для тестов
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.append('./')
import pyn_utils.plot_utils.model_plots as mp


@pytest.fixture
def classification_data():
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_classes=2,
        random_state=42
    )
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    y_score = model.predict_proba(X)[:, 1]
    return X, y, y_pred, y_score, model


def test_plot_confusion_matrix(classification_data):
    _, y, y_pred, _, _ = classification_data
    mp.plot_confusion_matrix(y, y_pred)
    plt.close("all")


def test_plot_roc_curve(classification_data):
    _, y, _, y_score, _ = classification_data
    mp.plot_roc_curve(y, y_score)
    plt.close("all")


def test_plot_precision_recall_curve(classification_data):
    _, y, _, y_score, _ = classification_data
    mp.plot_precision_recall_curve(y, y_score)
    plt.close("all")


def test_plot_learning_curve():
    epochs = 10
    train_scores = np.linspace(0.6, 0.95, epochs)
    val_scores = np.linspace(0.55, 0.9, epochs)
    mp.plot_learning_curve(train_scores, val_scores, epochs)
    plt.close("all")


def test_plot_feature_importance(classification_data):
    X, _, _, _, model = classification_data
    features = [f"f{i}" for i in range(X.shape[1])]
    importances = model.feature_importances_
    mp.plot_feature_importance(features, importances)
    plt.close("all")
