# Project Overview

This is `PyNeuroLab`, a Python project for data analysis, machine learning, and neural networks. It appears to be an educational project for learning and implementing these concepts from scratch.

The project is structured into two main packages:
- `ml_core`: Contains core machine learning components, including a generic `Trainer` class and implementations of ML models.
- `pyn_utils`: A collection of utilities for data handling, preprocessing, plotting, and file operations.

Key technologies used are:
- **Data Manipulation:** NumPy, Pandas
- **Visualization:** Matplotlib, Plotly, Seaborn
- **Machine Learning:** Scikit-learn (and likely custom implementations)
- **Testing:** PyTest

# Building and Running

## Installation

To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```

## Running Demos

The project contains demonstration scripts that showcase its capabilities.

To run the visualization demo:
```bash
python visualizer_demo.py
```
This will generate and display several plots and save them to the `exports/` directory.

The `main.py` file contains a simple script for testing small snippets of code.

## Running Tests

The project uses `pytest` for testing. To run the test suite, execute:
```bash
pytest
```

# Development Conventions

## Code Structure

- **`ml_core/`**: This directory is for machine learning models and related logic. Models should ideally inherit from a `base_model.py` or follow a similar API to be compatible with the `Trainer`.
- **`pyn_utils/`**: This is for general-purpose utility functions. The sub-package `plot_utils` is dedicated to visualization.
- **`tests/`**: Contains tests for the project's modules. Each new feature or utility should have corresponding tests.
- **`exports/`**: This directory is used to save generated plots, reports, and other artifacts. It is ignored by Git.

## Coding Style

- The code is written in Python and seems to follow standard PEP 8 conventions.
- Docstrings are used to document modules, classes, and functions.
- The project uses a mix of procedural and object-oriented programming.

## Workflow

1.  **Data Preprocessing:** Use `pyn_utils.data_preprocessor.DataPreprocessor` to load, clean, and normalize data.
2.  **Model Training:** Use the `ml_core.trainer.Trainer` class to train machine learning models. The trainer handles epochs, batching, validation, and checkpointing.
3.  **Visualization:** Use the functions in `pyn_utils.plot_utils` to create and export visualizations of data and model performance.
4.  **Exporting Results:** Use `pyn_utils.plot_utils.export_utils` to save plots and summary reports to the `exports` directory.

# Дальнейший план и конвенции

Цель проекта — создание полноценной интерактивной лаборатории `PyNeuroLab App` для анализа данных и работы с моделями машинного обучения, по аналогии с Matlab или Google Colab.

## Итоговая структура проекта

```
PyNeuroLab/
├── app/                  # Этап 9: Пользовательский интерфейс (Streamlit / FastAPI + Vue)
│   ├── __init__.py
│   ├── main.py             # Главный скрипт для запуска приложения
│   ├── pages/              # Директория для страниц/разделов в Streamlit
│   └── components/         # Общие UI компоненты
│
├── configs/              # Конфигурации для экспериментов, моделей и приложения
│   └── default_config.yaml
│
├── datasets/             # Наборы данных для обучения и тестов
│   └── .gitignore
│
├── experiments/          # Сохраненные эксперименты: логи, веса моделей, результаты
│   └── .gitignore
│
├── ml_core/              # Этапы 5, 6, 8: Ядро для МЛ и Нейросетей
│   ├── __init__.py
│   ├── base_model.py       # Абстрактный базовый класс для всех моделей
│   ├── trainer.py          # Класс для обучения
│   ├── metrics.py          # Функции метрик
│   ├── hub.py              # Этап 6: Для загрузки внешних моделей (Model Hub)
│   └── models/             # Этап 8: Архитектуры моделей (NeuralCore)
│       ├── custom/
│       └── external/
│
├── neuroml/              # Этап 7: Логика для анализа нейроданных
│   ├── __init__.py
│   ├── data_loader.py
│   ├── feature_extraction.py
│   └── presets.py
│
├── pyn_utils/            # Этапы 1-4: Утилиты (без изменений)
│
├── tests/                # Тесты для всех модулей
│
├── .gitignore
├── GEMINI.md
├── LICENSE
├── README.md
└── requirements.txt
```

## Правила взаимодействия
1.  **Язык**: Все мои ответы, пояснения и комментарии к коду должны быть на **русском языке**.
2.  **Стиль кода**: Весь генерируемый и изменяемый код должен строго соответствовать стандарту **PEP8**.
