import streamlit as st
import os
import sys
import numpy as np # Добавляем импорт numpy

# Добавляем корневую директорию проекта в sys.path для корректных импортов
# Это необходимо, если app/main.py запускается напрямую
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Импорты основных компонентов проекта
from ml_core.trainer import Trainer
from ml_core import ModelHub, LinearRegression, LogisticRegression
from neuroml.data_loader import NeuroDataLoader
from neuroml.feature_extraction import extract_signal_features, extract_features_from_samples
from neuroml.presets import eeg_classification_preset, run_eeg_demo # Предполагаем, что проблема с этим файлом решена
from pyn_utils.file_utils import FileHandler
# from configs.default_config import * # В будущем будем использовать библиотеку для парсинга YAML


st.set_page_config(
    page_title="PyNeuroLab App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Боковая панель ---
st.sidebar.title("PyNeuroLab Навигация")
st.sidebar.markdown("---")
page_selection = st.sidebar.radio(
    "Выберите раздел:",
    ["Главная", "Загрузка и Предварительная Обработка", "Извлечение Признаков", "Моделирование и Обучение", "Model Hub", "Эксперименты"]
)
st.sidebar.markdown("---")
st.sidebar.info("Разработано с ❤️ для анализа нейроданных")

# --- Основное содержимое страницы ---
st.title("🧠 PyNeuroLab App")
st.markdown("Добро пожаловать в интерактивную лабораторию для анализа нейроданных, машинного обучения и нейросетей.")

if page_selection == "Главная":
    st.header("Обзор")
    st.write("PyNeuroLab — это ваша персональная платформа для:")
    st.markdown("- **Загрузки и подготовки данных**")
    st.markdown("- **Извлечения и визуализации признаков сигнала**")
    st.markdown("- **Создания, обучения и сравнения моделей машинного обучения и нейросетей**")
    st.markdown("- **Управления экспериментами и Model Hub**")
    st.write("Используйте навигацию в боковой панели, чтобы начать.")

elif page_selection == "Загрузка и Предварительная Обработка":
    st.header("Загрузка и Предварительная Обработка Данных")
    st.write("Здесь вы сможете загрузить свои нейроданные (например, ЭЭГ из CSV), очистить их и выполнить базовую предобработку.")

    uploaded_file = st.file_uploader("Загрузите CSV файл с ЭЭГ данными", type="csv")

    if uploaded_file is not None:
        # Сохраняем файл временно, чтобы NeuroDataLoader мог его прочитать
        # Streamlit предоставляет FileIO, который можно прочитать напрямую
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name
        
        st.subheader("Параметры данных")
        col1, col2, col3 = st.columns(3)
        with col1:
            target_column = st.text_input("Имя целевой колонки (label):", value="label")
        with col2:
            num_channels = st.number_input("Количество каналов:", min_value=1, value=3)
        with col3:
            signal_length = st.number_input("Длина сигнала на канал:", min_value=1, value=256)

        load_button = st.button("Загрузить и Обработать Данные")

        if load_button:
            try:
                X, y = NeuroDataLoader.load_eeg_from_csv(
                    temp_file_path,
                    target_column=target_column,
                    num_channels=num_channels,
                    signal_length=signal_length
                )
                st.session_state['X_raw'] = X
                st.session_state['y_raw'] = y
                st.success("Данные успешно загружены!")
                st.write(f"Загруженные признаки (X) имеют форму: {X.shape}")
                st.write(f"Загруженная целевая переменная (y) имеет форму: {y.shape}")
            except Exception as e:
                st.error(f"Ошибка при загрузке или обработке данных: {e}")
            finally:
                os.remove(temp_file_path) # Удаляем временный файл

    else:
        st.info("Пожалуйста, загрузите CSV файл, чтобы начать.")
        # Для демонстрации, можем использовать наш eeg_sample.csv, если он существует
        if os.path.exists("datasets/eeg_sample.csv"):
            st.markdown("---")
            st.subheader("Использование демонстрационных данных")
            if st.button("Загрузить демо-данные (datasets/eeg_sample.csv)"):
                data_path = "datasets/eeg_sample.csv"
                target_column = "label"
                num_channels = 3
                signal_length = 256
                try:
                    X, y = NeuroDataLoader.load_eeg_from_csv(
                        data_path,
                        target_column=target_column,
                        num_channels=num_channels,
                        signal_length=signal_length
                    )
                    st.session_state['X_raw'] = X
                    st.session_state['y_raw'] = y
                    st.success("Демо-данные успешно загружены!")
                    st.write(f"Загруженные признаки (X) имеют форму: {X.shape}")
                    st.write(f"Загруженная целевая переменная (y) имеет форму: {y.shape}")
                except Exception as e:
                    st.error(f"Ошибка при загрузке демо-данных: {e}")

elif page_selection == "Извлечение Признаков":
    st.header("Извлечение и Визуализация Признаков")
    st.write("Преобразуйте сырые сигналы в значимые признаки для ваших моделей. Визуализируйте временные и частотные характеристики.")
    if 'X_raw' in st.session_state:
        st.write("Исходные данные X_raw доступны.")
        # TODO: Добавить функционал извлечения и отображения признаков
        st.subheader("Параметры извлечения признаков")
        sampling_rate_fe = st.number_input("Частота дискретизации (Гц):", min_value=1, value=128, key="sr_fe")
        if st.button("Извлечь признаки"):
            try:
                X_features = extract_features_from_samples(st.session_state['X_raw'], sampling_rate=sampling_rate_fe)
                st.session_state['X_features'] = X_features
                st.success("Признаки успешно извлечены!")
                st.write(f"Извлеченные признаки имеют форму: {X_features.shape}")
            except Exception as e:
                st.error(f"Ошибка при извлечении признаков: {e}")
    else:
        st.info("Пожалуйста, сначала загрузите данные на странице 'Загрузка и Предварительная Обработка'.")


elif page_selection == "Моделирование и Обучение":
    st.header("Моделирование и Обучение")
    st.write("Создавайте, обучайте и тестируйте свои модели. Используйте наш `Trainer` для эффективного управления процессом обучения.")
    if 'X_features' in st.session_state and 'y_raw' in st.session_state:
        st.write("Признаки и целевая переменная доступны.")
        st.subheader("Настройки модели Logistic Regression")
        col1_m, col2_m = st.columns(2)
        with col1_m:
            lr = st.number_input("Скорость обучения (lr):", min_value=1e-5, max_value=1.0, value=0.01, format="%.5f")
        with col2_m:
            l2 = st.number_input("L2 регуляризация:", min_value=0.0, max_value=1.0, value=0.01, format="%.5f")
        
        epochs = st.number_input("Количество эпох:", min_value=1, value=100)
        batch_size = st.number_input("Размер батча:", min_value=1, value=32)
        
        if st.button("Обучить модель"):
            try:
                model = LogisticRegression(lr=lr, l2=l2)
                trainer = Trainer(model, verbose=False) # verbose=False, чтобы не забивать консоль
                
                # Добавление прогресс-бара для обучения
                progress_text = "Обучение модели..."
                my_bar = st.progress(0, text=progress_text)
                
                # Имитация пошагового обучения для обновления прогресс-бара
                for epoch in range(epochs):
                    # Здесь должен быть вызов trainer.train для одной эпохи
                    # Но trainer.train() обучает на все эпохи сразу.
                    # Для Streamlit лучше переделать Trainer или имитировать
                    # st.write(f"Эпоха {epoch+1}/{epochs}") # Не выводить каждую эпоху в UI
                    my_bar.progress((epoch + 1) / epochs, text=f"Эпоха {epoch+1}/{epochs}")
                    # В реальной реализации: trainer.train_one_epoch(...)
                    
                # Пока что запускаем полный trainer.train
                trainer.train(st.session_state['X_features'], st.session_state['y_raw'],
                               epochs=epochs, batch_size=batch_size, shuffle=True)
                
                st.session_state['trained_model'] = model
                st.success("Модель успешно обучена!")
                
                st.subheader("Оценка модели")
                y_pred = model.predict(st.session_state['X_features'])
                evaluation_metrics = model.evaluate(st.session_state['X_features'], st.session_state['y_raw'])
                st.write(f"Метрики модели на обучающей выборке: {evaluation_metrics}")
                
            except Exception as e:
                st.error(f"Ошибка при обучении модели: {e}")
    else:
        st.info("Пожалуйста, сначала загрузите данные и извлеките признаки.")

elif page_selection == "Model Hub":
    st.header("Model Hub")
    st.write("Загружайте и управляйте внешними моделями (например, из Hugging Face) или собственными сохраненными моделями.")
    # TODO: Добавить функционал для ModelHub

elif page_selection == "Эксперименты":
    st.header("Управление Экспериментами")
    st.write("Отслеживайте свои эксперименты, сохраняйте конфигурации, логи и результаты.")
    # TODO: Добавить функционал управления экспериментами

st.markdown("---")
st.caption(f"Версия приложения: {st.session_state.get('version', '0.1.0')}") # Используем session_state для версии

# Инициализация состояния (если нужно)
if 'version' not in st.session_state:
    st.session_state['version'] = "0.1.0"

# Инструкции по запуску (только если файл запускается как скрипт, что Streamlit делает сам)
# Эта часть не будет видна в Streamlit приложении, но полезна для отладки
if __name__ == "__main__":
    pass # Streamlit запускает приложение через 'streamlit run'