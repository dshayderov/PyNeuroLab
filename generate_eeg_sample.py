import numpy as np
import pandas as pd
import sys

# Параметры для генерации данных
num_samples = 100  # Количество "эпох" или коротких сегментов ЭЭГ
signal_length = 256  # Длина каждого сегмента (временные точки)
sampling_rate = 128  # Частота дискретизации (Гц)

data = []
for i in range(num_samples):
    t = np.linspace(0, signal_length / sampling_rate, signal_length, endpoint=False)

    freq1 = 8 + np.random.rand() * 4  # Альфа-диапазон
    freq2 = 18 + np.random.rand() * 4 # Бета-диапазон
    
    channel1 = np.sin(2 * np.pi * freq1 * t + np.random.rand() * 2 * np.pi) + np.random.randn(signal_length) * 0.2
    channel2 = np.sin(2 * np.pi * freq2 * t + np.random.rand() * 2 * np.pi) + np.random.randn(signal_length) * 0.2
    channel3 = np.random.randn(signal_length) * 0.5

    label = 1 if np.mean(channel1) > 0.05 else 0

    sample_row = {}
    for j in range(signal_length):
        sample_row[f'ch1_t{j}'] = channel1[j]
        sample_row[f'ch2_t{j}'] = channel2[j]
        sample_row[f'ch3_t{j}'] = channel3[j]
    sample_row['label'] = label
    data.append(sample_row)

df = pd.DataFrame(data)

# Выводим данные в stdout в формате CSV
df.to_csv('datasets/eeg_sample.csv', index=False)
# sys.stdout.write(df.to_csv(index=False))
