import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import skew, kurtosis, entropy
from pyn_utils.data_utils import normalize, summary
from pyn_utils.array_utils import ArrayProcessor
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from .plot_utils import signal_plots as pu


class SignalProcessor:
    """Класс для обработки и анализа сигналов"""

    def __init__(self, signal, sampling_rate: float):
        """ Инициализация сигнала с проверкой типа"""
        
        if not isinstance(signal, (list, np.ndarray)):
            raise TypeError("Сигнал должен быть списком или массивом NumPy")
        if len(signal) == 0:
            raise ValueError("Сигнал не может быть пустым")
        self.signal = np.asarray(signal)
        self.sr = sampling_rate
        self.t = np.linspace(0, self.duration(), len(self.signal), endpoint=False)

    def normalize(self, scale=1):
        """Нормализация сигнала"""
        
        return normalize(self.signal, scale=scale)

    def moving_average(self, window=3):
        """Сглаживание сигнала скользящим средним"""
        
        return [sum(self.signal[i:i+window]) / window for i in range(len(self.signal) - window + 1)]

    def summary(self):
        """Минимум, максимум и среднее значения сигнала"""
        
        return summary(self.signal)
    
    def duration(self) -> float:
        """Возвращает длительность сигнала в секундах."""

        if not self.sr or self.sr <= 0:
            raise ValueError("Частота дискретизации должна быть задана")
        else:
            return len(self.signal) / self.sr

    def to_numpy(self) -> np.ndarray:
        """Возвращает сигнал как numpy-массив."""

        numpy_signal = np.asarray(self.signal)

        return numpy_signal

    def detect_outliers(self, method: str = "zscore", threshold: float = 3.0) -> np.ndarray:
        """Возвращает булев массив, где True — выбросы (для последующей фильтрации)."""

        x = self.to_numpy()

        if method == "zscore":
            scaler = ArrayProcessor()
            outliers_mask = np.abs(scaler.standardize(x)) > threshold

        elif method == "iqr":
            q1 = np.quantile(x, 0.25)
            q3 = np.quantile(x, 0.75)
            iqr = q3 - q1
            outliers_mask = (x < q1 - threshold * iqr) | (x > q3 + threshold * iqr)
        
        else:
            raise ValueError("Выбран неправильный метод")
        
        return outliers_mask

    def subtract_mean(self) -> np.ndarray:
        """Удаляет среднее значение (центровка сигнала)."""

        x = self.to_numpy()
        mu = np.mean(x)
        x_centered = x - mu

        return x_centered

    def detrend(self, order=1) -> np.ndarray:
        """Убирает линейный или полиномиальный тренд из сигнала."""
        x = self.to_numpy()
        t = np.arange(len(x)) / self.sr

        if order == 0:
            return self.subtract_mean()

        coeffs = np.polyfit(t, x, order)
        trend = np.polyval(coeffs, t)

        return x - trend

    def compute_fft(self) -> tuple[np.ndarray, np.ndarray]:
        """Вычисляет спектр сигнала (амплитудный и фазовый). Возвращает (amplitude, phase)."""

        x = self.to_numpy()
        fft_result = np.fft.fft(x)
        amplitude = np.abs(fft_result) / len(x)
        phase = np.angle(fft_result)

        return amplitude, phase

    def get_frequency_axis(self) -> np.ndarray:
        """Возвращает массив частот (в Гц) для текущего сигнала."""

        n = len(self.signal)

        return np.fft.fftfreq(n, d=1/self.sr)

    def compute_amplitude_spectrum(self):
        """Вычисляет амплитудный спектр сигнала."""

        amplitude, _ = self.compute_fft()

        return amplitude

    def compute_phase_spectrum(self):
        """Вычисляет фазовый спектр сигнала."""

        _, phase = self.compute_fft()
        
        return phase

    def reconstruct_signal(self):
        """Восстанавливает сигнал из спектра (обратное FFT)."""

        x = self.to_numpy()
        
        return np.real(np.fft.ifft(np.fft.fft(x)))

    def dominant_frequencies(self, n: int = 3):
        """Возвращает n частот с наибольшей амплитудой."""

        x = self.to_numpy()
        freqs = np.fft.rfftfreq(len(x), 1 / self.sr)
        amplitudes = np.abs(np.fft.rfft(x))

        indices = np.argsort(amplitudes)[-n:][::-1]  # топ-n

        return freqs[indices], amplitudes[indices]

    def get_main_frequency(self) -> float:
        """Определяет доминирующую частоту (с наибольшей амплитудой)."""

        freqs, amps = self.dominant_frequencies(1)

        return freqs[0]

    def filter_frequency_range(self, f_min: float, f_max: float) -> np.ndarray:
        """Выделяет часть сигнала в заданном диапазоне частот."""

        x = self.to_numpy()
        fft_vals = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(len(x), 1 / self.sr)

        mask = (freqs >= f_min) & (freqs <= f_max)
        fft_filtered = np.zeros_like(fft_vals)
        fft_filtered[mask] = fft_vals[mask]

        filtered_signal = np.fft.irfft(fft_filtered)

        return filtered_signal

    def low_pass_filter(self, cutoff: float, order: int = 4):
        """Применяет низкочастотный фильтр с заданной частотой среза."""

        if not (0 < cutoff < self.sr / 2):
            raise ValueError("cutoff должен быть в пределах (0, sr/2)")

        x = self.to_numpy()
        Wn = cutoff / (self.sr / 2)
        b, a = butter(order, Wn, btype='low')

        return filtfilt(b, a, x)

    def high_pass_filter(self, cutoff: float, order: int = 4):
        """Применяет высокочастотный фильтр с заданной частотой среза."""

        if not (0 < cutoff < self.sr / 2):
            raise ValueError("cutoff должен быть в пределах (0, sr/2)")

        x = self.to_numpy()
        Wn = cutoff / (self.sr / 2)
        b, a = butter(order, Wn, btype='high')

        return filtfilt(b, a, x)

    def band_pass_filter(self, low_cut: float, high_cut: float, order: int = 4):
        """Применяет полосовой фильтр (пропускает частоты между low_cut и high_cut)."""

        if not (0 < low_cut < high_cut < self.sr / 2):
            raise ValueError("low_cut и high_cut должны быть в пределах (0, sr/2)")

        x = self.to_numpy()
        Wn = [low_cut / (self.sr / 2), high_cut / (self.sr / 2)]
        b, a = butter(order, Wn, btype='band')

        return filtfilt(b, a, x)

    def band_stop_filter(self, low_cut: float, high_cut: float, order: int = 4):
        """Применяет режекторный фильтр (удаляет частоты между low_cut и high_cut)."""

        if not (0 < low_cut < high_cut < self.sr / 2):
            raise ValueError("low_cut и high_cut должны быть в пределах (0, sr/2)")

        x = self.to_numpy()
        Wn = [low_cut / (self.sr / 2), high_cut / (self.sr / 2)]
        b, a = butter(order, Wn, btype='bandstop')
        return filtfilt(b, a, x)

    def smooth(self, window_size: int = 5):
        """Применяет сглаживание скользящим средним."""

        if window_size < 1:
            raise ValueError("Размер окна должен быть >= 1")
        
        x = self.to_numpy()
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(x, kernel, mode='same')

        return smoothed

    def extract_time_features(self) -> dict:
        """Извлекает временные признаки: mean, var, RMS, skewness, kurtosis, zero-crossings, Hjorth."""

        x = self.to_numpy()
        n = len(x)

        mean_val = np.mean(x)
        var_val = np.var(x)
        rms_val = np.sqrt(np.mean(x**2))
        skew_val = skew(x)
        kurt_val = kurtosis(x)

        zero_crossings = np.sum(np.diff(np.sign(x)) != 0)

        # Hjorth-параметры
        dx = np.diff(x)
        var_dx = np.var(dx)
        mobility = np.sqrt(var_dx / var_val)
        complexity = np.sqrt(np.var(np.diff(dx)) / var_dx) / mobility

        return {
            "mean": mean_val,
            "var": var_val,
            "rms": rms_val,
            "skewness": skew_val,
            "kurtosis": kurt_val,
            "zero_crossings": zero_crossings,
            "hjorth_mobility": mobility,
            "hjorth_complexity": complexity
        }

    def extract_spectral_features(self) -> dict:
        """Извлекает частотные признаки на основе FFT: power, centroid, entropy."""

        x = self.to_numpy()
        freqs = np.fft.rfftfreq(len(x), 1 / self.sr)
        spectrum = np.abs(np.fft.rfft(x)) ** 2  # мощность

        total_power = np.sum(spectrum)
        centroid = np.sum(freqs * spectrum) / total_power
        spectral_entropy = entropy(spectrum / total_power)

        return {
            "power": total_power,
            "centroid": centroid,
            "entropy": spectral_entropy
        }

    def band_power(self, f_min: float, f_max: float) -> float:
        """Вычисляет суммарную мощность сигнала в заданном диапазоне частот."""

        x = self.to_numpy()
        freqs = np.fft.rfftfreq(len(x), 1 / self.sr)
        spectrum = np.abs(np.fft.rfft(x)) ** 2

        mask = (freqs >= f_min) & (freqs <= f_max)

        return np.sum(spectrum[mask])

    def extract_band_features(self) -> dict:
        """Возвращает мощность стандартных диапазонов (delta, theta, alpha, beta, gamma)."""

        bands = {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 45)
        }

        return {name: self.band_power(low, high) for name, (low, high) in bands.items()}

    def feature_vector(self) -> np.ndarray:
        """Возвращает объединённый вектор признаков (временные + спектральные)."""

        time_features = self.extract_time_features()
        spectral_features = self.extract_spectral_features()
        band_features = self.extract_band_features()

        all_features = {**time_features, **spectral_features, **band_features}
        
        return np.array(list(all_features.values()))

    def plot_signal(self):
        """Строит сигнал во временной области."""

        pu.plot_signal(self.t, self.signal, title="Исходный сигнал", xlabel="Время (с)", ylabel="Амплитуда")

    def plot_filtered(self, filtered_signal: np.ndarray, label: str = "Filtered"):
        """Строит исходный и отфильтрованный сигналы на одном графике."""

        pu.compare_signals(self.t, self.signal, filtered_signal, labels=("Исходный", label))

    def plot_spectrum(self):
        """Строит график амплитудного спектра сигнала."""

        freqs = np.fft.rfftfreq(len(self.signal), 1 / self.sampling_rate)
        amplitudes = np.abs(np.fft.rfft(self.signal))
        pu.plot_amplitude_spectrum(freqs, amplitudes)

    def plot_fft(self):
        """Строит амплитудный и фазовый спектры."""
        freqs = np.fft.rfftfreq(len(self.signal), 1 / self.sampling_rate)
        fft = np.fft.rfft(self.signal)
        amplitudes = np.abs(fft)
        phases = np.angle(fft)

        pu.plot_amplitude_spectrum(freqs, amplitudes)
        pu.plot_phase_spectrum(freqs, phases)

    def plot_spectrogram(self, nperseg: int = 128, noverlap: int = 64, cmap: str = "viridis"):
        """Строит спектрограмму сигнала (время–частота)."""

        f, t, Sxx = spectrogram(self.signal, self.sampling_rate, nperseg=nperseg, noverlap=noverlap)
        plt.figure(figsize=(8, 4))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading="gouraud", cmap=cmap)
        plt.ylabel("Частота (Гц)")
        plt.xlabel("Время (с)")
        plt.title("Спектрограмма сигнала")
        plt.colorbar(label="Мощность (дБ)")
        plt.show()

    def plot_features(self, features: dict):
        """Визуализирует признаки сигнала (в виде bar-графика)."""
        
        if not features:
            raise ValueError("Пустой словарь признаков")
        keys = list(features.keys())
        values = [features[k] for k in keys]
        plt.figure(figsize=(8, 4))
        plt.bar(keys, values, color="skyblue")
        plt.title("Признаки сигнала")
        plt.ylabel("Значение")
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

