import sys
sys.path.append('./')
import numpy as np
from pyn_utils import SignalProcessor


# === Тест 1. Инициализация ===
signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 500))  # 5 Гц
processor = SignalProcessor(signal, sampling_rate=500)
assert isinstance(processor.signal, (list, np.ndarray))
assert processor.sr == 500


# === Тест 2. Ошибка при пустом сигнале ===
try:
    SignalProcessor([], 500)
except ValueError as e:
    assert "не может быть пустым" in str(e)


# === Тест 3. Ошибка при неверном типе ===
try:
    SignalProcessor("12345", 500)
except TypeError as e:
    assert "должен быть списком" in str(e)


# === Тест 4. Нормализация ===
norm_signal = processor.normalize(scale=2)
assert np.isclose(np.max(norm_signal), 2.0, atol=1e-6)


# === Тест 5. Скользящее среднее ===
smoothed = processor.moving_average(window=5)
expected = np.convolve(signal, np.ones(5)/5, mode='valid')
assert np.allclose(smoothed, expected)


# === Тест 6. summary ===
summary = processor.summary()
assert isinstance(summary, dict)
assert set(summary.keys()) == {"min", "max", "mean"}


# === Тест 7. Длительность ===
assert np.isclose(processor.duration(), 1.0, atol=1e-3)


# === Тест 8. Преобразование в numpy ===
assert isinstance(processor.to_numpy(), np.ndarray)


# === Тест 9. FFT и спектр ===
amp, phase = processor.compute_fft()
assert len(amp) == len(phase) == len(signal)
freqs = processor.get_frequency_axis()
assert np.isclose(freqs[np.argmax(amp[1:])], 5, atol=0.5)


# === Тест 10. Основная частота ===
f_main = processor.get_main_frequency()
assert 4.5 < f_main < 5.5


# === Тест 11. Частотная фильтрация ===
filtered = processor.filter_frequency_range(4, 6)
assert len(filtered) == len(signal)
assert isinstance(filtered, np.ndarray)


# === Тест 12. Временные признаки ===
features_time = processor.extract_time_features()
assert all(k in features_time for k in ["mean", "rms", "zero_crossings"])


# === Тест 13. Спектральные признаки ===
features_spec = processor.extract_spectral_features()
assert "power" in features_spec and "entropy" in features_spec


# === Тест 14. Диапазоны частот ===
bands = processor.extract_band_features()
assert all(b in bands for b in ["delta", "theta", "alpha", "beta", "gamma"])


# === Тест 15. Вектор признаков ===
fv = processor.feature_vector()
assert isinstance(fv, np.ndarray)
assert fv.ndim == 1
assert len(fv) > 5


# === Тест 16. Низко- и высокочастотные фильтры ===
low = processor.low_pass_filter(10)
high = processor.high_pass_filter(10)
assert len(low) == len(high) == len(signal)


# === Тест 17. Сглаживание ===
smooth = processor.smooth(window_size=5)
assert len(smooth) == len(signal)


# === Тест 18. Графики (не должны выбрасывать ошибок) ===
try:
    processor.plot_spectrum()
    processor.plot_fft()
    processor.plot_spectrogram()
    processor.plot_features(features_spec)
except Exception as e:
    raise AssertionError(f"Ошибка при построении графика: {e}")


print("✅ Все тесты пройдены успешно!")
