"""
Подпакет plot_utils — инструменты визуализации данных, сигналов и моделей.

Содержит модули:
- feature_plots: статистическая визуализация (распределения, корреляции)
- signal_plots: графики сигналов
- model_plots: графики метрик моделей
- interactive_plots: интерактивные графики Plotly
- export_utils: экспорт графиков и отчётов
"""

from .feature_plots import (
    plot_distribution,
    plot_correlation_matrix,
    plot_pairwise,
    plot_feature_importance,
)

from .signal_plots import (
    generate_signal,
    plot_signal,
    compare_signals,
    plot_amplitude_spectrum,
    plot_phase_spectrum,
    plot_frequency_bands,
    plot_dominant_frequencies,
    plot_comparison_in_time_domain
)

from .model_plots import (
    plot_learning_curve,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    plot_precision_recall_curve
)

from .interactive_plots import (
    plot_interactive_signal,
    plot_3d_features,
    plot_heatmap_interactive,
    plot_interactive_learning_curve,
    plot_interactive_spectrum
)

from .export_utils import (
    create_export_dir,
    save_matplotlib_plot,
    save_plotly_figure,
    export_summary_report,
)

__all__ = [
    "plot_distribution", "plot_correlation_matrix", "plot_pairwise", "plot_feature_importance",
    "plot_learning_curve", "plot_confusion_matrix", "plot_roc_curve", "plot_precision_recall_curve",
    "plot_interactive_signal", "plot_3d_features", "plot_heatmap_interactive", "plot_interactive_learning_curve", "plot_interactive_spectrum",
    "create_export_dir", "save_matplotlib_plot", "save_plotly_figure", "export_summary_report",
    "generate_signal", "plot_signal", "compare_signals", "plot_amplitude_spectrum", "plot_phase_spectrum",
    "plot_frequency_bands", "plot_dominant_frequencies", "plot_comparison_in_time_domain"
]