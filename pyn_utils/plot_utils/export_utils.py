from pathlib import Path
import datetime
import json
from pyn_utils.file_utils import FileHandler


def create_export_dir(base_path="exports"):
    """Создаёт директорию для экспорта, если её нет, и возвращает путь."""
    
    directory = Path(base_path)
    directory.mkdir(parents=True, exist_ok=True)

    return directory


def get_timestamp():
    """Возвращает текущую временную метку (строкой)."""
    
    now = datetime.datetime.now()
    timestamp_string = now.strftime("%Y%m%d_%H%M%S")

    return timestamp_string


def save_matplotlib_plot(fig, filename, format="png", export_dir="exports"):
    """Сохраняет график Matplotlib в заданный формат."""
    
    timestamp = get_timestamp()
    filepath = Path(export_dir) / f"{filename}_{timestamp}.{format}"
    fig.savefig(filepath)


def save_plotly_figure(fig, filename, format="html", export_dir="exports"):
    """Сохраняет интерактивный Plotly-график (html или png)."""
    
    timestamp = get_timestamp()
    filepath = Path(export_dir) / f"{filename}_{timestamp}.{format}"

    if format == "html":
        fig.write_html(filepath)
    elif format == "png":
        fig.write_image(filepath)
    else:
        raise ValueError("Указан неверный формат файла")



def export_summary_report(summary_dict, filename="summary.json", export_dir="exports"):
    """Сохраняет словарь с метаданными анализа в JSON."""
    
    timestamp = get_timestamp()
    filepath = Path(export_dir) / f"{filename}_{timestamp}.json"

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=4, ensure_ascii=False)

    return filepath


def export_full_report(fig, df, summary, base_name="analysis_result"):
    """
    Экспорт полного отчёта:
      - сохраняет график (Matplotlib),
      - сохраняет DataFrame (через FileHandler),
      - сохраняет JSON-summary.
    """
    
    handler = FileHandler()
    export_dir = create_export_dir()
    timestamp = get_timestamp()

    # График
    fig_path = Path(export_dir) / f"{base_name}_{timestamp}.png"
    fig.savefig(fig_path)

    # Таблица
    df_path = Path(export_dir) / f"{base_name}_{timestamp}.csv"
    handler.save(df, df_path, format="csv")

    # Summary
    summary_path = Path(export_dir) / f"{base_name}_{timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    return {
        "figure": fig_path,
        "data": df_path,
        "summary": summary_path
    }