import os

def find_null_bytes(directory='.'):
    """
    Проверяет все файлы в указанной директории и ее поддиректориях
    на наличие нулевых байтов (b'\x00').

    :param directory: Путь к директории для сканирования.
    """
    found_null_files = []
    # Расширения файлов, которые мы хотим проверять
    # Можно добавить больше, если нужно
    file_extensions_to_check = ('.py', '.txt', '.md', '.json', '.csv', '.yaml', '.xml', '.yml')

    print(f"Сканирование директории '{os.path.abspath(directory)}' на наличие нулевых байтов...")

    for root, _, files in os.walk(directory):
        for file_name in files:
            # Пропускаем файлы в виртуальном окружении
            if "\venv" in root.lower() or "/venv/" in root.lower():
                continue
            # Проверяем только файлы с определенными расширениями
            if not file_name.lower().endswith(file_extensions_to_check):
                continue

            file_path = os.path.join(root, file_name)
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
                    if b'\x00' in content:
                        found_null_files.append(file_path)
                        print(f"  [ОБНАРУЖЕНО] {file_path}")
            except (IOError, OSError) as e:
                print(f"  [ОШИБКА ЧТЕНИЯ] {file_path}: {e}")
            except Exception as e:
                print(f"  [НЕИЗВЕСТНАЯ ОШИБКА] {file_path}: {e}")

    print("Сканирование завершено.")
    if found_null_files:
        print("Следующие файлы содержат нулевые байты:")
        for f in found_null_files:
            print(f"- {f}")
        print("Эти файлы могут вызывать 'SyntaxError: source code string cannot contain null bytes'.")
        print("Рекомендуется открыть их в бинарном редакторе или редакторе, который показывает непечатаемые символы, чтобы найти и удалить нулевые байты.")
    else:
        print("Нулевые байты не обнаружены ни в одном из проверенных файлов.")

if __name__ == "__main__":
    find_null_bytes()