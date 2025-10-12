import time


class Timer:
    """
    Контекстный менеджер для измерения времени выполнения блока кода.
    Пример:
        with Timer("Вычисления"):
            long_task()
    """
    
    def __init__(self, name):
        self.name = name
    
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start
        print(f"{self.name}: {elapsed_time} сек.")