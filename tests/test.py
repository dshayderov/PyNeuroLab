import os
import time
import unittest # Добавляю unittest для более формальных тестов

from pyn_utils.file_utils import read_json, write_json
from pyn_utils.data_utils import normalize
from pyn_utils.timing import Timer

# В unittest тесты лучше оформлять в классах
class TestPynUtilsIntegration(unittest.TestCase):

    def setUp(self):
        self.test_file = "data_test.json"
        self.data = {"name": "Peter", "age": 32}

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_json_rw(self):
        """Тест записи и чтения JSON."""
        write_json(self.test_file, self.data)
        self.assertTrue(os.path.exists(self.test_file))

        read_data = read_json(self.test_file)
        self.assertEqual(read_data, self.data)

    def test_normalize(self):
        """Тест нормализации."""
        self.assertEqual(normalize([1, 2, 3], scale=6), [0.0, 3.0, 6.0])

    def test_timer(self):
        """Тест работы Timer."""
        start = time.time()
        with Timer("Проверка задержки"):
            time.sleep(0.1)
        elapsed = time.time() - start
        self.assertGreaterEqual(elapsed, 0.1) # Используем assertGreaterEqual

if __name__ == '__main__':
    unittest.main()

