import threading


class ThreadSafeCounter:
    def __init__(self, initial_count: int = 0):
        self._count = initial_count
        self._lock = threading.Lock()

    def increment(self):
        with self._lock:
            self._count += 1

    def get_count(self):
        with self._lock:
            return self._count
