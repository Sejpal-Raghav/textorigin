from contextlib import contextmanager
from threading import BoundedSemaphore

from src.errors import InferenceCapacityError


class InferenceCapacity:
    def __init__(self, limit: int, wait_seconds: float):
        self._semaphore = BoundedSemaphore(limit)
        self._wait_seconds = wait_seconds

    @contextmanager
    def claim(self):
        acquired = self._semaphore.acquire(timeout=self._wait_seconds)
        if not acquired:
            raise InferenceCapacityError("Inference capacity is temporarily exhausted")
        try:
            yield
        finally:
            self._semaphore.release()
