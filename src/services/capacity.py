from contextlib import contextmanager
from threading import BoundedSemaphore
from time import monotonic

from src.errors import InferenceCapacityError


class InferenceCapacity:
    def __init__(self, limit: int, wait_seconds: float):
        self._semaphore = BoundedSemaphore(limit)
        self._wait_seconds = wait_seconds

    @contextmanager
    def claim(self, context=None):
        deadline = monotonic() + self._wait_seconds
        acquired = False
        while not acquired:
            if context is not None:
                context.check()
            remaining = max(0, deadline - monotonic())
            acquired = self._semaphore.acquire(timeout=min(.1, remaining))
            if not acquired and monotonic() >= deadline:
                raise InferenceCapacityError("Inference capacity is temporarily exhausted")
        try:
            if context is not None:
                context.check()
            yield
        finally:
            self._semaphore.release()
