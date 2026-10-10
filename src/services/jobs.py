"""Bounded, transient jobs for a trusted-local, single-process deployment."""
from collections import OrderedDict, deque
from dataclasses import dataclass, field
import secrets
from threading import Condition, Event, Thread
from time import monotonic
from typing import Callable

from src.errors import JobCapacityError, ModelUnavailableError, OperationCancelled
from src.services.execution import ExecutionContext

TERMINAL = {"completed", "failed", "cancelled"}


@dataclass
class Job:
    id: str
    operation: str
    payload: object | None
    state: str = "queued"
    stage: str = "queued"
    completed_units: int = 0
    total_units: int | None = None
    result: dict | None = None
    error: dict | None = None
    finished_at: float | None = None
    cancel_event: Event = field(default_factory=Event)
    payload_bytes: int = 0


class JobManager:
    def __init__(self, handler: Callable, *, workers=1, max_queued=4,
                 max_retained=32, ttl_seconds=600, max_payload_bytes=32 * 1024 * 1024):
        if min(workers, max_queued, max_retained, ttl_seconds, max_payload_bytes) < 1:
            raise ValueError("Job limits must be positive")
        self.handler = handler
        self.max_queued = max_queued
        self.max_retained = max_retained
        self.ttl_seconds = ttl_seconds
        self.max_payload_bytes = max_payload_bytes
        self._condition = Condition()
        self._jobs = OrderedDict()
        self._queue = deque()
        self._closed = False
        self._threads = [Thread(target=self._worker, name=f"textorigin-job-{i}", daemon=True)
                         for i in range(workers)]
        for thread in self._threads:
            thread.start()

    def _expire(self):
        now = monotonic()
        for key, job in list(self._jobs.items()):
            if job.finished_at is not None and now - job.finished_at >= self.ttl_seconds:
                del self._jobs[key]

    def submit(self, operation, payload) -> dict:
        size = len(payload.text.encode("utf-8"))
        with self._condition:
            self._expire()
            if self._closed or len(self._queue) >= self.max_queued:
                raise JobCapacityError("The local job queue is full or shutting down")
            retained_bytes = sum(job.payload_bytes for job in self._jobs.values())
            # Reserve space for original + revised output and metadata. This is
            # operational admission, not truncation or a product word limit.
            reservation = size * 4 + 4096
            if retained_bytes + reservation > self.max_payload_bytes:
                raise JobCapacityError("The transient job memory budget is exhausted")
            while len(self._jobs) >= self.max_retained:
                terminal = next((key for key, job in self._jobs.items() if job.state in TERMINAL), None)
                if terminal is None:
                    raise JobCapacityError("Too many active jobs")
                del self._jobs[terminal]
            job = Job(secrets.token_urlsafe(24), operation, payload, payload_bytes=reservation)
            self._jobs[job.id] = job
            self._queue.append(job.id)
            self._condition.notify()
            return self._snapshot(job)

    def _snapshot(self, job):
        return {"id": job.id, "operation": job.operation, "state": job.state,
                "stage": job.stage, "completed_units": job.completed_units,
                "total_units": job.total_units, "result": job.result,
                "error": job.error, "status_url": f"/jobs/{job.id}",
                "complete": job.state == "completed"}

    def get(self, key):
        with self._condition:
            self._expire()
            if key not in self._jobs:
                raise KeyError(key)
            return self._snapshot(self._jobs[key])

    def cancel(self, key):
        with self._condition:
            self._expire()
            if key not in self._jobs:
                raise KeyError(key)
            job = self._jobs[key]
            if job.state not in TERMINAL:
                job.cancel_event.set()
                if job.state == "queued":
                    self._queue.remove(key)
                    self._finish(job, "cancelled")
                    job.payload = None
                else:
                    job.state = "cancel_requested"
                    job.stage = "cancelling"
                self._condition.notify_all()
            return self._snapshot(job)

    def _finish(self, job, state, result=None, error=None):
        job.state = state
        job.stage = state
        job.result = result
        job.error = error
        job.finished_at = monotonic()
        job.payload = None
        if state != "completed":
            job.payload_bytes = 0

    def _worker(self):
        while True:
            with self._condition:
                self._expire()
                while not self._queue and not self._closed:
                    self._condition.wait(timeout=1)
                    self._expire()
                if self._closed and not self._queue:
                    return
                key = self._queue.popleft()
                job = self._jobs[key]
                job.state = "running"
                job.stage = "starting"
                payload = job.payload
                # The manager no longer retains a second credential-bearing reference.
                job.payload = None

            def progress(stage, done, total):
                with self._condition:
                    if job.state == "running":
                        job.stage, job.completed_units, job.total_units = stage, done, total

            context = ExecutionContext(job.cancel_event, progress)
            try:
                context.check()
                result = self.handler(job.operation, payload, context)
                context.check()
                with self._condition:
                    if job.cancel_event.is_set():
                        self._finish(job, "cancelled")
                    else:
                        self._finish(job, "completed", result=result)
            except OperationCancelled:
                with self._condition:
                    self._finish(job, "cancelled")
            except Exception as exc:
                # Never expose provider payloads, input, credentials or tracebacks.
                code = "model_unavailable" if isinstance(exc, ModelUnavailableError) else "job_failed"
                detail = "Required local model is unavailable" if code == "model_unavailable" else "Operation failed; check local configuration"
                with self._condition:
                    self._finish(job, "cancelled" if job.cancel_event.is_set() else "failed",
                                 error=None if job.cancel_event.is_set() else {"code": code, "detail": detail})
            finally:
                payload = None
                result = None

    def close(self, timeout=65):
        with self._condition:
            self._closed = True
            for job in self._jobs.values():
                if job.state not in TERMINAL:
                    job.cancel_event.set()
                    if job.state == "queued":
                        self._finish(job, "cancelled")
                    else:
                        job.state = "cancel_requested"
            self._queue.clear()
            self._condition.notify_all()
        deadline = monotonic() + timeout
        for thread in self._threads:
            thread.join(max(0, deadline - monotonic()))
        with self._condition:
            self._jobs.clear()
