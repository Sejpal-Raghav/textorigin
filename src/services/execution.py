from dataclasses import dataclass, field
from threading import Event
from typing import Callable

from src.errors import OperationCancelled


@dataclass
class ExecutionContext:
    cancelled: Event = field(default_factory=Event)
    on_progress: Callable[[str, int, int | None], None] = field(
        default=lambda _stage, _done, _total: None
    )

    def check(self) -> None:
        if self.cancelled.is_set():
            raise OperationCancelled("Operation cancelled")

    def progress(self, stage: str, done: int = 0, total: int | None = None) -> None:
        self.check()
        self.on_progress(stage, done, total)


def context_or_default(context: ExecutionContext | None) -> ExecutionContext:
    return context if context is not None else ExecutionContext()
