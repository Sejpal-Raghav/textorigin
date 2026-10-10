from typing import Protocol

from src.services.capacity import InferenceCapacity
from src.services.execution import ExecutionContext, context_or_default


class Classifier(Protocol):
    def predict(self, text: str) -> dict: ...


class ClassificationService:
    def __init__(self, classifier: Classifier, capacity: InferenceCapacity):
        self.classifier = classifier
        self.capacity = capacity

    def classify(self, text: str, include_signals: bool = True, context: ExecutionContext | None = None) -> dict:
        with self.capacity.claim(context):
            execution = context_or_default(context)
            execution.check()
            result = self.classifier.predict(text) if context is None else self.classifier.predict(text, context=context)
            execution.progress("observing_signals")
            if include_signals:
                from src.api.explainer import get_observed_signals

                result["observed_signals"] = get_observed_signals(text, context=execution)
            else:
                result["observed_signals"] = []
            execution.check()
        return result
