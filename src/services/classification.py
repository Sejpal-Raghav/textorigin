from typing import Protocol

from src.services.capacity import InferenceCapacity


class Classifier(Protocol):
    def predict(self, text: str) -> dict: ...


class ClassificationService:
    def __init__(self, classifier: Classifier, capacity: InferenceCapacity):
        self.classifier = classifier
        self.capacity = capacity

    def classify(self, text: str, include_signals: bool = True) -> dict:
        with self.capacity.claim():
            result = self.classifier.predict(text)
            if include_signals:
                from src.api.explainer import get_observed_signals

                result["observed_signals"] = get_observed_signals(text)
            else:
                result["observed_signals"] = []
        return result
