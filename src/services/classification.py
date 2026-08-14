from src.api.explainer import get_observed_signals
from src.inference.classifier import ClassifierEngine
from src.services.capacity import InferenceCapacity


class ClassificationService:
    def __init__(self, classifier: ClassifierEngine, capacity: InferenceCapacity):
        self.classifier = classifier
        self.capacity = capacity

    def classify(self, text: str, include_signals: bool = True) -> dict:
        with self.capacity.claim():
            result = self.classifier.predict(text)
            result["observed_signals"] = (
                get_observed_signals(text) if include_signals else []
            )
        return result
