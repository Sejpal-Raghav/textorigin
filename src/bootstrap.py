from dataclasses import dataclass
from functools import lru_cache

from src.config import Settings, get_settings
from src.inference.classifier import ClassifierEngine
from src.inference.similarity import SimilarityEngine
from src.integrations.ollama import OllamaClient
from src.services.capacity import InferenceCapacity
from src.services.classification import ClassificationService
from src.services.humanization import HumanizationService


@dataclass
class ApplicationServices:
    classification: ClassificationService
    humanization: HumanizationService
    classifier: ClassifierEngine
    ollama: OllamaClient

    def close(self) -> None:
        self.ollama.close()


def build_services(settings: Settings) -> ApplicationServices:
    capacity = InferenceCapacity(
        settings.max_concurrent_inferences, settings.inference_wait_seconds
    )
    classifier = ClassifierEngine(settings.model_path)
    similarity = SimilarityEngine()
    ollama = OllamaClient(
        settings.ollama_host,
        settings.ollama_model,
        settings.ollama_timeout_seconds,
    )
    return ApplicationServices(
        classification=ClassificationService(classifier, capacity),
        humanization=HumanizationService(
            similarity,
            ollama,
            capacity,
            settings.default_similarity_threshold,
            settings.minimum_similarity_threshold,
        ),
        classifier=classifier,
        ollama=ollama,
    )


@lru_cache(maxsize=1)
def get_services() -> ApplicationServices:
    return build_services(get_settings())
