from dataclasses import dataclass
from functools import lru_cache

from src.config import Settings, get_settings
from src.inference.classifier import ClassifierEngine
from src.inference.similarity import SimilarityEngine
from src.integrations.deepseek import DeepSeekClient
from src.services.capacity import InferenceCapacity
from src.services.classification import ClassificationService
from src.services.humanization import HumanizationService


@dataclass
class ApplicationServices:
    classification: ClassificationService
    humanization: HumanizationService
    classifier: ClassifierEngine
    polisher: DeepSeekClient

    def close(self) -> None:
        self.polisher.close()


def build_services(settings: Settings) -> ApplicationServices:
    capacity = InferenceCapacity(
        settings.max_concurrent_inferences, settings.inference_wait_seconds
    )
    classifier = ClassifierEngine(settings.model_path)
    similarity = SimilarityEngine()
    polisher = DeepSeekClient(
        settings.deepseek_base_url,
        settings.deepseek_model,
        settings.deepseek_timeout_seconds,
        settings.deepseek_api_key,
    )
    return ApplicationServices(
        classification=ClassificationService(classifier, capacity),
        humanization=HumanizationService(
            similarity,
            polisher,
            capacity,
            settings.default_similarity_threshold,
            settings.minimum_similarity_threshold,
        ),
        classifier=classifier,
        polisher=polisher,
    )


@lru_cache(maxsize=1)
def get_services() -> ApplicationServices:
    return build_services(get_settings())
