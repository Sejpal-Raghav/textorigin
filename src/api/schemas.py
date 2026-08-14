from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config import get_settings


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TextRequest(ApiModel):
    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        settings = get_settings()
        if not value.strip():
            raise ValueError("Text cannot be empty")
        if len(value) > settings.max_text_chars:
            raise ValueError(
                f"Text too long (max {settings.max_text_chars:,} characters)"
            )
        return value


class ClassifyRequest(TextRequest):
    include_signals: bool | None = None


class ClassifyResponse(ApiModel):
    human: float
    ai_written: float
    ai_paraphrased: float
    observed_signals: list[str]
    truncated: bool
    analyzed_tokens: int
    model_version: str | None = None
    warnings: list[str]


class FeatureMetrics(ApiModel):
    perplexity: float
    burstiness: float
    entropy: float
    ai_phrase_rate: float
    avg_sentence_length: float
    structural_regularity: float


class HumanizeRequest(TextRequest):
    use_llm: bool = True
    similarity_threshold: float | None = Field(default=None, ge=0, le=1)
    seed: int | None = None

    @field_validator("similarity_threshold")
    @classmethod
    def enforce_server_minimum(cls, value: float | None) -> float | None:
        minimum = get_settings().minimum_similarity_threshold
        if value is not None and value < minimum:
            raise ValueError(f"similarity_threshold cannot be below {minimum}")
        return value


class HumanizeResponse(ApiModel):
    original_text: str
    humanized_text: str
    status: Literal["unchanged", "accepted_heuristic", "accepted_llm"]
    warnings: list[str]
    similarity_threshold: float
    original_metrics: FeatureMetrics
    post_heuristic_metrics: FeatureMetrics
    heuristic_similarity: float
    llm_similarity: float | None = None
    post_llm_metrics: FeatureMetrics | None = None


class HealthResponse(ApiModel):
    status: Literal["ok"] = "ok"
