from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from src.config import get_settings


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TextRequest(ApiModel):
    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text cannot be empty")
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
    total_tokens: int | None = None
    window_count: int = 1
    coverage: float = 1.0
    complete: bool = True
    aggregation: str = "single_window"


class FeatureMetrics(ApiModel):
    perplexity: float
    burstiness: float
    entropy: float
    ai_phrase_rate: float
    avg_sentence_length: float
    structural_regularity: float


class HumanizeRequest(TextRequest):
    use_llm: bool = True
    style: Literal["natural", "formal", "casual"] = "natural"
    strength: Literal["light", "medium"] = "light"
    deepseek_api_key: SecretStr | None = Field(default=None, min_length=1, max_length=4096)
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
    style: Literal["natural", "formal", "casual"]
    strength: Literal["light", "medium"]
    original_metrics: FeatureMetrics
    final_metrics: FeatureMetrics | None = None
    post_heuristic_metrics: FeatureMetrics
    heuristic_similarity: float
    llm_similarity: float | None = None
    post_llm_metrics: FeatureMetrics | None = None
    chunks_processed: int = Field(ge=1)
    complete: bool


class HealthResponse(ApiModel):
    status: Literal["ok"] = "ok"


class CompareRequest(HumanizeRequest):
    use_llm: bool = False


class CompareResponse(ApiModel):
    schema_version: str
    operation: Literal["compare"]
    style: Literal["natural", "formal", "casual"]
    strength: Literal["light", "medium"]
    original_text: str
    humanized_text: str
    original_scores: dict[str, float]
    revised_scores: dict[str, float]
    delta_percentage_points: dict[str, float]
    original_analysis: dict
    revised_analysis: dict
    rewrite: HumanizeResponse
    complete: bool


class JobRequest(HumanizeRequest):
    operation: Literal["classify", "humanize", "compare"]
    use_llm: bool = False
    include_signals: bool = False


class JobResponse(ApiModel):
    id: str
    operation: Literal["classify", "humanize", "compare"]
    state: Literal["queued", "running", "cancel_requested", "cancelled", "completed", "failed"]
    stage: str
    completed_units: int
    total_units: int | None
    result: dict | None
    error: dict | None
    status_url: str
    complete: bool
