from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "models" if (PROJECT_ROOT / "pyproject.toml").is_file() else Path.home() / ".textorigin" / "models"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    path = Path(value).expanduser() if value else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.expanduser().resolve()


@dataclass(frozen=True)
class Settings:
    model_path: Path
    feature_thresholds_path: Path
    cors_origins: tuple[str, ...]
    default_similarity_threshold: float
    minimum_similarity_threshold: float
    max_concurrent_inferences: int
    inference_wait_seconds: float
    include_signals_by_default: bool
    deepseek_base_url: str
    deepseek_model: str
    deepseek_api_key: str | None
    deepseek_timeout_seconds: float
    log_level: str
    classifier_batch_size: int = 2
    job_queue_size: int = 4
    job_retained_count: int = 32
    job_ttl_seconds: int = 600
    job_memory_bytes: int = 32 * 1024 * 1024

    @property
    def manifest_path(self) -> Path:
        return self.model_path / "textorigin-manifest.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    origins = tuple(
        origin.strip()
        for origin in os.getenv("TEXTORIGIN_CORS_ORIGINS", "http://localhost:3000").split(",")
        if origin.strip()
    )
    settings = Settings(
        model_path=_env_path(
            "TEXTORIGIN_MODEL_PATH", DEFAULT_MODEL_ROOT / "roberta-classifier"
        ),
        feature_thresholds_path=_env_path(
            "TEXTORIGIN_FEATURE_THRESHOLDS_PATH",
            DEFAULT_MODEL_ROOT / "feature-thresholds.json",
        ),
        cors_origins=origins,
        default_similarity_threshold=_env_float(
            "TEXTORIGIN_SIMILARITY_THRESHOLD", 0.85
        ),
        minimum_similarity_threshold=_env_float(
            "TEXTORIGIN_MIN_SIMILARITY_THRESHOLD", 0.80
        ),
        max_concurrent_inferences=_env_int(
            "TEXTORIGIN_MAX_CONCURRENT_INFERENCES", 1
        ),
        inference_wait_seconds=_env_float("TEXTORIGIN_INFERENCE_WAIT_SECONDS", 5.0),
        include_signals_by_default=_env_bool(
            "TEXTORIGIN_INCLUDE_SIGNALS", False
        ),
        deepseek_base_url=os.getenv(
            "TEXTORIGIN_DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        ).rstrip("/"),
        deepseek_model=os.getenv("TEXTORIGIN_DEEPSEEK_MODEL", "deepseek-chat"),
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
        deepseek_timeout_seconds=_env_float(
            "TEXTORIGIN_DEEPSEEK_TIMEOUT_SECONDS", 60.0
        ),
        classifier_batch_size=_env_int("TEXTORIGIN_CLASSIFIER_BATCH_SIZE", 2),
        job_queue_size=_env_int("TEXTORIGIN_JOB_QUEUE_SIZE", 4),
        job_retained_count=_env_int("TEXTORIGIN_JOB_RETAINED_COUNT", 32),
        job_ttl_seconds=_env_int("TEXTORIGIN_JOB_TTL_SECONDS", 600),
        job_memory_bytes=_env_int("TEXTORIGIN_JOB_MEMORY_BYTES", 32 * 1024 * 1024),
        log_level=os.getenv("TEXTORIGIN_LOG_LEVEL", "INFO").upper(),
    )
    if settings.classifier_batch_size < 1:
        raise ValueError("Classifier batch size must be positive")
    if settings.max_concurrent_inferences < 1:
        raise ValueError("TEXTORIGIN_MAX_CONCURRENT_INFERENCES must be positive")
    if not 0 <= settings.minimum_similarity_threshold <= 1:
        raise ValueError("TEXTORIGIN_MIN_SIMILARITY_THRESHOLD must be between 0 and 1")
    if not settings.minimum_similarity_threshold <= settings.default_similarity_threshold <= 1:
        raise ValueError(
            "TEXTORIGIN_SIMILARITY_THRESHOLD must be between the configured minimum and 1"
        )
    if min(settings.job_queue_size, settings.job_retained_count, settings.job_ttl_seconds, settings.job_memory_bytes) < 1:
        raise ValueError("Job limits must be positive")
    if settings.deepseek_timeout_seconds <= 0 or settings.inference_wait_seconds < 0:
        raise ValueError("Timeout must be positive and inference wait nonnegative")
    return settings
