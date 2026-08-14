from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from src.api.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    HealthResponse,
    HumanizeRequest,
    HumanizeResponse,
)
from src.config import get_settings
from src.services.classification import ClassificationService
from src.services.humanization import HumanizationService

router = APIRouter()


def get_classification_service(request: Request) -> ClassificationService:
    return request.app.state.services.classification


def get_humanization_service(request: Request) -> HumanizationService:
    return request.app.state.services.humanization


@router.post("/classify", response_model=ClassifyResponse)
def classify(
    payload: ClassifyRequest,
    service: ClassificationService = Depends(get_classification_service),
):
    include_signals = payload.include_signals
    if include_signals is None:
        include_signals = get_settings().include_signals_by_default
    result = service.classify(payload.text, include_signals=include_signals)
    return ClassifyResponse(
        **result["scores"],
        observed_signals=result["observed_signals"],
        truncated=result["truncated"],
        analyzed_tokens=result["analyzed_tokens"],
        model_version=result["model_version"],
        warnings=result["warnings"],
    )


@router.post("/humanize", response_model=HumanizeResponse)
def humanize(
    payload: HumanizeRequest,
    service: HumanizationService = Depends(get_humanization_service),
):
    return service.humanize(
        payload.text,
        use_llm=payload.use_llm,
        similarity_threshold=payload.similarity_threshold,
        seed=payload.seed,
    )


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse()


@router.get("/ready")
def ready(request: Request):
    classifier = request.app.state.services.classifier.readiness()
    payload = {
        "status": "ready" if classifier["ready"] else "not_ready",
        "classifier": classifier,
        "optional_capabilities": {"ollama_configured": True},
    }
    return JSONResponse(payload, status_code=200 if classifier["ready"] else 503)
