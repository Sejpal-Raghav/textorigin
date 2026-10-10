from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from src.api.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    HealthResponse,
    HumanizeRequest,
    HumanizeResponse,
)
from src.api.schemas import CompareRequest, CompareResponse, JobRequest, JobResponse
from src.config import get_settings
from src.services.comparison import ComparisonService
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
        **{key: result[key] for key in ("total_tokens", "window_count", "coverage", "complete", "aggregation") if key in result},
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
        style=payload.style,
        strength=payload.strength,
        api_key=(
            payload.deepseek_api_key.get_secret_value()
            if payload.deepseek_api_key is not None
            else None
        ),
    )


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse()


@router.get("/ready")
def ready(request: Request, validate: bool = False):
    engine = request.app.state.services.classifier
    classifier = engine.readiness(validate=True) if validate else engine.readiness()
    settings = get_settings()
    payload = {
        "status": "ready" if classifier["ready"] else "not_ready",
        "classifier": classifier,
        "optional_capabilities": {
            "local_rewrite": True,
            "deepseek_owner_key_configured": bool(settings.deepseek_api_key),
            "deepseek_byok_supported": True,
        },
    }
    return JSONResponse(payload, status_code=200 if classifier["ready"] else 503)



def dispatch_operation(services, operation, payload, context=None):
    if operation == "classify":
        value = services.classification.classify(
            payload.text, include_signals=payload.include_signals, context=context
        )
        return ClassifyResponse(
            **value["scores"],
            **{k: v for k, v in value.items() if k != "scores"},
        ).model_dump()
    kwargs = dict(style=payload.style, strength=payload.strength,
                  use_llm=payload.use_llm,
                  api_key=payload.deepseek_api_key.get_secret_value() if payload.deepseek_api_key else None,
                  context=context, similarity_threshold=payload.similarity_threshold, seed=payload.seed)
    if operation == "humanize":
        return HumanizeResponse(**services.humanization.humanize(payload.text, **kwargs)).model_dump()
    service = getattr(services, "comparison", None) or ComparisonService(services.classification, services.humanization)
    return CompareResponse(**service.compare(payload.text, **kwargs)).model_dump()


@router.post("/compare", response_model=CompareResponse)
def compare(payload: CompareRequest, request: Request):
    return dispatch_operation(request.app.state.services, "compare", payload)


@router.post("/jobs", response_model=JobResponse, status_code=202)
def submit_job(payload: JobRequest, request: Request):
    return request.app.state.jobs.submit(payload.operation, payload)


@router.get("/jobs/{job_id}", response_model=JobResponse)
def job_status(job_id: str, request: Request):
    try:
        return request.app.state.jobs.get(job_id)
    except KeyError:
        return JSONResponse({"error": {"code": "job_not_found", "detail": "Job not found or expired"}}, status_code=404)


@router.post("/jobs/{job_id}/cancel", response_model=JobResponse)
def cancel_job(job_id: str, request: Request):
    try:
        return request.app.state.jobs.cancel(job_id)
    except KeyError:
        return JSONResponse({"error": {"code": "job_not_found", "detail": "Job not found or expired"}}, status_code=404)
