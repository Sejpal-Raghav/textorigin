from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import router, dispatch_operation
from src.services.jobs import JobManager
from src.config import get_settings
from src.errors import InferenceCapacityError, ModelUnavailableError, JobCapacityError


def create_app(services=None) -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(
            level=getattr(logging, settings.log_level, logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        if services is None:
            from src.bootstrap import build_services

            app.state.services = build_services(settings)
        else:
            app.state.services = services
        readiness = app.state.services.classifier.readiness()
        if not readiness["ready"]:
            logging.getLogger(__name__).warning(
                "classifier_not_ready details=%s", readiness
            )
        app.state.jobs = JobManager(
            lambda operation, payload, context: dispatch_operation(app.state.services, operation, payload, context),
            workers=1, max_queued=settings.job_queue_size,
            max_retained=settings.job_retained_count, ttl_seconds=settings.job_ttl_seconds,
            max_payload_bytes=settings.job_memory_bytes,
        )
        try:
            yield
        finally:
            app.state.jobs.close(timeout=settings.deepseek_timeout_seconds + 5)
            app.state.services.close()

    application = FastAPI(
        title="TextOrigin",
        description="Classifies text and reports separate heuristic writing signals.",
        version="1.2.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    application.include_router(router)

    @application.exception_handler(ModelUnavailableError)
    def model_unavailable(_request: Request, exc: ModelUnavailableError):
        return JSONResponse(
            {"error": {"code": "model_unavailable", "detail": str(exc)}},
            status_code=503,
        )

    @application.exception_handler(InferenceCapacityError)
    def capacity_exhausted(_request: Request, exc: InferenceCapacityError):
        return JSONResponse(
            {"error": {"code": "inference_capacity_exhausted", "detail": str(exc)}},
            status_code=503,
            headers={"Retry-After": "5"},
        )

    @application.exception_handler(JobCapacityError)
    def job_capacity(_request: Request, _exc: JobCapacityError):
        return JSONResponse({"error": {"code": "job_capacity_exhausted", "detail": "Local queue or transient memory budget exhausted"}}, status_code=503, headers={"Retry-After": "5"})

    @application.exception_handler(RequestValidationError)
    def validation_error(_request: Request, exc: RequestValidationError):
        detail = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        return JSONResponse(
            {"error": {"code": "validation_error", "detail": detail}},
            status_code=422,
        )

    return application


app = create_app()
