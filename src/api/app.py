from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import router
from src.bootstrap import build_services
from src.config import get_settings
from src.errors import InferenceCapacityError, ModelUnavailableError


def create_app(services=None) -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(
            level=getattr(logging, settings.log_level, logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        app.state.services = services or build_services(settings)
        readiness = app.state.services.classifier.readiness()
        if not readiness["ready"]:
            logging.getLogger(__name__).warning(
                "classifier_not_ready details=%s", readiness
            )
        try:
            yield
        finally:
            app.state.services.close()

    application = FastAPI(
        title="TextOrigin",
        description="Classifies text and reports separate heuristic writing signals.",
        version="1.1.0",
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
