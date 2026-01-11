import time
import uvicorn


from fastapi import FastAPI
from fastapi import Request
from contextlib import asynccontextmanager

from src.app.core.config import get_settings
from src.app.routes.agent import router as agent_router
from src.app.routes.health import router as health_router
from src.app.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__, log_level=settings.LOG_LEVEL, log_file=f"{settings.APP_NAME}.log")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Service starting up")
    try:
        yield
    finally:
        logger.info("Service shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version="1.0.0",
        lifespan=lifespan
    )

    # ---------------------
    # Include routers
    # ---------------------
    app.include_router(agent_router)
    app.include_router(health_router)
    
    # ---- Middleware: basic request logging ----
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.time()
        try:
            response = await call_next(request)
            return response
        finally:
            duration_ms = int((time.time() - start) * 1000)
            logger.info(
                "%s %s -> %s (%dms)",
                request.method,
                request.url.path,
                getattr(locals().get("response", None), "status_code", "NA"),
                duration_ms,
            )
    
    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.APP_ENV.lower() != "prod",
    )