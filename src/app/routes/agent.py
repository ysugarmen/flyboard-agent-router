import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.utils.logger import get_logger
from app.schemas.agent import RunRequest, RunResponse
from agent.runner import run_task
from agent.runner import UpstreamModelError, ToolIterationLimitError


logger = get_logger("routes.agent")

router = APIRouter(
    prefix="/v1/agent",
    tags=["agent"],
)

@router.post("/run", response_model=RunResponse)
def run_agent(request: RunRequest) -> RunResponse:
    """
    Endpoint to run the agent with the provided task and parameters.
    """

    start_time = time.time()
    try:
        response = run_task(
            task=request.task,
            customer_id=request.customer_id,
            language=request.language
        )

        logger.info(f"Task {request.task} completed in {time.time() - start_time:.2f} seconds.")
        return response
    
    except UpstreamModelError as e:
        logger.error(f"Upstream model error trace_id {e.trace_id}")
        return JSONResponse(
            status_code=502,
            content={
                "trace_id": e.trace_id,
                "error": "upstream_error",
                "message": str(e),
            },
        )
    
    except ToolIterationLimitError as e:
        logger.warning("Tool iteration limit exceeded trace_id=%s max_iterations=%d", e.trace_id, e.max_iterations)
        return JSONResponse(
            status_code=500,
            content={
                "trace_id": e.trace_id,
                "error": "tool_iteration_limit",
                "message": str(e),
                "max_iterations": e.max_iterations,
            },
        )

    except Exception as e:
        logger.exception("Unexpected error occurred while running agent.")
        raise HTTPException(status_code=500, detail="Internal server error") from e
    