from fastapi import APIRouter

router = APIRouter()

@router.get("/health", tags=["Health"])
def health_check() -> dict:
    return {"status": "ok"}