from fastapi import APIRouter

from app.api.routes.v1 import v1_router
from app.api.fast_api_predict import router as predict_router   # ← this line
from app.config import settings

head_router = APIRouter()
head_router.include_router(v1_router, prefix=settings.api_v1)
head_router.include_router(predict_router, prefix="/ai", tags=["AI: Recovery"])  # ← this line

__all__ = [
    "head_router",
]