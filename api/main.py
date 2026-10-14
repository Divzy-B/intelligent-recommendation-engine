from __future__ import annotations
import logging
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.service import RecommendationService

logging.basicConfig(level=logging.INFO, format='{"timestamp":"%(asctime)s","level":"%(levelname)s","message":"%(message)s"}')
logger = logging.getLogger("recommendation_api")
app = FastAPI(title="Intelligent Recommendation Engine", version="1.0.0", description="Personalized MovieLens recommendations using neural retrieval and ranking.")


class RecommendationRequest(BaseModel):
    user_id: str = Field(..., examples=["1"])
    k: int = Field(default=10, ge=1, le=100)
    history_item_ids: list[str] | None = Field(default=None, description="Optional session history, ordered oldest to newest.")


class Recommendation(BaseModel):
    item_id: str; title: str; score: float; retrieval_score: float | None = None


@lru_cache
def get_service() -> RecommendationService:
    return RecommendationService()


def service_or_503() -> RecommendationService:
    try: return get_service()
    except RuntimeError as error: raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/health")
def health():
    try:
        service = get_service()
        return {"status": "ok", "model_loaded": True, "retrieval_backend": service.retriever.backend}
    except RuntimeError as error:
        return {"status": "degraded", "model_loaded": False, "detail": str(error)}


@app.get("/recommend/{user_id}", response_model=list[Recommendation])
def recommend_user(user_id: str, k: int = 10):
    logger.info("recommendation request user=%s k=%s", user_id, k)
    try: return service_or_503().recommend(user_id, k)
    except ValueError as error: raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/recommend", response_model=list[Recommendation])
def recommend_post(request: RecommendationRequest):
    logger.info("session recommendation request user=%s k=%s", request.user_id, request.k)
    return service_or_503().recommend(request.user_id, request.k, request.history_item_ids)


@app.get("/similar-items/{item_id}")
def similar_items(item_id: str, k: int = 10):
    try: return service_or_503().similar_items(item_id, k)
    except KeyError as error: raise HTTPException(status_code=404, detail=f"Unknown item_id: {item_id}") from error


@app.get("/model-info")
def model_info(): return service_or_503().model_info()
