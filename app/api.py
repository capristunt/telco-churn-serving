"""FastAPI entrypoint for the churn scoring service."""

import logging

from fastapi import FastAPI, HTTPException

from app.predictor import predict
from app.schemas import CustomerProfile, PredictionResponse


logger = logging.getLogger(__name__)

app = FastAPI(
    title="Telco Churn Scoring API",
    description="Serve the finetuned churn model via HTTP.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe used by Docker and CI."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(payload: CustomerProfile) -> PredictionResponse:
    try:
        return predict(payload.model_dump())
    except Exception as exc:
        logger.exception("Prediction failed for payload")
        raise HTTPException(status_code=500, detail="Internal scoring error") from exc