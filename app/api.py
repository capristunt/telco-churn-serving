"""FastAPI entrypoint for the churn scoring service."""

import logging
import pandas as pd

from fastapi import FastAPI, HTTPException

from app.predictor import predict
from app.explainer import compute_contributions
from app.schemas import CustomerProfile, PredictionResponse, ExplainResponse, ContributionItem


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
    
@app.post("/explain", response_model=ExplainResponse)
def explain_endpoint(payload: CustomerProfile) -> ExplainResponse:
    """Score the customer and return top feature contributions.

    Same prediction as /predict, enriched with the top 10 signed contributions
    of input features to the pre-calibration logit. Useful for UI-facing
    clients that need to display why a customer is at risk.
    """
    try:
        prediction = predict(payload.model_dump())
        profile_df = pd.DataFrame([payload.model_dump()])
        contributions = compute_contributions(profile_df, top_k=10)
    except Exception as exc:
        logger.exception("Explain failed for payload")
        raise HTTPException(status_code=500, detail="Internal scoring error") from exc

    return ExplainResponse(
        **prediction.model_dump() if hasattr(prediction, "model_dump") else prediction,
        contributions=[
            ContributionItem(
                feature=c.feature,
                display_name=c.display_name,
                category=c.category,
                value=c.value,
                contribution=c.contribution,
                direction=c.direction,
            )
            for c in contributions
        ],
    )