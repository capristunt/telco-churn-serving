"""Scoring logic: model loading and inference for a single customer profile.

This module is the only piece of the V2 codebase that loads the .joblib
artifact. Both the API (api.py) and the UI (streamlit_app.py) call into it
without knowing any of the model details.

The model is loaded once at module import (singleton). The decision threshold
and quartile boundaries are frozen constants, computed on train + valid during
training.
"""

from pathlib import Path
from typing import TypedDict

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline


# Path to the artifact, relative to the repo root
MODEL_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "finetuned.joblib"

# Optimal decision threshold (gain 105 * TP - 15 * FP), V1 notebook 03
THRESHOLD = 0.141

# Quartile boundaries computed on train + valid (5626 customers) with finetuned.joblib
QUARTILE_BOUNDS = (0.040574, 0.197194, 0.395784)

# Columns expected by the model, in this order (cf. data_prep.FEATURES in V1)
FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges", "nb_services",
    "SeniorCitizen", "Partner", "Dependents", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]


class Prediction(TypedDict):
    """Structured output of the scoring function."""
    proba_churn: float
    label_pred: int
    risk_segment: str
    threshold: float


def _load_model(path: Path = MODEL_PATH) -> Pipeline:
    """Load the .joblib artifact from disk.

    Args:
        path: Path to the .joblib file.

    Returns:
        Fitted scikit-learn Pipeline, ready for predict_proba calls.

    Raises:
        FileNotFoundError: If the artifact cannot be found.
    """
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return joblib.load(path)


def _assign_segment(proba: float) -> str:
    """Assign a risk segment to a probability based on frozen quartile bounds.

    Args:
        proba: Churn probability in [0, 1].

    Returns:
        Segment label, from "Q1 (low)" to "Q4 (high)".
    """
    q1, q2, q3 = QUARTILE_BOUNDS
    if proba < q1:
        return "Q1 (low)"
    if proba < q2:
        return "Q2"
    if proba < q3:
        return "Q3"
    return "Q4 (high)"


# Singleton: model loaded once at module import
_MODEL: Pipeline = _load_model()


def predict(profile: dict) -> Prediction:
    """Score a customer profile.

    Args:
        profile: Dictionary with the keys listed in FEATURES (validation is
            performed upstream by Pydantic in schemas.py).

    Returns:
        Structured prediction with proba, binary label, segment, and threshold.
    """
    # The model expects a DataFrame with named columns (ColumnTransformer)
    df = pd.DataFrame([profile], columns=FEATURES)
    proba = float(_MODEL.predict_proba(df)[0, 1])
    return Prediction(
        proba_churn=proba,
        label_pred=int(proba >= THRESHOLD),
        risk_segment=_assign_segment(proba),
        threshold=THRESHOLD,
    )