"""Feature contribution explainer for the churn pipeline.

Extracts signed contributions (coef * standardized value) from the underlying
LogisticRegression estimators inside CalibratedClassifierCV. Coefficients are
averaged across the 5 calibrated sub-models (ensemble=True, cv=5).

Contributions are computed on the logit (pre-calibration). Isotonic calibration
is monotonic, so the ranking of drivers is preserved on the final probability;
only the magnitude in probability space is altered.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

MODEL_PATH = Path("artifacts/finetuned.joblib")


@dataclass(frozen=True)
class Contribution:
    """Signed contribution of a single (transformed) feature to the logit."""

    feature: str           # internal name (cat__InternetService_DSL)
    display_name: str      # human label (Internet Service: DSL)
    category: str          # raw column (InternetService)
    value: float
    contribution: float
    direction: Literal["push", "retain"]


def _load_pipeline() -> Pipeline:
    """Load the fitted scikit-learn Pipeline from disk."""
    return joblib.load(MODEL_PATH)


def _averaged_coefficients(pipeline: Pipeline) -> tuple[np.ndarray, float]:
    """Average coef_ and intercept_ across the 5 calibrated sub-estimators.

    Returns
    -------
    coef : np.ndarray of shape (n_features,)
    intercept : float
    """
    clf = pipeline.named_steps["classifier"]
    coefs = np.stack(
        [cc.estimator.coef_.ravel() for cc in clf.calibrated_classifiers_]
    )
    intercepts = np.array(
        [cc.estimator.intercept_.ravel()[0] for cc in clf.calibrated_classifiers_]
    )
    return coefs.mean(axis=0), float(intercepts.mean())

def _humanize_feature(name: str) -> tuple[str, str]:
    """Convert internal feature name to (display_name, category).

    Returns
    -------
    display_name : human-readable label (e.g. 'Internet Service: DSL')
    category : original raw column name (e.g. 'InternetService')
    """
    if name.startswith("num__"):
        raw = name.removeprefix("num__")
        return raw, raw
    if name.startswith("cat__"):
        body = name.removeprefix("cat__")
        # Split on last underscore to separate column from value
        for col in [
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod", "MultipleLines",
            "SeniorCitizen", "Partner", "Dependents",
        ]:
            if body.startswith(col + "_"):
                value = body.removeprefix(col + "_")
                return f"{col}: {value}", col
        return body, body
    return name, name


def compute_contributions(
    profile: pd.DataFrame,
    top_k: int | None = None,
) -> list[Contribution]:
    """Compute signed feature contributions to the logit for a single profile.

    Parameters
    ----------
    profile : pd.DataFrame
        Single-row DataFrame with the 18 raw input columns expected by the
        pipeline (same schema as for predict_proba).
    top_k : int or None
        If set, return only the top_k contributions by absolute magnitude.

    Returns
    -------
    list of Contribution, sorted by |contribution| descending.
    """
    if len(profile) != 1:
        raise ValueError(f"profile must have exactly 1 row, got {len(profile)}")

    pipeline = _load_pipeline()
    preprocessor = pipeline.named_steps["preprocessor"]

    transformed = preprocessor.transform(profile)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed).ravel()

    feature_names = list(preprocessor.get_feature_names_out())
    coef, _intercept = _averaged_coefficients(pipeline)

    raw = coef * transformed

    contribs = []
    for i, fname in enumerate(feature_names):
        display, category = _humanize_feature(fname)
        contribs.append(Contribution(
            feature=fname,
            display_name=display,
            category=category,
            value=float(transformed[i]),
            contribution=float(raw[i]),
            direction="push" if raw[i] > 0 else "retain",
        ))
    contribs.sort(key=lambda c: abs(c.contribution), reverse=True)
    return contribs[:top_k] if top_k else contribs


if __name__ == "__main__":
    # Smoke test on BASE_PROFILE (matches the one in predictor.py).
    base_profile = pd.DataFrame([{
        "tenure": 24,
        "MonthlyCharges": 70.0,
        "TotalCharges": 1680.0,
        "nb_services": 3,
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Credit card (automatic)",
    }])

    top = compute_contributions(base_profile, top_k=10)
    print(f"{'Display':<40} {'Category':<20} {'Value':>8} {'Contrib':>10} {'Dir':>8}")
    print("-" * 90)
    for c in top:
        print(f"{c.display_name:<40} {c.category:<20} {c.value:>8.3f} {c.contribution:>10.4f} {c.direction:>8}")