"""Tests for Pydantic schemas (schemas.py).

Two layers covered:
- CustomerProfile (input): accepts valid payloads, rejects unknown categories,
  out-of-range numerics, missing or extra fields, wrong types.
- PredictionResponse (output): mirrors the contract returned by predictor.predict.
"""

import pytest
from pydantic import ValidationError

from app.schemas import CustomerProfile, PredictionResponse


# Minimal valid payload reused across tests; individual fields are overridden
# in each test to probe a specific validation rule.
VALID_PAYLOAD = {
    "tenure": 12,
    "MonthlyCharges": 70.0,
    "TotalCharges": 840.0,
    "nb_services": 3,
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "PaperlessBilling": "Yes",
    "MultipleLines": "No",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "InternetService": "DSL",
    "Contract": "One year",
    "PaymentMethod": "Credit card (automatic)",
}


# CustomerProfile: happy path

def test_valid_payload_is_accepted():
    """A complete, well-formed payload instantiates without error."""
    profile = CustomerProfile(**VALID_PAYLOAD)
    assert profile.tenure == 12
    assert profile.Contract == "One year"


def test_model_dump_preserves_field_names():
    """model_dump() keeps the original PascalCase field names expected by the pipeline."""
    profile = CustomerProfile(**VALID_PAYLOAD)
    dumped = profile.model_dump()
    assert "MonthlyCharges" in dumped
    assert "Contract" in dumped
    assert dumped["Contract"] == "One year"


# CustomerProfile: invalid categorical values

@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("Contract", "Three year"),
        ("InternetService", "5G"),
        ("PaymentMethod", "Cash"),
        ("Partner", "Maybe"),
        ("OnlineSecurity", "Sometimes"),
        ("PaperlessBilling", "yes"),  # case-sensitive
    ],
)
def test_unknown_category_is_rejected(field, bad_value):
    """Categorical fields reject any value outside their Literal whitelist."""
    payload = {**VALID_PAYLOAD, field: bad_value}
    with pytest.raises(ValidationError):
        CustomerProfile(**payload)


# CustomerProfile: numeric constraints

@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("tenure", 0),  # tenure must be >= 1
        ("tenure", -5),
        ("MonthlyCharges", -10.0),
        ("TotalCharges", -1.0),
        ("nb_services", -1),
        ("nb_services", 9),  # nb_services capped at 8
        ("SeniorCitizen", 2),  # only 0 or 1 allowed
    ],
)
def test_out_of_range_numeric_is_rejected(field, bad_value):
    """Numeric fields reject values outside their declared bounds."""
    payload = {**VALID_PAYLOAD, field: bad_value}
    with pytest.raises(ValidationError):
        CustomerProfile(**payload)


# CustomerProfile: structural errors

def test_missing_field_is_rejected():
    """A payload missing any required field is rejected."""
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "Contract"}
    with pytest.raises(ValidationError):
        CustomerProfile(**payload)


def test_extra_field_is_rejected():
    """Unknown fields are rejected thanks to extra='forbid'."""
    payload = {**VALID_PAYLOAD, "gender": "Male"}
    with pytest.raises(ValidationError):
        CustomerProfile(**payload)


def test_wrong_type_is_rejected():
    """A field with the wrong type (string where int expected) is rejected."""
    payload = {**VALID_PAYLOAD, "tenure": "twelve"}
    with pytest.raises(ValidationError):
        CustomerProfile(**payload)


# PredictionResponse: output contract

def test_prediction_response_accepts_valid_output():
    """A well-formed predictor output instantiates PredictionResponse."""
    response = PredictionResponse(
        proba_churn=0.65,
        label_pred=1,
        risk_segment="Q4 (high)",
        threshold=0.141,
    )
    assert response.label_pred == 1


def test_prediction_response_rejects_proba_out_of_range():
    """proba_churn must stay within [0, 1]."""
    with pytest.raises(ValidationError):
        PredictionResponse(
            proba_churn=1.5,
            label_pred=1,
            risk_segment="Q4 (high)",
            threshold=0.141,
        )


def test_prediction_response_rejects_unknown_segment():
    """risk_segment must be one of the four declared values."""
    with pytest.raises(ValidationError):
        PredictionResponse(
            proba_churn=0.5,
            label_pred=1,
            risk_segment="Q5",
            threshold=0.141,
        )


def test_prediction_response_rejects_invalid_label():
    """label_pred must be 0 or 1."""
    with pytest.raises(ValidationError):
        PredictionResponse(
            proba_churn=0.5,
            label_pred=2,
            risk_segment="Q4 (haut)",
            threshold=0.141,
        )