"""Integration tests for the HTTP layer (api.py).

Exercises the full stack: HTTP -> Pydantic -> predictor -> JSON response.
The deeper invariants (monotonicities, segment boundaries) are already covered
by test_predictor.py and not duplicated here.
"""

import pytest
from fastapi.testclient import TestClient

from app.api import app


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

HIGH_RISK_PAYLOAD = {
    "tenure": 5, "MonthlyCharges": 95.0, "TotalCharges": 475.0, "nb_services": 4,
    "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "PaperlessBilling": "Yes",
    "MultipleLines": "No", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "InternetService": "Fiber optic",
    "Contract": "Month-to-month", "PaymentMethod": "Electronic check",
}


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Shared TestClient for all tests in this module.

    scope='module' avoids re-instantiating the client (and re-loading the model
    transitively via predictor import) for every test function.
    """
    return TestClient(app)


# /health

def test_health_returns_ok(client):
    """/health responds with 200 and the expected status payload."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# /predict: happy path

def test_predict_returns_200_on_valid_payload(client):
    """A valid payload yields a 200 with the full response contract."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"proba_churn", "label_pred", "risk_segment", "threshold"}


def test_predict_response_types_are_correct(client):
    """Response fields have the expected JSON types."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    body = response.json()
    assert isinstance(body["proba_churn"], float)
    assert isinstance(body["label_pred"], int)
    assert isinstance(body["risk_segment"], str)
    assert isinstance(body["threshold"], float)


def test_predict_high_risk_profile_via_http(client):
    """High-risk archetypal profile yields label=1 through the HTTP layer.

    Sanity check that the model is wired correctly behind the API.
    """
    response = client.post("/predict", json=HIGH_RISK_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["label_pred"] == 1
    assert body["risk_segment"] == "Q4 (high)"


# /predict: validation errors (422)

def test_predict_rejects_unknown_category(client):
    """An unknown categorical value triggers a 422 with field-level detail."""
    payload = {**VALID_PAYLOAD, "Contract": "Three year"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any("Contract" in str(err["loc"]) for err in detail)


def test_predict_rejects_missing_field(client):
    """A missing required field triggers a 422."""
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "tenure"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_extra_field(client):
    """An unknown field triggers a 422 thanks to extra='forbid'."""
    payload = {**VALID_PAYLOAD, "gender": "Male"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_out_of_range_numeric(client):
    """A numeric out of bounds (tenure=0) triggers a 422."""
    payload = {**VALID_PAYLOAD, "tenure": 0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_wrong_type(client):
    """A field with the wrong type triggers a 422."""
    payload = {**VALID_PAYLOAD, "tenure": "twelve"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


# /predict: malformed requests

def test_predict_rejects_empty_body(client):
    """An empty JSON body triggers a 422."""
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_rejects_get_method(client):
    """/predict only accepts POST, GET should return 405."""
    response = client.get("/predict")
    assert response.status_code == 405