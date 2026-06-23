"""Tests for the scoring logic (predictor.py).

Three families of tests:
- Technical invariants: output structure, proba in [0,1], label/threshold
  consistency, checked on a diverse set of profiles.
- Business tests: archetypal profiles and known monotonicities (Contract and
  tenure), aligned with the drivers identified during EDA.
- Utility function: _assign_segment on representative values.
"""

import copy

import pytest

from app.predictor import THRESHOLD, _assign_segment, predict


# Neutral reference profile, used as a baseline for monotonicity tests.
# Internet-dependent options are kept inactive to stay consistent with the
# DSL internet service.
BASE_PROFILE = {
    "tenure": 24,
    "MonthlyCharges": 70.0,
    "TotalCharges": 1680.0,
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


HIGH_RISK_PROFILE = {
    "tenure": 5, "MonthlyCharges": 95.0, "TotalCharges": 475.0, "nb_services": 4,
    "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "PaperlessBilling": "Yes",
    "MultipleLines": "No", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "InternetService": "Fiber optic",
    "Contract": "Month-to-month", "PaymentMethod": "Electronic check",
}

LOW_RISK_PROFILE = {
    "tenure": 60, "MonthlyCharges": 65.0, "TotalCharges": 3900.0, "nb_services": 6,
    "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes", "PaperlessBilling": "No",
    "MultipleLines": "Yes", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
    "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
    "StreamingMovies": "No", "InternetService": "DSL", "Contract": "Two year",
    "PaymentMethod": "Bank transfer (automatic)",
}

MID_RISK_PROFILE = {
    "tenure": 18, "MonthlyCharges": 80.0, "TotalCharges": 1440.0, "nb_services": 3,
    "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "PaperlessBilling": "Yes",
    "MultipleLines": "Yes", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "No", "InternetService": "Fiber optic",
    "Contract": "One year", "PaymentMethod": "Mailed check",
}

# Edge case: long tenure paired with a month-to-month contract (mixed signal)
LONG_TENURE_MONTHLY = {
    "tenure": 65, "MonthlyCharges": 85.0, "TotalCharges": 5525.0, "nb_services": 4,
    "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "PaperlessBilling": "Yes",
    "MultipleLines": "Yes", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "Yes", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "InternetService": "Fiber optic",
    "Contract": "Month-to-month", "PaymentMethod": "Credit card (automatic)",
}

# Edge case: senior with no internet service (less frequent in training)
SENIOR_NO_INTERNET = {
    "tenure": 30, "MonthlyCharges": 20.0, "TotalCharges": 600.0, "nb_services": 1,
    "SeniorCitizen": 1, "Partner": "No", "Dependents": "No", "PaperlessBilling": "No",
    "MultipleLines": "No", "OnlineSecurity": "No internet service",
    "OnlineBackup": "No internet service", "DeviceProtection": "No internet service",
    "TechSupport": "No internet service", "StreamingTV": "No internet service",
    "StreamingMovies": "No internet service", "InternetService": "No",
    "Contract": "Two year", "PaymentMethod": "Mailed check",
}

# Edge case: brand-new customer on a heavy plan
NEW_HEAVY_USER = {
    "tenure": 2, "MonthlyCharges": 105.0, "TotalCharges": 210.0, "nb_services": 5,
    "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "PaperlessBilling": "Yes",
    "MultipleLines": "Yes", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "Yes", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "InternetService": "Fiber optic",
    "Contract": "Month-to-month", "PaymentMethod": "Electronic check",
}

# Edge case: loyal long-tenure customer with paperless billing + electronic check
LOYAL_PAPERLESS = {
    "tenure": 70, "MonthlyCharges": 55.0, "TotalCharges": 3850.0, "nb_services": 4,
    "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes", "PaperlessBilling": "Yes",
    "MultipleLines": "No", "OnlineSecurity": "Yes", "OnlineBackup": "No",
    "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
    "StreamingMovies": "No", "InternetService": "DSL",
    "Contract": "Two year", "PaymentMethod": "Electronic check",
}

ALL_PROFILES = [
    BASE_PROFILE, HIGH_RISK_PROFILE, LOW_RISK_PROFILE, MID_RISK_PROFILE,
    LONG_TENURE_MONTHLY, SENIOR_NO_INTERNET, NEW_HEAVY_USER, LOYAL_PAPERLESS,
]


# Technical invariants across all profiles

@pytest.mark.parametrize("profile", ALL_PROFILES)
def test_predict_returns_expected_keys(profile):
    """Output always contains the four contract keys."""
    result = predict(profile)
    assert set(result.keys()) == {"proba_churn", "label_pred", "risk_segment", "threshold"}


@pytest.mark.parametrize("profile", ALL_PROFILES)
def test_predict_proba_in_unit_interval(profile):
    """Probability stays within [0, 1] for any profile."""
    result = predict(profile)
    assert 0.0 <= result["proba_churn"] <= 1.0


@pytest.mark.parametrize("profile", ALL_PROFILES)
def test_predict_label_consistent_with_threshold(profile):
    """label_pred is 1 iff proba >= threshold, across all profiles."""
    result = predict(profile)
    expected = int(result["proba_churn"] >= result["threshold"])
    assert result["label_pred"] == expected


@pytest.mark.parametrize("profile", ALL_PROFILES)
def test_predict_threshold_is_module_constant(profile):
    """Returned threshold always matches the module constant."""
    result = predict(profile)
    assert result["threshold"] == THRESHOLD


@pytest.mark.parametrize("profile", ALL_PROFILES)
def test_predict_segment_consistent_with_proba(profile):
    """Returned segment is consistent with the proba via _assign_segment."""
    result = predict(profile)
    assert result["risk_segment"] == _assign_segment(result["proba_churn"])


# Business tests on archetypal profiles

def test_high_risk_profile_classified_as_churn():
    """Archetypal high-risk profile must yield label=1 and segment Q4."""
    result = predict(HIGH_RISK_PROFILE)
    assert result["label_pred"] == 1
    assert result["risk_segment"] == "Q4 (high)"


def test_low_risk_profile_classified_as_no_churn():
    """Archetypal low-risk profile must yield label=0 and segment Q1."""
    result = predict(LOW_RISK_PROFILE)
    assert result["label_pred"] == 0
    assert result["risk_segment"] == "Q1 (low)"


def test_high_risk_proba_greater_than_low_risk():
    """Discrimination: high-risk proba should exceed low-risk by a wide margin."""
    high = predict(HIGH_RISK_PROFILE)["proba_churn"]
    low = predict(LOW_RISK_PROFILE)["proba_churn"]
    assert high > low + 0.5


# Monotonicity tests: known business relations

def test_contract_monotonicity():
    """All else equal, Two year < One year < Month-to-month in proba.

    Reflects the strongest categorical driver of the model (Contract, V=0.41).
    """
    profile_2y = copy.deepcopy(BASE_PROFILE)
    profile_2y["Contract"] = "Two year"
    profile_1y = copy.deepcopy(BASE_PROFILE)
    profile_1y["Contract"] = "One year"
    profile_mtm = copy.deepcopy(BASE_PROFILE)
    profile_mtm["Contract"] = "Month-to-month"

    p_2y = predict(profile_2y)["proba_churn"]
    p_1y = predict(profile_1y)["proba_churn"]
    p_mtm = predict(profile_mtm)["proba_churn"]

    assert p_2y < p_1y < p_mtm


def test_tenure_monotonicity():
    """All else equal, increasing tenure lowers the churn proba.

    Reflects tenure being the dominant feature by permutation importance (0.177).
    """
    short = copy.deepcopy(BASE_PROFILE)
    short["tenure"] = 3
    long_ = copy.deepcopy(BASE_PROFILE)
    long_["tenure"] = 60

    assert predict(short)["proba_churn"] > predict(long_)["proba_churn"]


def test_online_security_reduces_churn():
    """OnlineSecurity=Yes should reduce the proba (retention driver)."""
    without = copy.deepcopy(BASE_PROFILE)
    without["OnlineSecurity"] = "No"
    with_security = copy.deepcopy(BASE_PROFILE)
    with_security["OnlineSecurity"] = "Yes"

    assert predict(with_security)["proba_churn"] < predict(without)["proba_churn"]


# Utility function _assign_segment

@pytest.mark.parametrize(
    "proba, expected",
    [
        (0.0, "Q1 (low)"),
        (0.039, "Q1 (low)"),
        (0.041, "Q2"),
        (0.10, "Q2"),
        (0.196, "Q2"),
        (0.198, "Q3"),
        (0.30, "Q3"),
        (0.394, "Q3"),
        (0.396, "Q4 (high)"),
        (0.80, "Q4 (high)"),
        (1.0, "Q4 (high)"),
    ],
)
def test_assign_segment_boundaries(proba, expected):
    """Quartile boundaries, including values just below and above each bound."""
    assert _assign_segment(proba) == expected