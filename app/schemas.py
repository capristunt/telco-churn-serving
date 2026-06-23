"""Pydantic schemas for API input and output validation."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CustomerProfile(BaseModel):
    """Customer profile to be scored."""

    model_config = ConfigDict(extra="forbid")

    # Numeric
    tenure: int = Field(ge=1, description="Tenure in months (>= 1, tenure=0 excluded at training time).")
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)
    nb_services: int = Field(ge=0, le=8, description="Number of active services (derived feature).")
    SeniorCitizen: Literal[0, 1]

    # Binary categorical
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    PaperlessBilling: Literal["Yes", "No"]

    # Categorical
    MultipleLines: Literal["Yes", "No", "No phone service"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]

    # Multi-class categorical
    InternetService: Literal["DSL", "Fiber optic", "No"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]


class PredictionResponse(BaseModel):
    """Response returned by /predict, mirrors predictor.Prediction."""

    proba_churn: float = Field(ge=0, le=1)
    label_pred: Literal[0, 1]
    risk_segment: Literal["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
    threshold: float