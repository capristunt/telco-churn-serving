"""Schémas Pydantic pour la validation des entrées et sorties de l'API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CustomerProfile(BaseModel):
    """Profil d'un client à scorer. """

    model_config = ConfigDict(extra="forbid")

    # Numériques
    tenure: int = Field(ge=1, description="Ancienneté en mois (>= 1, tenure=0 exclu à l'entraînement).")
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)
    nb_services: int = Field(ge=0, le=8, description="Nombre de services actifs (feature dérivée).")
    SeniorCitizen: Literal[0, 1]

    # Catégorielles binaires
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    PaperlessBilling: Literal["Yes", "No"]

    # Catégorielles
    MultipleLines: Literal["Yes", "No", "No phone service"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]

    # Catégorielles multi-classes
    InternetService: Literal["DSL", "Fiber optic", "No"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]


class PredictionResponse(BaseModel):
    """Réponse renvoyée par /predict, miroir de predictor.Prediction."""

    proba_churn: float = Field(ge=0, le=1)
    label_pred: Literal[0, 1]
    risk_segment: Literal["Q1 (bas)", "Q2", "Q3", "Q4 (haut)"]
    threshold: float