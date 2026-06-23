"""Logique de scoring : chargement du modèle et inférence pour un profil client.

Ce module est la seule pièce du V2 qui charge l'artefact .joblib. L'API (api.py)
et l'UI (streamlit_app.py) l'appellent sans connaître les détails du modèle.

Le modèle est chargé une fois à l'import du module (singleton). Le seuil de
décision et les bornes de quartiles sont des constantes figées, calculées sur
train + valid lors de l'entraînement.
"""

from pathlib import Path
from typing import TypedDict

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline


# Chemin vers l'artefact, relatif à la racine du repo
MODEL_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "finetuned.joblib"

# Seuil de décision optimal (gain 105 * TP - 15 * FP), notebook 03 du V1
THRESHOLD = 0.141

# Bornes de quartiles calculées sur train + valid (5626 clients) avec finetuned.joblib
QUARTILE_BOUNDS = (0.040574, 0.197194, 0.395784)

# Colonnes attendues par le modèle, dans cet ordre (cf. data_prep.FEATURES du V1)
FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges", "nb_services",
    "SeniorCitizen", "Partner", "Dependents", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]


class Prediction(TypedDict):
    """Résultat structuré du scoring d'un client."""
    proba_churn: float
    label_pred: int
    risk_segment: str
    threshold: float


def _load_model(path: Path = MODEL_PATH) -> Pipeline:
    """Charge l'artefact .joblib depuis le disque.

    Args:
        path: Chemin vers le fichier .joblib.

    Returns:
        Pipeline scikit-learn fittée, prête à être appelée en predict_proba.

    Raises:
        FileNotFoundError: Si l'artefact est introuvable.
    """
    if not path.exists():
        raise FileNotFoundError(f"Artefact introuvable : {path}")
    return joblib.load(path)


def _assign_segment(proba: float) -> str:
    """Assigne un segment de risque à une proba selon les bornes figées.

    Args:
        proba: Probabilité de churn dans [0, 1].

    Returns:
        Label du segment, de "Q1 (bas)" à "Q4 (haut)".
    """
    q1, q2, q3 = QUARTILE_BOUNDS
    if proba < q1:
        return "Q1 (bas)"
    if proba < q2:
        return "Q2"
    if proba < q3:
        return "Q3"
    return "Q4 (haut)"


# Singleton : modèle chargé une fois à l'import du module
_MODEL: Pipeline = _load_model()


def predict(profile: dict) -> Prediction:
    """Score un profil client.

    Args:
        profile: Dictionnaire avec les clés de FEATURES (validation faite en amont
            par Pydantic dans schemas.py).

    Returns:
        Prédiction structurée avec proba, label binaire, segment et seuil utilisé.
    """
    # Le modèle attend un DataFrame avec les colonnes nommées (ColumnTransformer)
    df = pd.DataFrame([profile], columns=FEATURES)
    proba = float(_MODEL.predict_proba(df)[0, 1])
    return Prediction(
        proba_churn=proba,
        label_pred=int(proba >= THRESHOLD),
        risk_segment=_assign_segment(proba),
        threshold=THRESHOLD,
    )