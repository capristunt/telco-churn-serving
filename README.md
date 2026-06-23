# telco-churn-serving

Industrialisation du modèle de scoring de churn produit dans
[`telco-churn-scoring`](https://github.com/capristunt/telco-churn-scoring).

Ce repo transforme l'artefact `finetuned.joblib` en produit servable :
API REST (FastAPI), tests (pytest), conteneur (Docker), interface (Streamlit),
CI (GitHub Actions), documentation (Model Card).

## Statut

En construction. Voir le cahier des charges et le planning J1-J10.

## Architecture

telco-churn-serving/
├── app/                  # API FastAPI
├── ui/                   # Interface Streamlit
├── tests/                # Tests pytest
├── artifacts/            # Modèle sérialisé (joblib)
├── Dockerfile            # Conteneurisation
├── MODEL_CARD.md         # Documentation du modèle
└── .github/workflows/    # CI

## Reproductibilité

À venir.