# telco-churn-serving

![CI](https://github.com/capristunt/telco-churn-serving/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Docker](https://img.shields.io/badge/docker-ready-2496ed)

Production-style serving layer for a customer churn model — REST API, tests,
container, UI, and CI built around a frozen `.joblib` artifact.

This repository turns the model trained in
[`telco-churn-scoring`](https://github.com/capristunt/telco-churn-scoring)
into a deployable product. The model itself is not modified: everything here
focuses on the layers around it.

![Streamlit UI](docs/screenshot.png)

## Why this repo

`telco-churn-scoring` answers _"I can model"_. This repo answers _"I can deploy"_.

It demonstrates an end-to-end serving stack on a portfolio scale:

- **FastAPI** REST service with input/output validation
- **pytest** suite covering predictor, schemas, and HTTP integration
- **Docker** image with healthcheck
- **Streamlit** UI calling the API over HTTP
- **GitHub Actions** CI running tests on every push
- **Model Card** documenting intended use, metrics, and limitations

## Architecture

Client (HTTP / Streamlit)
│  customer profile (JSON)
▼
FastAPI  /predict
│  Pydantic validation (schemas.py)
▼

Predictor (predictor.py)
│  finetuned.joblib.predict_proba
▼
Response: { proba_churn, label_pred, risk_segment, threshold }

The UI never loads the model directly. It calls the API like any other HTTP
client, keeping a clean separation between presentation and scoring logic.

## Quickstart — Docker (recommended)

```bash
git clone https://github.com/capristunt/telco-churn-serving.git
cd telco-churn-serving

docker build -t telco-churn-serving:v1 .
docker run --rm -d -p 8000:8000 --name telco-api telco-churn-serving:v1
```

Then in a browser:

- `http://localhost:8000/health` → liveness probe
- `http://localhost:8000/docs` → Swagger UI to try `/predict` interactively

To stop the container:

```bash
docker stop telco-api
```

## Quickstart — Local Python

```bash
python -m venv .venv
.venv\Scripts\activate          # on Windows
source .venv/bin/activate       # on macOS/Linux

pip install -r requirements.txt
```

Run the API:

```bash
uvicorn app.api:app --reload --port 8000
```

Run the Streamlit UI (in a second terminal, with the API already running):

```bash
streamlit run ui/streamlit_app.py
```

The UI opens on `http://localhost:8501` and calls the API at
`http://localhost:8000/predict`.

## API reference

### `GET /health`

Returns `{"status": "ok"}`. Used by Docker's healthcheck and CI smoke checks.

### `POST /predict`

Scores one customer profile.

**Request body** (18 fields, all required):

```json
{
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
  "PaymentMethod": "Credit card (automatic)"
}
```

**Response** (200):

```json
{
  "proba_churn": 0.087,
  "label_pred": 0,
  "risk_segment": "Q2",
  "threshold": 0.141
}
```

**Validation errors** return `422` with a Pydantic-style detail describing the
offending field. Categorical values are case-sensitive and restricted to the
modalities seen at training time. Unknown fields are also rejected
(`extra="forbid"`).

## Tests

90 tests across three layers, running in roughly 1.5 seconds locally:

| File | Focus | What it covers |
|---|---|---|
| `tests/test_predictor.py` | Scoring logic | Technical invariants, business archetypes, monotonicities (Contract, tenure, OnlineSecurity), segment boundaries |
| `tests/test_schemas.py` | Pydantic schemas | Categorical whitelists, numeric bounds, missing/extra fields, response contract |
| `tests/test_api.py` | HTTP integration | `/health`, `/predict` happy path, 422 on invalid input, 405 on wrong verb |

Run locally:

```bash
pytest -v
```

The monotonicity tests are worth highlighting: they encode known business
relations from the V1 EDA (e.g. `Two year < One year < Month-to-month` in
predicted probability, all else equal). They act as a runnable specification —
if the artifact is replaced with a model that breaks these relations, the
test fails immediately.

## CI

GitHub Actions runs the test suite on every push and pull request to `main`:

- Ubuntu runner, Python 3.12
- Dependencies cached between runs (~10s install after the first run)
- Status badge at the top of this README

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml) for details and
the [Actions tab](https://github.com/capristunt/telco-churn-serving/actions)
for run history.

## Tech stack

| Layer | Library | Version |
|---|---|---|
| API framework | FastAPI | 0.115 |
| Validation | Pydantic | 2.10 |
| Model | scikit-learn | 1.5.2 |
| Inference | pandas, joblib | 2.2 / 1.5 |
| UI | Streamlit | 1.40 |
| HTTP client (UI → API) | requests | 2.34 |
| Testing | pytest | 8.3 |
| Runtime | Python | 3.12 |
| Container base | `python:3.12-slim` | — |

Exact pinned versions are in [`requirements.txt`](requirements.txt).

## Project structure

telco-churn-serving/
├── app/
│   ├── api.py              FastAPI app: /health and /predict
│   ├── schemas.py          Pydantic models for input/output validation
│   └── predictor.py        Model loading and scoring logic
├── ui/
│   └── streamlit_app.py    UI calling the API over HTTP
├── tests/
│   ├── test_api.py
│   ├── test_schemas.py
│   └── test_predictor.py
├── artifacts/
│   └── finetuned.joblib    Frozen model, copied from V1
├── .github/workflows/
│   └── ci.yml              GitHub Actions workflow
├── Dockerfile
├── .dockerignore
├── conftest.py
├── requirements.txt
├── MODEL_CARD.md
└── README.md

## Limitations and next steps

Documented in detail in [`MODEL_CARD.md`](MODEL_CARD.md). Key points:

- **`nb_services` is requested from the client** rather than computed from the
  individual service fields. A V2.1 improvement would compute it server-side
  in `predictor.py` for a cleaner API contract.
- **Quartile boundaries are static**, computed once on the V1 train+valid set.
  In production they would need to be refreshed periodically or replaced by
  a streaming quantile estimator.
- **No drift monitoring** in this version. Mentioned in the Model Card as a
  natural extension.
- **No authentication, no persistence, no rate limiting** — out of scope for
  this portfolio project.

## Related

- [`telco-churn-scoring`](https://github.com/capristunt/telco-churn-scoring) —
  V1 repository: EDA, modeling, calibration, and threshold tuning that
  produced the `finetuned.joblib` served here.

## License

For portfolio and educational use. Dataset: Telco Customer Churn (Kaggle).