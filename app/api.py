"""FastAPI entrypoint for the churn scoring service."""

from fastapi import FastAPI

app = FastAPI(
    title="Telco Churn Scoring API",
    description="Serve the finetuned churn model via HTTP.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe used by Docker and CI."""
    return {"status": "ok"}