# Base: official Python slim image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install Python dependencies first (cached as long as requirements.txt doesn't change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and the model artifact
COPY app/ ./app/
COPY artifacts/ ./artifacts/

# Document the port the application listens on
EXPOSE 8000

# Healthcheck so Docker knows whether the API is alive
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

# Start the API server, listening on all interfaces
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]