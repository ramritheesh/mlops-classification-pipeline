# syntax=docker/dockerfile:1
# Multi-stage production Dockerfile for Breast Cancer Classification MLOps API

FROM python:3.11-slim as base

# Set environment variables for Python in container
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Set working directory
WORKDIR /app

# Install system dependencies (curl for container debugging if needed, clean up apt caches)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root user and group
RUN groupadd -r appgroup && useradd -r -g appgroup -u 1000 appuser

# Install Python dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and required model artifacts
COPY --chown=appuser:appgroup app/ ./app/
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup models/model.joblib ./models/model.joblib
COPY --chown=appuser:appgroup models/metadata.json ./models/metadata.json

# Create writable local logs directory for prediction logging
RUN mkdir -p logs && chown -R appuser:appgroup logs

# Switch to non-privileged user
USER appuser

# Expose API port
EXPOSE 8000

# Define container health check using the FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()" || exit 1

# Start Uvicorn ASGI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
