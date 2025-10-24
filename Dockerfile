# MLOps Framework - Production Dockerfile
# This demonstrates containerizing an ML platform for cloud deployment

# Base image with Python 3.9
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build tools for Python packages
    gcc \
    g++ \
    make \
    # Git for version control integration
    git \
    # Database clients
    libpq-dev \
    # Clean up to reduce image size
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching optimization)
# If requirements.txt doesn't change, this layer is cached
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for MLflow and FEAST data
RUN mkdir -p /app/mlruns \
    /app/feast_data \
    /app/logs \
    /app/monitoring/reports

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=sqlite:///app/mlruns/mlflow.db \
    FEAST_REPO_PATH=/app/src/feature_store

# Expose ports
# 5000: MLflow UI
# 6566: FEAST feature server
# 8501: Streamlit dashboard (optional)
EXPOSE 5000 6566 8501

# Health check endpoint (verify container is running)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health')"

# Default command: Start MLflow server
# Override this in docker-compose.yml or Kubernetes for different services
CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:///app/mlruns/mlflow.db", \
     "--default-artifact-root", "/app/mlruns", \
     "--host", "0.0.0.0", \
     "--port", "5000"]


# ============================================================================
# Usage Examples
# ============================================================================

# Build image
# docker build -t mlops-framework:latest .

# Run MLflow server
# docker run -p 5000:5000 -v $(pwd)/mlruns:/app/mlruns mlops-framework:latest

# Run training job (override CMD)
# docker run mlops-framework:latest python src/models/heart_disease/train.py

# Run inference server
# docker run -p 8000:8000 mlops-framework:latest \
#   uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Interactive shell for debugging
# docker run -it mlops-framework:latest /bin/bash


# ============================================================================
# Docker Compose Example (Multi-Service Deployment)
# ============================================================================

# docker-compose.yml:
#
# version: '3.8'
# services:
#   mlflow:
#     build: .
#     ports:
#       - "5000:5000"
#     volumes:
#       - ./mlruns:/app/mlruns
#     command: mlflow server --backend-store-uri sqlite:///app/mlruns/mlflow.db
#
#   feast:
#     build: .
#     ports:
#       - "6566:6566"
#     volumes:
#       - ./feast_data:/app/feast_data
#     command: feast serve --host 0.0.0.0
#
#   inference:
#     build: .
#     ports:
#       - "8000:8000"
#     depends_on:
#       - mlflow
#       - feast
#     command: uvicorn src.api.main:app --host 0.0.0.0
#
#   dashboard:
#     build: .
#     ports:
#       - "8501:8501"
#     depends_on:
#       - mlflow
#     command: streamlit run src/dashboard/app.py


