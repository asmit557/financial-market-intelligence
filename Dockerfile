# ============================================================
# STAGE 1: Builder - Install dependencies and build FAISS index
# ============================================================
FROM python:3.12-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ============================================================
# STAGE 2: Runtime - Lean production image
# ============================================================
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Build FAISS index during image build
RUN python ingest.py

# Expose both ports
EXPOSE 8000
EXPOSE 8501

# Health check for deployment validation
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/')" || exit 1

# Start both services using bash -c
CMD bash -c "uvicorn main:app --host 0.0.0.0 --port 8000 & sleep 5 && streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0"
