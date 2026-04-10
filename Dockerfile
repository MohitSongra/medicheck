# ===========================================================
# MediCheck — Production Dockerfile (Hardened)
# ===========================================================
# Build: docker build -t medicheck .
# Run:   docker run --env-file .env -p 8000:8000 -p 8501:8501 medicheck
# ===========================================================

# ── Stage 1: Builder ──────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ── Stage 2: Production ───────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Install curl for health checks
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user (security best practice)
RUN addgroup --system appgroup && \
    adduser --system appuser --ingroup appgroup

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy only what's needed — never COPY . .
COPY api/ ./api/
COPY model/inference.py ./model/inference.py
COPY model/train.py ./model/train.py
COPY model/__init__.py ./model/__init__.py
COPY frontend/ ./frontend/
COPY data/disease_info.json ./data/disease_info.json
COPY data/disease_symptom.csv ./data/disease_symptom.csv
COPY data/download_kaggle.py ./data/download_kaggle.py
COPY start.sh .
COPY requirements.txt .

# Never copy .env — inject at runtime via --env-file or -e flags
# .env is already in .dockerignore

# Environment
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONPATH=/home/appuser/.local/lib/python3.11/site-packages \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Generate the ML model artifact before dropping privileges
RUN python -m model.train

# Run as non-root user
RUN chmod +x start.sh
USER appuser

EXPOSE 8000 8501

CMD ["./start.sh"]
