# ===========================================================
# MediCheck — Multi-Stage Production Dockerfile
# ===========================================================
# Build: docker build -t medicheck .
# Run:   docker run -p 8000:8000 -p 8501:8501 medicheck
# ===========================================================

FROM python:3.13-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (if any native libs are needed)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# --------------- Dependencies layer (cached) ----------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --------------- Application code ---------------------------
COPY . .

# --------------- Health check -------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# --------------- Expose ports -------------------------------
# 8000 = FastAPI  |  8501 = Streamlit
EXPOSE 8000 8501

# --------------- Startup script -----------------------------
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
