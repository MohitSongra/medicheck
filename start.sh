#!/bin/bash
# ===========================================================
# MediCheck — Container Startup Script (Hardened)
# Starts both FastAPI (backend) and Streamlit (frontend)
# ===========================================================

set -e

echo "🚀 Starting MediCheck services..."

# Start FastAPI backend in the background
echo "  [1/2] Starting FastAPI on port ${APP_PORT:-8000}..."
uvicorn api.main:app \
    --host 0.0.0.0 \
    --port "${APP_PORT:-8000}" \
    --workers 1 &

# Give the API a moment to load the model
sleep 3

# Start Streamlit frontend in the background
echo "  [2/2] Starting Streamlit on port ${STREAMLIT_PORT:-8501}..."
streamlit run frontend/app.py \
    --server.port="${STREAMLIT_PORT:-8501}" \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false &

# Wait for all background processes — clean shutdown on SIGTERM
wait
