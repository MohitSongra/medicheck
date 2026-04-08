"""
MediCheck - FastAPI Application
================================
REST API gateway that receives symptoms and returns ranked disease
predictions from the Bayesian Network inference engine.
"""

import time
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import sys
from pathlib import Path

# Ensure project root is on sys.path so model/ is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.inference import MediCheckInference
from api.schemas import (
    SymptomRequest,
    PredictionResponse,
    DiseaseResult,
    HealthResponse,
    SymptomListResponse,
)
from api.utils import get_disease_details, format_disease_name

# ---------------------------------------------------------------------------
# Global inference engine (loaded once at startup)
# ---------------------------------------------------------------------------
engine: MediCheckInference | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup."""
    global engine
    print("Loading Bayesian Network model ...")
    engine = MediCheckInference()
    print(
        f"  Model ready – {len(engine.get_all_diseases())} diseases, "
        f"{len(engine.get_all_symptoms())} symptoms"
    )
    yield
    engine = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="MediCheck API",
    description=(
        "AI-powered Medical Symptom Checker using Bayesian Network inference. "
        "**For educational purposes only — not a medical diagnosis tool.**"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API and model health."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(
        status="ok",
        model_loaded=True,
        diseases_count=len(engine.get_all_diseases()),
        symptoms_count=len(engine.get_all_symptoms()),
    )


@app.get("/symptoms", response_model=SymptomListResponse, tags=["Reference"])
async def list_symptoms():
    """Return all symptoms the model understands."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    symptoms = engine.get_all_symptoms()
    return SymptomListResponse(symptoms=symptoms, count=len(symptoms))


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_disease(request: SymptomRequest):
    """
    Predict the top-5 most likely diseases given observed symptoms.

    Returns disease name, confidence %, description, and suggested action.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate that at least one symptom is known
    known = [s for s in request.symptoms if s in engine.get_all_symptoms()]
    if not known:
        raise HTTPException(
            status_code=422,
            detail=(
                "None of the provided symptoms are recognized. "
                f"Valid symptoms: {engine.get_all_symptoms()[:10]}..."
            ),
        )

    # Run inference
    predictions, inference_ms = engine.predict(known, top_n=5)

    # Enrich with disease info
    results: List[DiseaseResult] = []
    for pred in predictions:
        desc, action = get_disease_details(pred["disease"])
        results.append(
            DiseaseResult(
                disease_name=format_disease_name(pred["disease"]),
                confidence_pct=pred["confidence_pct"],
                description=desc,
                suggested_action=action,
            )
        )

    return PredictionResponse(
        predictions=results,
        symptoms_analyzed=known,
        inference_time_ms=round(inference_ms, 2),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
