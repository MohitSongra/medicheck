"""
MediCheck - Pydantic request / response schemas for the FastAPI layer.
"""

from pydantic import BaseModel, Field
from typing import List


class SymptomRequest(BaseModel):
    """POST body for /predict."""
    symptoms: List[str] = Field(
        ...,
        min_length=1,
        description="List of symptom names the patient is experiencing.",
        json_schema_extra={"examples": [["fever", "cough", "headache"]]},
    )


class DiseaseResult(BaseModel):
    """A single disease prediction result."""
    disease_name: str = Field(..., description="Name of the predicted disease")
    confidence_pct: float = Field(
        ..., ge=0, le=100, description="Confidence percentage (0-100)"
    )
    description: str = Field(..., description="Brief description of the disease")
    suggested_action: str = Field(
        ..., description="Recommended next step for the patient"
    )


class PredictionResponse(BaseModel):
    """Response payload for /predict."""
    predictions: List[DiseaseResult]
    symptoms_analyzed: List[str]
    inference_time_ms: float = Field(
        ..., description="Model inference time in milliseconds"
    )


class HealthResponse(BaseModel):
    """Response for /health."""
    status: str = "ok"
    model_loaded: bool
    diseases_count: int
    symptoms_count: int


class SymptomListResponse(BaseModel):
    """Response for /symptoms."""
    symptoms: List[str]
    count: int
