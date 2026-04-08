"""
MediCheck - API Endpoint Tests
================================
Tests the FastAPI endpoints using the built-in TestClient.

Updated for Kaggle dataset: 42 diseases, 132 symptoms.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.main import app


@pytest.fixture(scope="module")
def client():
    """Create a TestClient for the FastAPI app."""
    with TestClient(app) as c:
        yield c


# -----------------------------------------------------------------------
# Health endpoint
# -----------------------------------------------------------------------
class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_model_loaded(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] is True

    def test_health_counts(self, client):
        data = client.get("/health").json()
        assert data["diseases_count"] >= 40
        assert data["symptoms_count"] >= 130


# -----------------------------------------------------------------------
# Symptoms endpoint
# -----------------------------------------------------------------------
class TestSymptomsEndpoint:
    def test_symptoms_returns_200(self, client):
        resp = client.get("/symptoms")
        assert resp.status_code == 200

    def test_symptoms_list(self, client):
        data = client.get("/symptoms").json()
        assert data["count"] >= 130
        assert "itching" in data["symptoms"]


# -----------------------------------------------------------------------
# Predict endpoint
# -----------------------------------------------------------------------
class TestPredictEndpoint:
    def test_predict_valid(self, client):
        resp = client.post(
            "/predict",
            json={"symptoms": ["itching", "skin_rash", "high_fever"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["predictions"]) > 0
        assert len(data["predictions"]) <= 5

    def test_predict_response_schema(self, client):
        resp = client.post(
            "/predict",
            json={"symptoms": ["high_fever", "fatigue"]},
        )
        data = resp.json()
        pred = data["predictions"][0]
        assert "disease_name" in pred
        assert "confidence_pct" in pred
        assert "description" in pred
        assert "suggested_action" in pred
        assert "inference_time_ms" in data

    def test_predict_contains_symptoms_analyzed(self, client):
        symptoms = ["high_fever", "chills", "sweating"]
        resp = client.post("/predict", json={"symptoms": symptoms})
        data = resp.json()
        assert set(data["symptoms_analyzed"]) == set(symptoms)

    def test_predict_empty_body_fails(self, client):
        resp = client.post("/predict", json={"symptoms": []})
        assert resp.status_code == 422

    def test_predict_unknown_symptoms(self, client):
        resp = client.post(
            "/predict",
            json={"symptoms": ["not_a_real_symptom"]},
        )
        assert resp.status_code == 422

    def test_predict_under_2000ms(self, client):
        """API response should be under 2000ms for the larger model."""
        resp = client.post(
            "/predict",
            json={"symptoms": ["high_fever", "cough", "headache", "fatigue"]},
        )
        data = resp.json()
        assert data["inference_time_ms"] < 2000

    def test_predict_confidence_ordering(self, client):
        """Results should be ordered by confidence descending."""
        resp = client.post(
            "/predict",
            json={"symptoms": ["high_fever", "nausea", "vomiting"]},
        )
        pcts = [p["confidence_pct"] for p in resp.json()["predictions"]]
        assert pcts == sorted(pcts, reverse=True)
