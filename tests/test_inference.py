"""
MediCheck - Inference Module Tests
====================================
Validates that the BN model loads correctly, produces predictions,
and meets performance requirements.

Updated for Kaggle dataset: 42 diseases, 132 symptoms, 10k+ rows.
"""

import time
import sys
from pathlib import Path

import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.inference import MediCheckInference


@pytest.fixture(scope="module")
def engine():
    """Load the inference engine once for all tests."""
    return MediCheckInference()


# -----------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------
class TestModelLoading:
    def test_model_loads_successfully(self, engine):
        """The BN model loads without exceptions."""
        assert engine.model is not None

    def test_model_has_disease_node(self, engine):
        """The model contains a 'disease' node."""
        assert "disease" in engine.model.nodes()

    def test_model_has_symptoms(self, engine):
        """The model has at least 130 symptom nodes (Kaggle dataset)."""
        assert len(engine.get_all_symptoms()) >= 130

    def test_model_has_diseases(self, engine):
        """The model knows about at least 40 diseases (Kaggle dataset)."""
        assert len(engine.get_all_diseases()) >= 40

    def test_model_is_valid(self, engine):
        """pgmpy internal model check passes."""
        assert engine.model.check_model()


# -----------------------------------------------------------------------
# Prediction correctness
# -----------------------------------------------------------------------
class TestPredictions:
    def test_predict_returns_results(self, engine):
        """Predict returns a non-empty list for valid symptoms."""
        preds, _ = engine.predict(["itching", "skin_rash", "nodal_skin_eruptions"])
        assert len(preds) > 0

    def test_predict_top5(self, engine):
        """Predict returns at most 5 results by default."""
        preds, _ = engine.predict(["high_fever", "chills"])
        assert len(preds) <= 5

    def test_predict_probabilities_sum(self, engine):
        """All returned probabilities are between 0 and 1."""
        preds, _ = engine.predict(["high_fever", "sweating", "vomiting"])
        for p in preds:
            assert 0 <= p["probability"] <= 1

    def test_predict_confidence_pct(self, engine):
        """confidence_pct == probability * 100 (rounded to 2dp)."""
        preds, _ = engine.predict(["headache", "nausea"])
        for p in preds:
            assert p["confidence_pct"] == round(p["probability"] * 100, 2)

    def test_predict_empty_symptoms(self, engine):
        """Empty symptom list returns nothing."""
        preds, _ = engine.predict([])
        assert preds == []

    def test_predict_invalid_symptoms(self, engine):
        """Unknown symptoms are ignored and produce no results."""
        preds, _ = engine.predict(["made_up_symptom"])
        assert preds == []

    def test_predict_fungal_symptoms(self, engine):
        """Fungal infection symptoms should rank fungal diseases highly."""
        preds, _ = engine.predict(
            ["itching", "skin_rash", "nodal_skin_eruptions", "dischromic_patches"]
        )
        top_diseases = [p["disease"] for p in preds[:3]]
        fungal_related = {"Fungal infection", "Psoriasis", "Acne"}
        assert any(d in fungal_related for d in top_diseases)

    def test_predict_respiratory_symptoms(self, engine):
        """Respiratory symptoms should surface respiratory diseases."""
        preds, _ = engine.predict(
            ["cough", "breathlessness", "high_fever", "phlegm"]
        )
        top_diseases = [p["disease"] for p in preds[:3]]
        respiratory = {"Bronchial Asthma", "Pneumonia", "Common Cold", "Tuberculosis"}
        assert any(d in respiratory for d in top_diseases)

    def test_predict_liver_symptoms(self, engine):
        """Liver symptoms should surface hepatitis/jaundice."""
        preds, _ = engine.predict(
            ["yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "yellowing_of_eyes"]
        )
        top_diseases = [p["disease"] for p in preds[:5]]
        liver_diseases = {
            "Jaundice", "Chronic cholestasis", "hepatitis A",
            "Hepatitis B", "Hepatitis C", "Hepatitis D", "Hepatitis E",
            "Alcoholic hepatitis",
        }
        assert any(d in liver_diseases for d in top_diseases)

    def test_predict_custom_top_n(self, engine):
        """top_n parameter limits results correctly."""
        preds, _ = engine.predict(["high_fever"], top_n=3)
        assert len(preds) <= 3


# -----------------------------------------------------------------------
# Performance – inference under 2000 ms (larger model)
# -----------------------------------------------------------------------
class TestPerformance:
    def test_inference_under_2000ms(self, engine):
        """Single inference call must complete in < 2000 ms (132 symptoms model)."""
        _, elapsed_ms = engine.predict(
            ["high_fever", "cough", "headache", "fatigue", "chills"]
        )
        assert elapsed_ms < 2000, f"Inference took {elapsed_ms:.1f} ms (limit: 2000 ms)"

    def test_inference_timing_multiple(self, engine):
        """Average of 5 runs should be well under 2000 ms."""
        total = 0.0
        for _ in range(5):
            _, ms = engine.predict(["high_fever", "nausea", "vomiting", "diarrhoea"])
            total += ms
        avg = total / 5
        assert avg < 2000, f"Average inference: {avg:.1f} ms (limit: 2000 ms)"
