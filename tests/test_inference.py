"""
MediCheck - Inference Module Tests
====================================
Validates that the BN model loads correctly, produces predictions,
and meets performance requirements (AC-01, AC-06, AC-08).
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
# AC-01: Model loads without errors
# -----------------------------------------------------------------------
class TestModelLoading:
    def test_model_loads_successfully(self, engine):
        """The BN model loads without exceptions."""
        assert engine.model is not None

    def test_model_has_disease_node(self, engine):
        """The model contains a 'disease' node."""
        assert "disease" in engine.model.nodes()

    def test_model_has_symptoms(self, engine):
        """The model has at least 50 symptom nodes."""
        assert len(engine.get_all_symptoms()) >= 50

    def test_model_has_diseases(self, engine):
        """The model knows about at least 15 diseases."""
        assert len(engine.get_all_diseases()) >= 15

    def test_model_is_valid(self, engine):
        """pgmpy internal model check passes."""
        assert engine.model.check_model()


# -----------------------------------------------------------------------
# Prediction correctness
# -----------------------------------------------------------------------
class TestPredictions:
    def test_predict_returns_results(self, engine):
        """Predict returns a non-empty list for valid symptoms."""
        preds, _ = engine.predict(["fever", "cough", "headache"])
        assert len(preds) > 0

    def test_predict_top5(self, engine):
        """Predict returns at most 5 results by default."""
        preds, _ = engine.predict(["fever", "cough"])
        assert len(preds) <= 5

    def test_predict_probabilities_sum(self, engine):
        """All returned probabilities are between 0 and 1."""
        preds, _ = engine.predict(["fever", "chills", "sweating"])
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

    def test_predict_respiratory_symptoms(self, engine):
        """Respiratory symptoms should rank respiratory diseases highly."""
        preds, _ = engine.predict(
            ["cough", "shortness_of_breath", "wheezing", "chest_tightness"]
        )
        top_diseases = [p["disease"] for p in preds[:3]]
        respiratory = {"Asthma", "Bronchitis", "Pneumonia", "COVID_19"}
        assert any(d in respiratory for d in top_diseases)

    def test_predict_uti_symptoms(self, engine):
        """UTI-specific symptoms should surface UTI."""
        preds, _ = engine.predict(
            ["burning_urination", "frequent_urination", "back_pain"]
        )
        top_diseases = [p["disease"] for p in preds[:3]]
        assert "Urinary_Tract_Infection" in top_diseases

    def test_predict_custom_top_n(self, engine):
        """top_n parameter limits results correctly."""
        preds, _ = engine.predict(["fever"], top_n=3)
        assert len(preds) <= 3


# -----------------------------------------------------------------------
# AC-06: Performance – inference under 500 ms
# -----------------------------------------------------------------------
class TestPerformance:
    def test_inference_under_500ms(self, engine):
        """Single inference call must complete in < 500 ms."""
        _, elapsed_ms = engine.predict(
            ["fever", "cough", "headache", "fatigue", "chills"]
        )
        assert elapsed_ms < 500, f"Inference took {elapsed_ms:.1f} ms (limit: 500 ms)"

    def test_inference_timing_multiple(self, engine):
        """Average of 5 runs should be well under 500 ms."""
        total = 0.0
        for _ in range(5):
            _, ms = engine.predict(["fever", "nausea", "vomiting", "diarrhea"])
            total += ms
        avg = total / 5
        assert avg < 500, f"Average inference: {avg:.1f} ms (limit: 500 ms)"
