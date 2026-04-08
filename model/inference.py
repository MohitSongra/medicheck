"""
MediCheck - Bayesian Network Inference Module
==============================================
Loads the trained Bayesian Network model and performs disease
prediction via Variable Elimination given observed symptoms.
"""

import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional

from pgmpy.inference import VariableElimination

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "bayesian_model.pkl"


class MediCheckInference:
    """Disease inference engine using a trained Bayesian Network."""

    def __init__(self, model_path: Optional[str | Path] = None):
        path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.model = self._load_model(path)
        self.inference_engine = VariableElimination(self.model)
        self.valid_symptoms: list[str] = sorted(
            node for node in self.model.nodes() if node != "disease"
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _load_model(model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                "Run  python -m model.train  first."
            )
        with open(model_path, "rb") as fh:
            return pickle.load(fh)

    # ------------------------------------------------------------------
    def predict(
        self,
        symptoms: List[str],
        top_n: int = 5,
    ) -> tuple[list[dict], float]:
        """
        Predict the most likely diseases for a set of observed symptoms.

        Parameters
        ----------
        symptoms : list[str]
            Symptom names that the patient is experiencing.
        top_n : int
            Number of top predictions to return.

        Returns
        -------
        (predictions, elapsed_ms)
            predictions – list of dicts with keys: disease, probability, confidence_pct
            elapsed_ms – inference wall-clock time in milliseconds
        """
        valid_evidence = {s: 1 for s in symptoms if s in self.valid_symptoms}

        if not valid_evidence:
            return [], 0.0

        t0 = time.perf_counter()

        result = self.inference_engine.query(
            variables=["disease"],
            evidence=valid_evidence,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Build ranked list
        disease_probs: list[dict] = []
        for idx, disease_name in enumerate(result.state_names["disease"]):
            prob = float(result.values[idx])
            disease_probs.append(
                {
                    "disease": disease_name,
                    "probability": prob,
                    "confidence_pct": round(prob * 100, 2),
                }
            )

        disease_probs.sort(key=lambda d: d["probability"], reverse=True)
        return disease_probs[:top_n], elapsed_ms

    # ------------------------------------------------------------------
    def get_all_symptoms(self) -> list[str]:
        """Return every symptom the model knows about."""
        return list(self.valid_symptoms)

    def get_all_diseases(self) -> list[str]:
        """Return every disease the model knows about."""
        disease_cpd = self.model.get_cpds("disease")
        return list(disease_cpd.state_names["disease"])
