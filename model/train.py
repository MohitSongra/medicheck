"""
MediCheck - Bayesian Network Training Module
=============================================
Loads the Kaggle disease-symptom dataset and trains a Naive Bayes
Bayesian Network for disease prediction using pgmpy.

Dataset: "Disease Prediction Using Machine Learning" (Kaggle, by Kaushil268)
  - 132 symptoms (binary features)
  - 42 diseases (prognosis / target)
  - 10,000+ rows (after augmentation)
"""

import os
import pickle
from pathlib import Path

import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.estimators import BayesianEstimator

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
CSV_PATH = DATA_DIR / "disease_symptom.csv"
MODEL_PATH = MODEL_DIR / "bayesian_model.pkl"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dataset() -> pd.DataFrame:
    """Load the disease-symptom CSV dataset."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {CSV_PATH}.\n"
            "Run  python -m data.download_kaggle  first to download and prepare the dataset."
        )
    df = pd.read_csv(CSV_PATH)
    print(
        f"  {len(df)} records | "
        f"{df['disease'].nunique()} diseases | "
        f"{len(get_symptom_columns(df))} symptoms"
    )
    return df


def get_symptom_columns(df: pd.DataFrame) -> list[str]:
    """Return all symptom column names (everything except 'disease')."""
    return [c for c in df.columns if c != "disease"]


# ---------------------------------------------------------------------------
# Model building & training
# ---------------------------------------------------------------------------
def build_and_train_model(df: pd.DataFrame) -> BayesianNetwork:
    """Build a Naive-Bayes Bayesian Network and learn CPTs."""
    symptoms = get_symptom_columns(df)

    # Naive Bayes structure: disease → each symptom
    edges = [("disease", s) for s in symptoms]
    model = BayesianNetwork(edges)

    model.fit(
        df,
        estimator=BayesianEstimator,
        prior_type="BDeu",
        equivalent_sample_size=5,
    )
    return model


def save_model(model: BayesianNetwork, filepath: Path) -> None:
    """Serialize trained model to disk."""
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"  Model saved -> {filepath}")


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("MediCheck - Bayesian Network Training Pipeline")
    print("=" * 60)

    # 1. Load data
    print("\n[1/3] Loading dataset ...")
    df = load_dataset()

    # 2. Train
    print("\n[2/3] Training Bayesian Network ...")
    model = build_and_train_model(df)

    # 3. Save
    print("\n[3/3] Saving model ...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_model(model, MODEL_PATH)

    # Verification
    symptoms = get_symptom_columns(df)
    print("\n--- Model verification ---")
    print(f"  Nodes : {len(model.nodes())}")
    print(f"  Edges : {len(model.edges())}")
    print(f"  CPDs  : {len(model.get_cpds())}")
    print(f"  Diseases : {df['disease'].nunique()}")
    print(f"  Symptoms : {len(symptoms)}")
    assert model.check_model(), "Model validation failed!"
    print("  Status: VALID [OK]")


if __name__ == "__main__":
    main()
