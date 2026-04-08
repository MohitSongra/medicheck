"""
MediCheck - Bayesian Network Training Module
=============================================
Generates synthetic medical symptom data and trains a Naive Bayes
Bayesian Network for disease prediction using pgmpy.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.estimators import BayesianEstimator

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"

# ---------------------------------------------------------------------------
# Symptom definitions (52 unique symptoms)
# ---------------------------------------------------------------------------
SYMPTOMS = [
    "fever", "cough", "headache", "fatigue", "body_ache", "sore_throat",
    "runny_nose", "sneezing", "shortness_of_breath", "chest_pain",
    "nausea", "vomiting", "diarrhea", "abdominal_pain", "loss_of_appetite",
    "weight_loss", "night_sweats", "chills", "joint_pain", "muscle_pain",
    "skin_rash", "itching", "swollen_lymph_nodes", "dizziness",
    "blurred_vision", "frequent_urination", "burning_urination", "back_pain",
    "high_blood_pressure", "increased_thirst", "dry_mouth", "sweating",
    "pale_skin", "rapid_heartbeat", "weakness", "loss_of_smell",
    "loss_of_taste", "watery_eyes", "nasal_congestion", "wheezing",
    "chest_tightness", "blood_in_sputum", "yellow_skin", "dark_urine",
    "light_sensitivity", "stiff_neck", "red_eyes", "swelling",
    "bruising", "constipation", "excessive_hunger", "facial_pain",
]

# ---------------------------------------------------------------------------
# Disease profiles  –  symptom -> P(symptom=1 | disease)
# Unlisted symptoms default to 0.03 background rate
# ---------------------------------------------------------------------------
DISEASE_PROFILES = {
    "Common_Cold": {
        "fever": 0.30, "cough": 0.80, "headache": 0.40, "fatigue": 0.50,
        "sore_throat": 0.70, "runny_nose": 0.90, "sneezing": 0.85,
        "nasal_congestion": 0.85, "muscle_pain": 0.25, "weakness": 0.30,
    },
    "Influenza": {
        "fever": 0.95, "cough": 0.80, "headache": 0.80, "fatigue": 0.90,
        "body_ache": 0.85, "sore_throat": 0.50, "chills": 0.80,
        "muscle_pain": 0.85, "weakness": 0.75, "sweating": 0.45,
        "runny_nose": 0.35, "nasal_congestion": 0.30, "nausea": 0.25,
    },
    "COVID_19": {
        "fever": 0.85, "cough": 0.75, "headache": 0.60, "fatigue": 0.85,
        "body_ache": 0.60, "shortness_of_breath": 0.55,
        "loss_of_smell": 0.70, "loss_of_taste": 0.65, "sore_throat": 0.45,
        "chills": 0.40, "diarrhea": 0.25, "nausea": 0.20,
        "muscle_pain": 0.50,
    },
    "Pneumonia": {
        "fever": 0.90, "cough": 0.90, "shortness_of_breath": 0.80,
        "chest_pain": 0.70, "chills": 0.65, "fatigue": 0.75,
        "sweating": 0.50, "blood_in_sputum": 0.35, "muscle_pain": 0.40,
        "nausea": 0.25, "rapid_heartbeat": 0.45, "weakness": 0.60,
    },
    "Bronchitis": {
        "cough": 0.95, "fatigue": 0.65, "shortness_of_breath": 0.55,
        "chest_tightness": 0.60, "sore_throat": 0.50, "body_ache": 0.40,
        "chills": 0.35, "fever": 0.40, "wheezing": 0.45,
        "muscle_pain": 0.30, "headache": 0.30,
    },
    "Asthma": {
        "wheezing": 0.90, "shortness_of_breath": 0.85,
        "chest_tightness": 0.80, "cough": 0.75, "fatigue": 0.40,
        "rapid_heartbeat": 0.35, "sweating": 0.20, "dizziness": 0.25,
    },
    "Allergic_Rhinitis": {
        "sneezing": 0.90, "runny_nose": 0.90, "nasal_congestion": 0.85,
        "watery_eyes": 0.80, "itching": 0.70, "headache": 0.35,
        "sore_throat": 0.30, "fatigue": 0.30, "red_eyes": 0.40,
        "facial_pain": 0.20,
    },
    "Sinusitis": {
        "headache": 0.80, "nasal_congestion": 0.85, "facial_pain": 0.75,
        "fever": 0.40, "runny_nose": 0.70, "cough": 0.45,
        "fatigue": 0.50, "sore_throat": 0.35, "dizziness": 0.25,
        "loss_of_smell": 0.30,
    },
    "Tuberculosis": {
        "cough": 0.90, "blood_in_sputum": 0.55, "night_sweats": 0.80,
        "weight_loss": 0.75, "fatigue": 0.85, "fever": 0.70,
        "loss_of_appetite": 0.70, "chest_pain": 0.45, "chills": 0.40,
        "weakness": 0.65, "shortness_of_breath": 0.40,
    },
    "Malaria": {
        "fever": 0.95, "chills": 0.90, "sweating": 0.80, "headache": 0.75,
        "nausea": 0.60, "vomiting": 0.50, "body_ache": 0.70,
        "fatigue": 0.80, "muscle_pain": 0.55, "diarrhea": 0.30,
        "abdominal_pain": 0.35, "weakness": 0.60, "joint_pain": 0.40,
    },
    "Dengue_Fever": {
        "fever": 0.95, "headache": 0.85, "body_ache": 0.80,
        "joint_pain": 0.75, "skin_rash": 0.60, "nausea": 0.55,
        "fatigue": 0.80, "bruising": 0.35, "vomiting": 0.40,
        "muscle_pain": 0.70, "loss_of_appetite": 0.50, "red_eyes": 0.30,
        "swollen_lymph_nodes": 0.25,
    },
    "Typhoid_Fever": {
        "fever": 0.95, "headache": 0.70, "abdominal_pain": 0.75,
        "loss_of_appetite": 0.70, "weakness": 0.75, "diarrhea": 0.55,
        "constipation": 0.45, "fatigue": 0.80, "nausea": 0.40,
        "vomiting": 0.30, "skin_rash": 0.25, "chills": 0.35,
    },
    "Gastroenteritis": {
        "nausea": 0.85, "vomiting": 0.80, "diarrhea": 0.90,
        "abdominal_pain": 0.80, "fever": 0.55, "loss_of_appetite": 0.65,
        "weakness": 0.60, "fatigue": 0.55, "chills": 0.30,
        "headache": 0.35, "muscle_pain": 0.25,
    },
    "Urinary_Tract_Infection": {
        "burning_urination": 0.90, "frequent_urination": 0.85,
        "back_pain": 0.55, "abdominal_pain": 0.50, "fever": 0.45,
        "nausea": 0.30, "fatigue": 0.35, "chills": 0.30,
        "dark_urine": 0.40, "weakness": 0.25,
    },
    "Migraine": {
        "headache": 0.95, "nausea": 0.65, "light_sensitivity": 0.80,
        "blurred_vision": 0.50, "dizziness": 0.55, "vomiting": 0.40,
        "fatigue": 0.50, "stiff_neck": 0.30, "weakness": 0.25,
    },
    "Hypertension": {
        "headache": 0.60, "dizziness": 0.55, "blurred_vision": 0.40,
        "shortness_of_breath": 0.35, "chest_pain": 0.35,
        "high_blood_pressure": 0.95, "fatigue": 0.40, "nausea": 0.20,
        "sweating": 0.25, "rapid_heartbeat": 0.35,
    },
    "Diabetes_Type_2": {
        "frequent_urination": 0.85, "increased_thirst": 0.85,
        "fatigue": 0.70, "blurred_vision": 0.50, "weight_loss": 0.55,
        "excessive_hunger": 0.75, "dry_mouth": 0.70, "weakness": 0.50,
        "swelling": 0.30, "dizziness": 0.25, "itching": 0.25,
    },
    "Anemia": {
        "fatigue": 0.90, "weakness": 0.85, "pale_skin": 0.80,
        "dizziness": 0.65, "shortness_of_breath": 0.55,
        "rapid_heartbeat": 0.60, "headache": 0.45, "chest_pain": 0.25,
        "nausea": 0.20, "loss_of_appetite": 0.30,
    },
}


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def generate_dataset(
    num_samples_per_disease: int = 12,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic disease-symptom binary dataset."""
    rng = np.random.default_rng(random_seed)
    base_rate = 0.03
    records: list[dict] = []

    for disease, profile in DISEASE_PROFILES.items():
        for _ in range(num_samples_per_disease):
            record: dict = {"disease": disease}
            for symptom in SYMPTOMS:
                prob = profile.get(symptom, base_rate)
                record[symptom] = int(rng.random() < prob)
            records.append(record)

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Model building & training
# ---------------------------------------------------------------------------
def build_and_train_model(df: pd.DataFrame) -> BayesianNetwork:
    """Build a Naive-Bayes Bayesian Network and learn CPTs."""
    edges = [("disease", s) for s in SYMPTOMS]
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
    csv_path = DATA_DIR / "disease_symptom.csv"
    model_path = MODEL_DIR / "bayesian_model.pkl"

    # 1. Prepare data
    if not csv_path.exists():
        print("[1/3] Generating synthetic dataset ...")
        df = generate_dataset(num_samples_per_disease=12)
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"  Saved {len(df)} records  ->  {csv_path}")
    else:
        print("[1/3] Loading existing dataset ...")
        df = pd.read_csv(csv_path)
        print(
            f"  {len(df)} records | "
            f"{df['disease'].nunique()} diseases | "
            f"{len(SYMPTOMS)} symptoms"
        )

    # 2. Train
    print("[2/3] Training Bayesian Network ...")
    model = build_and_train_model(df)

    # 3. Save
    print("[3/3] Saving model ...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_model(model, model_path)

    # Verification
    print("\n--- Model verification ---")
    print(f"  Nodes : {len(model.nodes())}")
    print(f"  Edges : {len(model.edges())}")
    print(f"  CPDs  : {len(model.get_cpds())}")
    assert model.check_model(), "Model validation failed!"
    print("  Status: VALID ✓")


if __name__ == "__main__":
    main()
