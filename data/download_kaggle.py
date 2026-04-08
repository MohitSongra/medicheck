"""
MediCheck - Kaggle Dataset Downloader & Augmenter
===================================================
Downloads the 'Disease Prediction Using Machine Learning' dataset
from a public GitHub mirror, combines Training + Testing data,
and augments to 10,000+ rows using noise injection.

Source: https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning
License: Open Database License (ODbL)
"""

import os
import io
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "kaggle_raw"
OUTPUT_PATH = DATA_DIR / "disease_symptom.csv"

TRAINING_URL = (
    "https://raw.githubusercontent.com/parthsompura/"
    "Disease-prediction-using-Machine-Learning/master/Training.csv"
)
TESTING_URL = (
    "https://raw.githubusercontent.com/parthsompura/"
    "Disease-prediction-using-Machine-Learning/master/Testing.csv"
)

TARGET_ROWS = 10_080  # 240 per disease × 42 diseases
NOISE_RATE = 0.03     # probability of flipping a zero to one during augmentation
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_csv(url: str, save_path: Path) -> pd.DataFrame:
    """Download a CSV from URL and save locally."""
    import urllib.request

    print(f"  Downloading {url.split('/')[-1]} ...")
    os.makedirs(save_path.parent, exist_ok=True)

    urllib.request.urlretrieve(url, str(save_path))
    df = pd.read_csv(save_path)
    print(f"    -> {len(df)} rows, {len(df.columns)} columns")
    return df


# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names and target column."""
    # Rename last column to 'disease' (it's called 'prognosis' in the Kaggle data)
    if "prognosis" in df.columns:
        df = df.rename(columns={"prognosis": "disease"})

    # Strip whitespace from column names and disease names
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    df["disease"] = df["disease"].str.strip()

    # Drop any completely empty rows
    df = df.dropna(how="all").reset_index(drop=True)

    # Ensure all symptom columns are integer (0/1)
    symptom_cols = [c for c in df.columns if c != "disease"]
    df[symptom_cols] = df[symptom_cols].fillna(0).astype(int)

    return df


# ---------------------------------------------------------------------------
# Augment
# ---------------------------------------------------------------------------
def augment_dataset(
    df: pd.DataFrame,
    target_rows: int = TARGET_ROWS,
    noise_rate: float = NOISE_RATE,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Augment the dataset to reach target_rows by creating noisy copies.

    For each augmented sample:
    - Take a random existing row of the same disease
    - With probability `noise_rate`, flip 0 → 1 for non-characteristic symptoms
    - With probability `noise_rate / 2`, flip 1 → 0 (symptom dropping)
    """
    rng = np.random.default_rng(seed)
    symptom_cols = [c for c in df.columns if c != "disease"]
    diseases = df["disease"].unique()
    rows_per_disease = target_rows // len(diseases)

    augmented_frames = []

    for disease in diseases:
        disease_df = df[df["disease"] == disease].reset_index(drop=True)
        existing_count = len(disease_df)
        needed = max(0, rows_per_disease - existing_count)

        # Keep all original rows
        augmented_frames.append(disease_df)

        if needed > 0:
            # Sample with replacement from existing rows
            indices = rng.integers(0, existing_count, size=needed)
            new_rows = disease_df.iloc[indices].copy().reset_index(drop=True)

            # Apply noise
            symptom_values = new_rows[symptom_cols].values.astype(float)
            noise_mask_flip_on = rng.random(symptom_values.shape) < noise_rate
            noise_mask_flip_off = rng.random(symptom_values.shape) < (noise_rate / 2)

            # Flip 0 → 1 where noise triggers on zero-valued cells
            symptom_values = np.where(
                (symptom_values == 0) & noise_mask_flip_on, 1, symptom_values
            )
            # Flip 1 → 0 where noise triggers on one-valued cells (less aggressive)
            symptom_values = np.where(
                (symptom_values == 1) & noise_mask_flip_off, 0, symptom_values
            )

            new_rows[symptom_cols] = symptom_values.astype(int)
            augmented_frames.append(new_rows)

    result = pd.concat(augmented_frames, ignore_index=True)
    # Shuffle
    result = result.sample(frac=1, random_state=seed).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("MediCheck - Kaggle Dataset Downloader & Augmenter")
    print("=" * 60)

    # 1. Download
    print("\n[1/4] Downloading dataset from GitHub mirror ...")
    train_path = RAW_DIR / "Training.csv"
    test_path = RAW_DIR / "Testing.csv"

    df_train = download_csv(TRAINING_URL, train_path)
    df_test = download_csv(TESTING_URL, test_path)

    # 2. Combine & clean
    print("\n[2/4] Combining and cleaning ...")
    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    df_clean = clean_dataset(df_combined)
    print(f"  Combined: {len(df_clean)} rows, {df_clean['disease'].nunique()} diseases")
    symptom_cols = [c for c in df_clean.columns if c != "disease"]
    print(f"  Symptoms: {len(symptom_cols)}")

    # 3. Augment
    print(f"\n[3/4] Augmenting to {TARGET_ROWS}+ rows ...")
    df_augmented = augment_dataset(df_clean, target_rows=TARGET_ROWS)
    print(f"  Final: {len(df_augmented)} rows")
    print(f"  Diseases: {df_augmented['disease'].nunique()}")
    print(f"  Rows per disease (approx): {len(df_augmented) // df_augmented['disease'].nunique()}")

    # 4. Save
    print(f"\n[4/4] Saving to {OUTPUT_PATH} ...")
    os.makedirs(DATA_DIR, exist_ok=True)
    df_augmented.to_csv(OUTPUT_PATH, index=False)
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"  Saved! ({size_kb:.0f} KB)")

    print("\n[OK] Dataset ready for training!")
    print(f"   Total rows: {len(df_augmented)}")
    print(f"   Diseases:   {df_augmented['disease'].nunique()}")
    print(f"   Symptoms:   {len(symptom_cols)}")


if __name__ == "__main__":
    main()
