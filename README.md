# MediCheck – AI Symptom Checker 🩺

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?logo=streamlit&logoColor=white)
![pgmpy](https://img.shields.io/badge/pgmpy-Bayesian_Network-orange)
![License](https://img.shields.io/badge/License-Academic_Use-lightgrey)

> **Disclaimer:** This project is for **educational and academic purposes only**. It is NOT a medical diagnosis tool and must never be used as a substitute for professional medical advice.

## Overview

MediCheck is an AI-powered Medical Symptom Checker that uses a **Bayesian Network** to predict the most likely disease(s) based on user-entered symptoms. The system employs a Naive Bayes graphical model trained on the [Kaggle Disease Prediction dataset](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning) (10,000+ rows, 41 diseases, 132 symptoms) and performs inference via **Variable Elimination**.

## 📸 Screenshot

![MediCheck UI](docs/screenshot.png)

## Architecture

```
┌────────────────┐     POST /predict     ┌────────────────┐     pgmpy VE     ┌──────────────┐
│   Streamlit    │ ──────────────────────▶│    FastAPI      │ ───────────────▶│  Bayesian    │
│   Frontend     │◀────── JSON ──────────│    Gateway      │◀───────────────│  Network     │
└────────────────┘                       └────────────────┘                 └──────────────┘
```

## Tech Stack

| Layer      | Technology                |
|------------|---------------------------|
| Frontend   | Streamlit 1.32+           |
| API        | FastAPI 0.110+ / Uvicorn  |
| ML Model   | pgmpy (Bayesian Network)  |
| Data       | pandas, numpy             |
| Viz        | Plotly, Altair            |
| Testing    | pytest                    |

## File Structure

```
medicheck/
├── data/
│   ├── download_kaggle.py       # Dataset downloader & augmenter
│   ├── disease_symptom.csv      # Training data (10k+ rows)
│   ├── disease_info.json        # Disease descriptions & actions (41 diseases)
│   └── kaggle_raw/              # Raw Kaggle CSV files
├── model/
│   ├── train.py                 # BN training pipeline
│   ├── inference.py             # Variable Elimination inference
│   └── bayesian_model.pkl       # Serialised model (generated)
├── api/
│   ├── main.py                  # FastAPI app & endpoints
│   ├── schemas.py               # Pydantic models
│   └── utils.py                 # Helper utilities
├── frontend/
│   ├── app.py                   # Streamlit entry point
│   └── pages/
│       ├── symptom_input.py     # Symptom selection UI (132 symptoms)
│       └── results.py           # Results visualisation
├── tests/
│   ├── test_inference.py        # Model & inference tests
│   └── test_api.py              # API endpoint tests
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
cd medicheck
pip install -r requirements.txt
```

### 2. Download & Prepare Dataset

```bash
python -m data.download_kaggle
```

This will:
- Download the Kaggle Disease Prediction dataset
- Augment to 10,000+ rows using noise injection
- Save `data/disease_symptom.csv`

### 3. Train the Model

```bash
python -m model.train
```

This will:
- Load the dataset (10,000+ records, 41 diseases, 132 symptoms)
- Train the Bayesian Network with BDeu priors
- Save `model/bayesian_model.pkl`

### 4. Start the API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: `http://localhost:8000/docs`

### 5. Launch the Frontend

```bash
streamlit run frontend/app.py
```

Open `http://localhost:8501` in your browser.

### 6. Run Tests

```bash
pytest tests/ -v
```

## API Endpoints

| Method | Path        | Description                           |
|--------|-------------|---------------------------------------|
| GET    | `/health`   | Health check & model status           |
| GET    | `/symptoms` | List all recognised symptoms          |
| POST   | `/predict`  | Predict diseases from symptom list    |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"symptoms": ["itching", "skin_rash", "high_fever"]}'
```

## Model Details

- **Structure:** Naive Bayes Bayesian Network (Disease -> Symptom_1, Disease -> Symptom_2, ...)
- **Estimation:** Bayesian Estimation with BDeu prior (equivalent sample size = 5)
- **Inference:** Variable Elimination for exact posterior computation
- **Dataset:** Kaggle Disease Prediction (10,000+ rows, augmented from ~5,000 original)
- **Diseases:** 41 conditions across respiratory, dermatological, hepatic, metabolic, and infectious categories
- **Symptoms:** 132 clinical features

## Environment Setup

```bash
cp .env.example .env
# Edit .env with your values (optional – defaults work out of the box)
```

> **Note:** The `.env` file is git-ignored. Never commit real credentials.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is intended for academic use. No medical claims are made.
