# Comprehensive Disease Prediction

A multi-disease risk prediction project built with Python. It trains binary classifiers for multiple medical conditions, exposes a FastAPI endpoint for predictions, and provides an interactive Streamlit dashboard for single-patient and batch scoring.

## Project Structure

- `training.py` - trains models for each disease using a shared preprocessing pipeline, selects the best classifier per target, and saves trained artifacts in the `models/` folder.
- `app.py` - FastAPI service that loads the saved pipelines and serves `/health`, `/info`, `/predict`, and `/stats` endpoints.
- `streamlit_app.py` - Streamlit dashboard for interactive single-patient prediction, batch file upload, and SHAP-based model explanation.
- `healthcare_dataset.csv` - dataset used to train the models.
- `models/` - contains generated artifacts after training:
  - `pipeline.joblib`
  - `features.joblib`
- `requirements.txt` - Python dependencies required to run the project.

## Key Features

- Trains disease-specific classifiers for:
  - Heart Disease
  - Diabetes
  - Stroke
  - Kidney Disease
  - Cancer
  - Alzheimer’s Disease
  - COPD
  - Liver Disease
  - Parkinson’s Disease
  - Tuberculosis
- Uses preprocessing for numeric and categorical values.
- Evaluates multiple models per disease and selects the best by balanced accuracy.
- Supports both REST API and interactive UI workflows.
- Includes local session statistics and SHAP explanation support in Streamlit.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Setup and Usage

### 1. Train the models

Generate the model artifacts used by the API and dashboard:

```bash
python training.py
```

This creates:

- `models/pipeline.joblib`
- `models/features.joblib`

### 2. Run the FastAPI app

Start the API server:

```bash
python app.py
```

The API will be available at `http://0.0.0.0:8000`.

Available endpoints:

- `GET /health` - basic health check and model load status
- `GET /info` - returns required features and target disease list
- `POST /predict` - send patient data and receive risk predictions
- `GET /stats` - session statistics for the API

### 3. Run the Streamlit dashboard

Start the dashboard with:

```bash
streamlit run streamlit_app.py
```

The app includes:

- Single patient feature form
- Batch CSV upload and scoring
- Local session metrics
- SHAP waterfall explanations for selected disease predictions

## Input Features

The project expects the following patient features:

- `Age`
- `Gender`
- `Blood Pressure`
- `Cholesterol`
- `Glucose`
- `Smoking`
- `Alcohol Consumption`
- `Exercise`
- `BMI`
- `Family History`

## Batch Prediction Notes

For batch processing in `streamlit_app.py`, upload a CSV file containing the exact feature column names above. The dashboard app will generate one prediction column per disease.

## Notes

- If `models/pipeline.joblib` or `models/features.joblib` are missing, both the API and Streamlit app will prompt you to run `python training.py` first.
- The project currently uses class-balanced training for improved handling of imbalanced disease labels.

## Contact

This repository is designed as a research/demo application for multi-disease risk profiling using machine learning. For Collaboration, kindly reach out via fortuneodesanya@gmail.com
