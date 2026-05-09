from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import uvicorn

app = FastAPI(title="Multi-Disease Prediction API")

try:
    # pipeline_dict contains a dictionary of trained pipelines for each disease
    pipelines_dict = joblib.load('models/pipeline.joblib')
    features_info = joblib.load('models/features.joblib')
except Exception as e:
    pipelines_dict = None
    features_info = None

stats = {
    "total_patients_screened": 0,
    "total_positive_disease_flags": 0
}

class PredictRequest(BaseModel):
    Age: int
    Gender: str
    Blood_Pressure: str = Field(alias="Blood Pressure")
    Cholesterol: str
    Glucose: str
    Smoking: str
    Alcohol_Consumption: str = Field(alias="Alcohol Consumption")
    Exercise: str
    BMI: float
    Family_History: str = Field(alias="Family History")

@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": pipelines_dict is not None}

@app.get("/info")
def get_info():
    if not features_info:
        raise HTTPException(status_code=404, detail="Features info not found")
    return {
        "required_features": features_info['features'], 
        "target_diseases": features_info['targets'],
        "categorical_values": features_info['categories']
    }

@app.post("/predict")
def predict(data: PredictRequest):
    if not pipelines_dict:
        raise HTTPException(status_code=500, detail="Model pipelines not loaded. Run training.py.")
    
    df_input = pd.DataFrame([{
        'Age': data.Age, 'Gender': data.Gender, 'Blood Pressure': data.Blood_Pressure,
        'Cholesterol': data.Cholesterol, 'Glucose': data.Glucose, 'Smoking': data.Smoking,
        'Alcohol Consumption': data.Alcohol_Consumption, 'Exercise': data.Exercise,
        'BMI': data.BMI, 'Family History': data.Family_History
    }])
    
    try:
        results = {}
        flags = 0
        
        # Iterate over each disease pipeline
        for disease, pipeline in pipelines_dict.items():
            prediction = pipeline.predict(df_input)[0]
            probability = pipeline.predict_proba(df_input)[0][1] if hasattr(pipeline, "predict_proba") else None
            
            results[disease] = {
                "prediction": int(prediction),
                "prediction_label": "Positive Risk" if prediction == 1 else "Negative Risk",
                "probability": float(probability) if probability is not None else None
            }
            if prediction == 1:
                flags += 1
                
        # Update global stats
        stats["total_patients_screened"] += 1
        stats["total_positive_disease_flags"] += flags
            
        return {"patient_risk_profile": results}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/stats")
def get_stats():
    return stats

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)