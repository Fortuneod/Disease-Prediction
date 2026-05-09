import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import balanced_accuracy_score

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

# Main training function
def run_training():
    print("=" * 60)
    print(" DISEASE PREDICTION MODEL TRAINING ")
    print("=" * 60)

    print("Loading dataset...")
    df = pd.read_csv('healthcare_dataset.csv')

    print(f"\n[DATA] Loaded {df.shape[0]} patients, {df.shape[1]} columns")
    
    # Define features and targets
    features = ['Age', 'Gender', 'Blood Pressure', 'Cholesterol', 'Glucose', 
                'Smoking', 'Alcohol Consumption', 'Exercise', 'BMI', 'Family History']
    
    targets = ['Heart Disease', 'Diabetes', 'Stroke', 'Kidney Disease', 'Cancer', 
               "Alzheimer's Disease", 'COPD', 'Liver Disease', "Parkinson's Disease", 'Tuberculosis']
    
    # Prepare feature matrix
    X = df[features]

    # Identify categorical and numeric columns for preprocessing
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=['object']).columns.tolist()

    # Define a robust preprocessor for both numeric and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ])
    
    final_pipelines = {}
    
    print("\nTraining and Evaluating Models for EACH Disease (Using Class Weights):")
    for target in targets:
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 1. Calculate Imbalance Ratio dynamically for XGBoost
        neg_cases = (y_train == 0).sum()
        pos_cases = (y_train == 1).sum()
        scale_pos = neg_cases / pos_cases if pos_cases > 0 else 1.0
        
        # 2. Add class_weight='balanced' to penalize missed positive cases
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'), 
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=500, min_samples_leaf=5, class_weight='balanced'),
            'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42, scale_pos_weight=scale_pos),
            'SVM': SVC(probability=True, random_state=42, class_weight='balanced')
        }
        
        best_model = None
        best_score = 0
        best_name = ""
        
        for name, model in models.items():
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor), 
                ('classifier', model)
            ])
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            score = balanced_accuracy_score(y_test, y_pred)
            
            if score > best_score:
                best_score = score
                best_model = pipeline
                best_name = name
                
        print(f" - {target}: Best model is {best_name} (Balanced Accuracy {best_score:.4f})")
        
        # Retrain best model on full data and save
        best_model.fit(X, y)
        final_pipelines[target] = best_model
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(final_pipelines, 'models/pipeline.joblib')
    
    feature_info = {
        'features': features,
        'targets': targets,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'categories': {col: X[col].unique().tolist() for col in categorical_cols}
    }

    joblib.dump(feature_info, 'models/features.joblib')
    print("\n✅ Multi-disease class-weighted pipelines saved successfully in 'models/' folder.")

# Run the training process
if __name__ == '__main__':
    run_training()