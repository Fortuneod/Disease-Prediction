import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Comprehensive Disease Predictor", page_icon="🏥", layout="wide")

# --- INITIALIZE LOCAL STATISTICS ---
# This replaces the API stats to keep the dashboard fully standalone
if 'total_screened' not in st.session_state:
    st.session_state['total_screened'] = 0
if 'total_positive_flags' not in st.session_state:
    st.session_state['total_positive_flags'] = 0

# --- LOAD MODELS ---
@st.cache_resource
def load_model_artifacts():
    if os.path.exists('models/pipeline.joblib') and os.path.exists('models/features.joblib'):
        return joblib.load('models/pipeline.joblib'), joblib.load('models/features.joblib')
    return None, None

pipelines_dict, features_info = load_model_artifacts()

# --- SIDEBAR CONFIGURATION ---
st.sidebar.title("App Dashboard")
st.sidebar.subheader("Session Usage Statistics")
st.sidebar.metric("Patients Screened", st.session_state['total_screened'])
st.sidebar.metric("Total Positive Flags", st.session_state['total_positive_flags'])

st.sidebar.subheader("Available Screenings")
if features_info:
    for disease in features_info['targets']:
        st.sidebar.markdown(f"- {disease}")

# --- MAIN PAGE CONFIGURATION ---
st.title("🏥 Comprehensive Disease Predictor")

if not pipelines_dict:
    st.error("Model artifacts missing! Please run `python training.py` first to generate the models.")
    st.stop()

tab1, tab2 = st.tabs(["Patient Feature Form (Single View)", "Batch File Processing"])

# TAB 1: Single Input Prediction
with tab1:
    st.header("Patient Feature Input")
    cats = features_info['categories']
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=45)
            gender = st.selectbox("Gender", cats['Gender'])
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
            blood_pressure = st.selectbox("Blood Pressure", cats['Blood Pressure'])
        with col2:
            cholesterol = st.selectbox("Cholesterol", cats['Cholesterol'])
            glucose = st.selectbox("Glucose", cats['Glucose'])
            smoking = st.selectbox("Smoking", cats['Smoking'])
        with col3:
            alcohol = st.selectbox("Alcohol Consumption", cats['Alcohol Consumption'])
            exercise = st.selectbox("Exercise", cats['Exercise'])
            family_history = st.selectbox("Family History", cats['Family History'])
            
        submit = st.form_submit_button("Predict Comprehensive Risk")

    if submit:
        input_data = pd.DataFrame([{
            'Age': age, 'Gender': gender, 'Blood Pressure': blood_pressure, 
            'Cholesterol': cholesterol, 'Glucose': glucose, 'Smoking': smoking, 
            'Alcohol Consumption': alcohol, 'Exercise': exercise, 'BMI': bmi, 
            'Family History': family_history
        }])
        
        st.subheader("Patient Risk Profile Results")
        results = []
        positive_flags_count = 0
        
        # Iterate and Predict across all targeted diseases locally
        for disease, pipeline in pipelines_dict.items():
            pred = pipeline.predict(input_data)[0]
            prob = pipeline.predict_proba(input_data)[0][1] if hasattr(pipeline, "predict_proba") else 0
            
            if pred == 1:
                positive_flags_count += 1
                
            results.append({
                "Disease": disease, 
                "Risk": "Positive" if pred == 1 else "Negative", 
                "Probability": prob
            })
            
        # Update local session stats
        st.session_state['total_screened'] += 1
        st.session_state['total_positive_flags'] += positive_flags_count
        
        res_df = pd.DataFrame(results) #.sort_values(by="Probability", ascending=False)
        res_df.index = res_df.index + 1  # Start index at 1 for better readability
        # Display as styled blocks
        positives = res_df[res_df['Risk'] == 'Positive']
        
        if not positives.empty:
            st.error(f"⚠️ High Risk Detected for: {', '.join(positives['Disease'].tolist())}")
        else:
            st.success("✅ No Risks detected based on provided biometrics.")
            
        st.dataframe(res_df.style.format({'Probability': '{:.2%}'}), width='stretch')

        # Store input data in session state for SHAP rendering
        st.session_state['last_input'] = input_data

    # --- SHAP EXPLANATION ---
    st.divider()
    st.subheader("Decision Breakdown (SHAP Waterfall)")
    
    if 'last_input' in st.session_state:
        selected_disease = st.selectbox("Select Disease Model to Analyze:", features_info['targets'])
        
        if st.button(f"Generate SHAP Explanation for {selected_disease}"):
            try:
                active_pipeline = pipelines_dict[selected_disease]
                data_in = st.session_state['last_input']
                
                # Manual preprocessing to extract exact dummy variable paths
                X_transformed = active_pipeline.named_steps['preprocessor'].transform(data_in)
                classifier_model = active_pipeline.named_steps['classifier']
                
                cat_features = active_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out()
                all_features = list(features_info['numeric_cols']) + list(cat_features)
                
                explainer = shap.Explainer(classifier_model, X_transformed)
                shap_values = explainer(X_transformed)
                shap_values.feature_names = all_features
                
                # Isolate target binary matrix based on sub-model architecture
                shap_obj = shap_values[0, :, 1] if len(shap_values.shape) == 3 else shap_values[0]
                
                fig, ax = plt.subplots(figsize=(10, 5))
                shap.plots.waterfall(shap_obj, show=False)
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not calculate SHAP explanation dynamically for this model architecture. {e}")
    else:
        st.info("Run a prediction above to enable SHAP explanations.")

# TAB 2: BATCH PROCESSING 
with tab2:
    st.header("Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV File with Patients' Records. Ensure that the column names match the expected features.", type="csv")
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        missing_cols = [col for col in features_info['features'] if col not in batch_df.columns]
        
        if missing_cols:
            st.error(f"❌ Failed. Missing required columns: {missing_cols}")
        else:
            st.write(f"Loaded **{len(batch_df)}** records for processing.")
            
            if st.button("✓ Run Comprehensive Batch Prediction"):
                with st.spinner("Processing Model Predictions across all diseases..."):
                    X_batch = batch_df[features_info['features']]
                    batch_positive_flags = 0
                    
                    # Run predictions for every disease
                    for disease, pipeline in pipelines_dict.items():
                        preds = pipeline.predict(X_batch)
                        batch_df[f'{disease}_Prediction'] = preds
                        # Count positive flags for stats
                        batch_positive_flags += sum(preds)
                        
                    # Update session state stats
                    st.session_state['total_screened'] += len(batch_df)
                    st.session_state['total_positive_flags'] += batch_positive_flags
                        
                    st.success("Batch Prediction Complete!")
                    st.dataframe(batch_df.head(15)) 
                    
                    csv_export = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Download Full Predicted CSV",
                        data=csv_export,
                        file_name='comprehensive_batch_predictions.csv',
                        mime='text/csv',
                    )