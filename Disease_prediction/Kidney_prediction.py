import pandas as pd
import pickle
from typing import Dict, Union
from sklearn.pipeline import Pipeline
import streamlit as st

class KidneyDiseasePredictor:
    def __init__(self, model_path: str = r'E:\Proj_4\venv\kidney_disease_model.pkl'):
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)

            # Expected features and validation rules
            self.required_features = ['hemo', 'sg', 'al', 'sc']
            self.feature_ranges = {
                'hemo': (3.0, 20.0),     # Hemoglobin (g/dL)
                'sg': (1.0, 1.05),       # Specific Gravity
                'al': (0, 5),             # Albumin (0-4 ordinal)
                'sc': (0.1, 20.0)         # Serum Creatinine (mg/dL)
            }

            # Categorical mappings (if any)
            self.categorical_values = {
                # Add categorical validations if needed
            }

        except Exception as e:
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def validate_input(self, input_data: Dict[str, Union[float, int]]) -> Dict[str, Union[float, int]]:
        # Check missing fields
        missing = [f for f in self.required_features if f not in input_data]
        if missing:
            raise ValueError(f"Missing required fields: {', '.join(missing)}")

        # Validate numerical ranges
        for feat, (min_val, max_val) in self.feature_ranges.items():
            value = input_data[feat]
            if not (min_val <= float(value) <= max_val):
                raise ValueError(
                    f"Invalid {feat}: {value}. Must be between {min_val}-{max_val}"
                )

        # Validate categoricals (example)
        if 'al' in input_data and input_data['al'] not in range(5):
            raise ValueError("Albumin (al) must be integer 0-4")

        return input_data

    def preprocess(self, input_data: Dict[str, Union[float, int]]) -> pd.DataFrame:

        try:
            validated = self.validate_input(input_data)
            return pd.DataFrame([validated])[self.required_features]
        except Exception as e:
            raise ValueError(f"Preprocessing failed: {str(e)}")

    def predict(self, input_data: Dict[str, Union[float, int]]) -> Dict[str, Union[int, float, str]]:

        try:
            input_df = self.preprocess(input_data)
            proba = self.model.predict_proba(input_df)[0][1]

            return {
                'prediction': int(self.model.predict(input_df)[0]),
                'probability': float(proba),
                'status': 'success'
            }
        except ValueError as e:
            return {'status': 'error', 'message': str(e)}
        except Exception as e:
            return {'status': 'error', 'message': f"Prediction failed: {str(e)}"}
        
    def show_kidney(self):
  
    
        st.markdown("""
        <div class="header">
            <h1>🧑‍⚕️ Kidney Disease Predictor</h1>
            <p>Enter patient's clinical measurements to assess kidney disease risk</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Clinical Measurements")
            
            # Input fields with validation hints
            hemo = st.number_input(
                "Hemoglobin (g/dL)", 
                min_value=3.0, 
                max_value=20.0, 
                value=12.0,
                step=0.1,
                help="Normal range: 12-16 g/dL for women, 13-17 g/dL for men"
            )
            
            sg = st.number_input(
                "Specific Gravity", 
                min_value=1.000, 
                max_value=1.050, 
                value=1.020,
                step=0.001,
                format="%.3f",
                help="Normal range: 1.005-1.030"
            )
            
            al = st.number_input(
                "Albumin (0-4 scale)", 
                min_value=0, 
                max_value=4, 
                value=0,
                step=1,
                help="0: None, 1: Trace, 2: 1+, 3: 2+, 4: 3+ or more"
            )
            
            sc = st.number_input(
                "Serum Creatinine (mg/dL)", 
                min_value=0.1, 
                max_value=20.0, 
                value=0.9,
                step=0.1,
                help="Normal range: 0.6-1.2 mg/dL for men, 0.5-1.1 mg/dL for women"
            )

            if st.button("Predict Kidney Disease Risk"):
                try:
                    input_data = {
                        'hemo': hemo,
                        'sg': sg,
                        'al': al,
                        'sc': sc
                    }
                    
                    # Make prediction
                    result = self.predict(input_data)
                    
                    with col2:
                        st.markdown("### Prediction Result")
                        if result['status'] == 'success':
                            prediction = "High Risk" if result['prediction'] == 1 else "Low Risk"
                            probability = result['probability'] * 100
                            
                            st.success(f"Prediction: {prediction}")
                            st.metric("Probability", f"{probability:.1f}%")
                            
                            # Add interpretation
                            if result['prediction'] == 1:
                                st.error("⚠️ This result suggests potential kidney disease risk. Please consult with a nephrologist.")
                                st.info("Common next steps: Urine test, eGFR calculation, ultrasound")
                            else:
                                st.success("✅ Results suggest low risk, but regular monitoring is recommended.")
                                st.info("Maintain kidney health by staying hydrated and controlling blood pressure")
                        else:
                            st.error(f"Prediction failed: {result['message']}")
                            
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")