import pandas as pd
import joblib
from typing import Dict, Union
import streamlit as st

class ParkinsonsPredictor:
    def __init__(self, model_path: str = r'E:\Proj_4\venv\parkinsons_model.pkl'):
        try:
            # Load model pipeline (StandardScaler + LogisticRegression)
            self.model = joblib.load(model_path)

            # Expected features and validation ranges
            self.required_features = ['spread1', 'PPE', 'spread2', 'MDVP:Fo(Hz)']
            self.feature_ranges = {
                'spread1': (-10.0, 0.0),
                'PPE': (0.0, 0.6),
                'spread2': (0.0, 0.5),
                'MDVP:Fo(Hz)': (80.0, 260.0)
            }

        except Exception as e:
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def validate_input(self, input_data: Dict[str, float]) -> Dict[str, float]:

        # Check missing fields
        missing = [f for f in self.required_features if f not in input_data]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        # Validate ranges
        for feat, (min_val, max_val) in self.feature_ranges.items():
            value = input_data[feat]
            if not (min_val <= value <= max_val):
                raise ValueError(
                    f"Value for {feat} must be between {min_val} and {max_val}"
                )

        return input_data

    def predict(self, input_data: Dict[str, float]) -> Dict[str, Union[int, float, str]]:

        try:
            # Validate and convert to DataFrame
            validated = self.validate_input(input_data)
            input_df = pd.DataFrame([validated])[self.required_features]

            # Predict
            prediction = self.model.predict(input_df)[0]
            probability = self.model.predict_proba(input_df)[0][1]

            return {
                "prediction": int(prediction),
                "probability": float(probability),
                "status": "success"
            }

        except ValueError as e:
            return {"status": "error", "message": str(e)}
        except Exception as e:
            return {"status": "error", "message": f"Prediction failed: {str(e)}"}
        
    def show_parkinsons(self):
             
        st.markdown("""
        <div class="header">
            <h1>🎤 Parkinson's Disease Predictor</h1>
            <p>Enter voice measurement parameters to assess Parkinson's disease risk</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Voice Measurement Parameters")
            
            # Input fields with validation hints
            spread1 = st.number_input(
                "Spread1 (Nonlinear measure of fundamental frequency variation)", 
                min_value=-10.0, 
                max_value=0.0, 
                value=-5.0,
                step=0.1,
                help="Typical range: -10.0 to 0.0"
            )
            
            ppe = st.number_input(
                "PPE (Pitch Period Entropy)", 
                min_value=0.0, 
                max_value=0.6, 
                value=0.2,
                step=0.01,
                format="%.2f",
                help="Typical range: 0.0 to 0.6"
            )
            
            spread2 = st.number_input(
                "Spread2 (Nonlinear measure of fundamental frequency variation)", 
                min_value=0.0, 
                max_value=0.5, 
                value=0.2,
                step=0.01,
                format="%.2f",
                help="Typical range: 0.0 to 0.5"
            )
            
            mdvp_fo = st.number_input(
                "MDVP:Fo(Hz) (Average vocal fundamental frequency)", 
                min_value=80.0, 
                max_value=260.0, 
                value=150.0,
                step=1.0,
                help="Typical range: 80-260 Hz"
            )

            if st.button("Predict Parkinson's Risk"):
                try:
                    input_data = {
                        'spread1': spread1,
                        'PPE': ppe,
                        'spread2': spread2,
                        'MDVP:Fo(Hz)': mdvp_fo
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
                                st.error("⚠️ This result suggests potential Parkinson's disease risk. Please consult with a neurologist.")
                                st.info("Common next steps: Neurological examination, DaTscan, motor symptom assessment")
                            else:
                                st.success("✅ Results suggest low risk, but monitoring is recommended if symptoms appear.")
                                st.info("Maintain regular check-ups and monitor for symptoms like tremors or rigidity")
                        else:
                            st.error(f"Prediction failed: {result['message']}")
                            
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")