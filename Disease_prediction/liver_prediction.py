# liver_prediction.py
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import streamlit as st

class LiverDiseasePredictor:
    def __init__(self, model_path=r'E:\Proj_4\venv\liver_model.pkl'):
        try:
            saved_data = joblib.load(model_path)
            self.model = saved_data['model']
            self.imputer = saved_data['imputer']
            self.power_transformer = saved_data['power_transformer']
            self.selector = saved_data['selector']
            self.gender_encoder = saved_data['gender_encoder']
            self.feature_names = saved_data['feature_names']
            self.selected_features = saved_data['selected_features']

            # Store allowed gender values for validation
            self.allowed_genders = list(self.gender_encoder.classes_)
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")

    def validate_input(self, input_data):
        # Check required fields
        required_fields = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
                         'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
                         'Aspartate_Aminotransferase', 'Albumin']

        missing_fields = [field for field in required_fields if field not in input_data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

        # Validate gender
        if input_data['Gender'] not in self.allowed_genders:
            raise ValueError(
                f"Invalid gender '{input_data['Gender']}'. "
                f"Allowed values are: {', '.join(self.allowed_genders)}"
            )

        return input_data

    def create_features(self, input_df):

        input_df = input_df.copy()

        # Calculate all engineered features
        input_df['AST_ALT_ratio'] = input_df['Aspartate_Aminotransferase'] / (input_df['Alamine_Aminotransferase'] + 1)
        input_df['TB_DB_ratio'] = input_df['Total_Bilirubin'] / (input_df['Direct_Bilirubin'] + 0.001)
        input_df['AP_ALT_ratio'] = input_df['Alkaline_Phosphotase'] / (input_df['Alamine_Aminotransferase'] + 1)
        input_df['Age_ALT'] = input_df['Age'] * input_df['Alamine_Aminotransferase'] / 100
        input_df['Age_AST'] = input_df['Age'] * input_df['Aspartate_Aminotransferase'] / 100
        input_df['ALT_squared'] = input_df['Alamine_Aminotransferase'] ** 2
        input_df['AST_squared'] = input_df['Aspartate_Aminotransferase'] ** 2
        input_df['Liver_Function_Score'] = input_df['Albumin'] / (input_df['Total_Bilirubin'] + 0.001)
        input_df['Gender_ALT'] = input_df['Gender'] * input_df['Alamine_Aminotransferase']

        return input_df

    def preprocess(self, input_data):

        try:
            # Validate and prepare input
            validated_data = self.validate_input(input_data)

            # Convert to DataFrame
            input_df = pd.DataFrame([validated_data])

            # Encode gender
            input_df['Gender'] = self.gender_encoder.transform(input_df['Gender'])

            # Create all engineered features
            input_df = self.create_features(input_df)

            # Ensure correct column order and fill any missing with 0
            for col in self.feature_names:
                if col not in input_df:
                    input_df[col] = 0

            # Apply preprocessing pipeline
            input_df = input_df[self.feature_names]
            input_imputed = pd.DataFrame(
                self.imputer.transform(input_df),
                columns=self.feature_names
            )
            input_scaled = pd.DataFrame(
                self.power_transformer.transform(input_imputed),
                columns=self.feature_names
            )

            return self.selector.transform(input_scaled)

        except Exception as e:
            raise ValueError(f"Preprocessing failed: {str(e)}")

    def predict(self, input_data):

        try:
            processed_data = self.preprocess(input_data)
            prediction = self.model.predict(processed_data)[0]
            probability = self.model.predict_proba(processed_data)[0][1]
            return {
                'prediction': int(prediction),
                'probability': float(probability),
                'status': 'success'
            }
        except ValueError as e:
            return {
                'prediction': None,
                'probability': None,
                'status': 'error',
                'message': str(e)
            }
        except Exception as e:
            return {
                'prediction': None,
                'probability': None,
                'status': 'error',
                'message': f"Unexpected error during prediction: {str(e)}"
            }
    def show_liver(self):        
        
        st.markdown("""
        <div class="header">
            <h1>🏥 Liver Disease Predictor</h1>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Patient Details")
            
            age = st.number_input("Age (years)", min_value=0, max_value=120, value=30)
            gender = st.selectbox("Gender", options=self.allowed_genders)
            total_bilirubin = st.number_input("Total Bilirubin (mg/dL)", min_value=0.0, value=0.7, step=0.1)
            direct_bilirubin = st.number_input("Direct Bilirubin (mg/dL)", min_value=0.0, value=0.2, step=0.1)
            alkaline_phosphotase = st.number_input("Alkaline Phosphotase (IU/L)", min_value=0, value=150)
            alamine_aminotransferase = st.number_input("Alamine Aminotransferase (ALT) (IU/L)", min_value=0, value=25)
            aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase (AST) (IU/L)", min_value=0, value=30)
            albumin = st.number_input("Albumin (g/dL)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)

            if st.button("Predict Liver Disease Risk"):
                try:
                    input_data = {
                        'Age': age,
                        'Gender': gender,
                        'Total_Bilirubin': total_bilirubin,
                        'Direct_Bilirubin': direct_bilirubin,
                        'Alkaline_Phosphotase': alkaline_phosphotase,
                        'Alamine_Aminotransferase': alamine_aminotransferase,
                        'Aspartate_Aminotransferase': aspartate_aminotransferase,
                        'Albumin': albumin
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
                            
                            # Add some interpretation
                            if result['prediction'] == 1:
                                st.warning("This result suggests potential liver disease risk. Please consult with a healthcare professional.")
                            else:
                                st.info("Results suggest low risk, but regular check-ups are still recommended.")
                        else:
                            st.error(f"Prediction failed: {result['message']}")
                            
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")