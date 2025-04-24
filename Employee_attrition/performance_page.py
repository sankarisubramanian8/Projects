# performance_page.py
import streamlit as st
import pickle
import pandas as pd

@st.cache_resource
def load_model():
    try:
        with open('performance_model.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        
        return {
            'model': artifacts['model'],
            'jobrole_encoder': artifacts['jobrole_encoder'],
            'valid_job_roles': artifacts['valid_job_roles'],
            'feature_names': artifacts['feature_names']
        }
    except Exception as e:
        st.error(f"""
        Failed to load performance model: {str(e)}
        
        Please ensure:
        1. You've run train_models.py first
        2. performance_model.pkl exists
        3. The file is not corrupted
        """)
        st.stop()

model_data = load_model()

def show():
    st.markdown("""
    <div class="header">
        <h1>📈 Employee Performance Rating Predictor</h1>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Employee Details")
        
        age = st.number_input("Age (years)", min_value=18, max_value=70, value=30)
        working_years = st.number_input("Total Working Years", min_value=0, max_value=50, value=5)
        income = st.number_input("Monthly Income", min_value=0, value=5000)
        salary_hike = st.number_input("Percent Salary Hike", min_value=0, max_value=100, value=10)
        job_role = st.selectbox("Job Role", options=model_data['valid_job_roles'])

        if st.button("Predict Performance Rating"):
            try:
                input_data = pd.DataFrame({
                    'Age': [age],
                    'TotalWorkingYears': [working_years],
                    'MonthlyIncome': [income],
                    'PercentSalaryHike': [salary_hike],
                    'JobRole': [job_role]
                })
                
                # Encode JobRole
                input_data['JobRole'] = model_data['jobrole_encoder'].transform(input_data['JobRole'])
                
                # Reorder columns to match training
                input_data = input_data[model_data['feature_names']]
                
                # Predict
                prediction = model_data['model'].predict(input_data)[0]
                
                with col2:
                    st.markdown("### Prediction Result")
                    st.success(f"Predicted Performance Rating: {prediction}")
                    
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    show()