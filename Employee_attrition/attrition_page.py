# attrition_page.py
import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report



# Load model and encoders
@st.cache_resource
def load_model():
    try:
        with open('attrition_model.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        
        return {
            'model': artifacts['model'],
            'overtime_encoder': artifacts['overtime_encoder'],
            'y_encoder': artifacts['target_encoder'],
            'feature_names': artifacts['feature_names'],
            'test_data': artifacts.get('test_data', (None, None))  # Handle missing test data
        }
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

model_data = load_model()

# if model_data is None:
#     st.stop()  # Stop execution if model fails to load

# Extract components
model = model_data['model']
overtime_encoder = model_data['overtime_encoder']
y_encoder = model_data['y_encoder']
feature_names = model_data['feature_names']
x_test, y_test = model_data['test_data']

# Input validation
def validate_inputs(inputs):
    try:
        validated = {
            'OverTime': inputs['OverTime'],
            'Age': float(inputs['Age']),
            'MonthlyIncome': float(inputs['MonthlyIncome']),
            'DailyRate': float(inputs['DailyRate']),
            'TotalWorkingYears': float(inputs['TotalWorkingYears']),
            'MonthlyRate': float(inputs['MonthlyRate'])
        }
        
        # Additional validation
        if validated['Age'] < 18 or validated['Age'] > 70:
            st.warning("Age seems unusual for an employee")
        if validated['MonthlyIncome'] < 0:
            st.error("Income cannot be negative")
            
        return validated
    except ValueError as e:
        st.error(f"Invalid input: {str(e)}")
        return None

# Main function
def show():
    st.markdown("""
    <div class="header">
        <h1>📊 Attrition Risk Prediction Dashboard</h1>
        <p>Predict whether an employee is likely to leave the company</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Employee Details")
        inputs = {}
        
        st.markdown('<p class="input-label">OverTime Status</p>', unsafe_allow_html=True)
        inputs['OverTime'] = st.selectbox(
            "Select OverTime Status",
            options=["No", "Yes"],
            label_visibility="collapsed"
        )
        
        st.markdown('<p class="input-label">Age</p>', unsafe_allow_html=True)
        inputs['Age'] = st.number_input(
            "Enter Age", 
            min_value=18, max_value=70, value=30,
            label_visibility="collapsed"
        )
        
        st.markdown('<p class="input-label">Monthly Income</p>', unsafe_allow_html=True)
        inputs['MonthlyIncome'] = st.number_input(
            "Enter Monthly Income",
            min_value=0, value=5000,
            label_visibility="collapsed"
        )
        
        st.markdown('<p class="input-label">Daily Rate</p>', unsafe_allow_html=True)
        inputs['DailyRate'] = st.number_input(
            "Enter Daily Rate",
            min_value=0, value=200,
            label_visibility="collapsed"
        )
        
        st.markdown('<p class="input-label">Total Working Years</p>', unsafe_allow_html=True)
        inputs['TotalWorkingYears'] = st.number_input(
            "Enter Total Working Years",
            min_value=0, max_value=50, value=5,
            label_visibility="collapsed"
        )
        
        st.markdown('<p class="input-label">Monthly Rate</p>', unsafe_allow_html=True)
        inputs['MonthlyRate'] = st.number_input(
            "Enter Monthly Rate",
            min_value=0, value=20000,
            label_visibility="collapsed"
        )

        if st.button("Predict Attrition"):
            with st.spinner('Analyzing...'):
                validated = validate_inputs(inputs)
                
                if validated:
                    # Prepare input data in correct feature order
                    input_data = [
                        overtime_encoder.transform([validated['OverTime']])[0],
                        validated['Age'],
                        validated['MonthlyIncome'],
                        validated['DailyRate'],
                        validated['TotalWorkingYears'],
                        validated['MonthlyRate']
                    ]
                    
                    # Convert to DataFrame to ensure feature order matches training
                    input_df = pd.DataFrame([input_data], columns=feature_names)
                    
                    # Make prediction
                    prediction = model.predict(input_df)[0]
                    result = "Yes" if prediction == 1 else "No"
                    
                    if y_encoder:  # Handle case where target was encoded
                        result = y_encoder.inverse_transform([prediction])[0]

                    with col2:
                        st.markdown("### Prediction Result")
                        if result == "Yes":
                            st.error(f"🚨 High Risk: Employee is likely to leave (Attrition: {result})")
                        else:
                            st.success(f"✅ Low Risk: Employee is likely to stay (Attrition: {result})")
                        
                        # Show model metrics if test data is available
                        if x_test is not None and y_test is not None:
                            with st.expander("Model Performance Metrics"):
                                y_pred = model.predict(x_test)
                                st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
                                st.write("**Classification Report:**")
                                st.code(classification_report(y_test, y_pred))

if __name__ == "__main__":
    show()