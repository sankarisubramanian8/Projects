import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st

# Set page config
st.set_page_config(
    page_title="Attrition Predictor",
    page_icon="📊",
    layout="wide"
)

# Custom CSS styling - Soft Blue-Grey Theme
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        background-color: #ECEFF1 !important;
        color: #333333 !important;
    }
    .stApp {
        background-color: #ECEFF1 !important;
    }
    .header {
        color: #333333;
        font-family: 'Arial', sans-serif;
        text-align: center;
    }
    .input-label {
        font-weight: bold !important;
        font-style: italic !important;
        color: #1E88E5 !important;
        margin-bottom: 5px !important;
    }
    /* Input fields */
    .stTextInput>div>div>input, 
    .stSelectbox>div>div>select {
        background-color: #FFFFFF !important;
        color: #333333 !important;
        border: 1px solid #B0BEC5 !important;
        border-radius: 5px !important;
    }
    /* Focus state */
    .stTextInput>div>div>input:focus, 
    .stSelectbox>div>div>select:focus {
        border: 2px solid #1E88E5 !important;
        box-shadow: 0 0 5px #1E88E5 !important;
    }
    /* Placeholder text */
    .stTextInput>div>div>input::placeholder {
        color: #90A4AE !important;
    }
    /* Button styling */
    .stButton>button {
        background-color: #1E88E5 !important;
        color: white !important;
        border-radius: 5px !important;
        border: none !important;
        padding: 10px 24px !important;
        font-weight: bold !important;
        width: 100% !important;
    }
    .stButton>button:hover {
        background-color: #1565C0 !important;
    }
    /* Result boxes */
    .stSuccess {
        background-color: #E8F5E9 !important;
        color: #2E7D32 !important;
        border-radius: 5px !important;
        padding: 15px !important;
    }
    .stError {
        background-color: #FFEBEE !important;
        color: #C62828 !important;
        border-radius: 5px !important;
        padding: 15px !important;
    }
    /* Expander styling */
    .stExpander {
        border: 1px solid #CFD8DC !important;
        border-radius: 5px !important;
        background-color: #FFFFFF !important;
    }
    .streamlit-expanderHeader {
        background-color: #FFFFFF !important;
        color: #333333 !important;
    }
    /* Custom container for input fields */
    .stContainer {
        background-color: #FFFFFF !important;
        border-radius: 10px !important;
        padding: 20px !important;
        margin-bottom: 20px !important;
        border: 1px solid #CFD8DC !important;
    }
    /* Metric styling */
    .stMetric {
        background-color: #FFFFFF !important;
        border-radius: 5px !important;
        padding: 10px !important;
        border: 1px solid #CFD8DC !important;
    }
    </style>
    """, unsafe_allow_html=True)



# -------------------- 1. MODEL TRAINING --------------------
@st.cache_resource
def train_model():
    # Load data
    df = pd.read_csv(r'E:/Emp_attrition.csv')

    # Features and target
    x = df[['OverTime', 'Age', 'MonthlyIncome', 'DailyRate', 'TotalWorkingYears', 'MonthlyRate']]
    y = df['Attrition']

    # Initialize encoders
    overtime_encoder = LabelEncoder()
    x['OverTime'] = overtime_encoder.fit_transform(x['OverTime'])

    # Encode target if needed
    y_encoder = LabelEncoder() if y.dtype == 'object' else None
    if y_encoder:
        y = y_encoder.fit_transform(y)

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train model
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced']
    }
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(x_train, y_train)
    
    return grid_search.best_estimator_, overtime_encoder, y_encoder, x.columns.tolist(), (x_test, y_test)

# -------------------- 2. INPUT VALIDATION --------------------
def validate_inputs(inputs):
    try:
        return {
            'OverTime': inputs['OverTime'],
            'Age': float(inputs['Age']),
            'MonthlyIncome': float(inputs['MonthlyIncome']),
            'DailyRate': float(inputs['DailyRate']),
            'TotalWorkingYears': float(inputs['TotalWorkingYears']),
            'MonthlyRate': float(inputs['MonthlyRate'])
        }
    except ValueError:
        return None

# -------------------- 3. STREAMLIT APP --------------------
def main():
    # Header with icon
    st.markdown("""
    <div class="header">
        <h1>📊 Employee Attrition Prediction System</h1>
        <p>Predict whether an employee is likely to leave the company</p>
    </div>
    """, unsafe_allow_html=True)

    # Load model (cached)
    model, overtime_encoder, y_encoder, feature_names, (x_test, y_test) = train_model()

    # Create two columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Employee Details")
        
        # Container for input fields
        with st.container():
            # Input fields with proper labels and styling
            inputs = {}
            
            st.markdown('<p class="input-label">OverTime Status</p>', unsafe_allow_html=True)
            inputs['OverTime'] = st.selectbox(
                "Select OverTime Status",
                options=["No", "Yes"],
                label_visibility="collapsed",
                key='overtime'
            )
            
            st.markdown('<p class="input-label">Age</p>', unsafe_allow_html=True)
            inputs['Age'] = st.text_input(
                "Enter Age",
                label_visibility="collapsed",
                key='age'
            )
            
            st.markdown('<p class="input-label">Monthly Income ($)</p>', unsafe_allow_html=True)
            inputs['MonthlyIncome'] = st.text_input(
                "Enter Monthly Income",
                label_visibility="collapsed",
                key='income'
            )
            
            st.markdown('<p class="input-label">Daily Rate ($)</p>', unsafe_allow_html=True)
            inputs['DailyRate'] = st.text_input(
                "Enter Daily Rate",
                label_visibility="collapsed",
                key='daily'
            )
            
            st.markdown('<p class="input-label">Total Working Years</p>', unsafe_allow_html=True)
            inputs['TotalWorkingYears'] = st.text_input(
                "Enter Total Working Years",
                label_visibility="collapsed",
                key='years'
            )
            
            st.markdown('<p class="input-label">Monthly Rate ($)</p>', unsafe_allow_html=True)
            inputs['MonthlyRate'] = st.text_input(
                "Enter Monthly Rate",
                label_visibility="collapsed",
                key='monthly'
            )

        # Predict button
        if st.button("Predict Attrition", key="predict_button"):
            with st.spinner('Analyzing...'):
                # Validate and convert inputs
                validated = validate_inputs(inputs)
                
                if validated is None:
                    st.error("❗ Please enter valid numbers for all fields")
                else:
                    # Prepare input data
                    input_data = [
                        overtime_encoder.transform([validated['OverTime']])[0],
                        validated['Age'],
                        validated['MonthlyIncome'],
                        validated['DailyRate'],
                        validated['TotalWorkingYears'],
                        validated['MonthlyRate']
                    ]
                    
                    # Make prediction
                    prediction = model.predict([input_data])[0]
                    result = y_encoder.inverse_transform([prediction])[0] if y_encoder else "Yes" if prediction == 1 else "No"

                    # Display results in right column
                    with col2:
                        st.markdown("### Prediction Result")
                        if result == "Yes":
                            st.error(f"🚨 High Risk: Employee is likely to leave (Attrition: {result})")
                        else:
                            st.success(f"✅ Low Risk: Employee is likely to stay (Attrition: {result})")
                        
                        # Performance metrics
                        with st.expander("View Model Performance Metrics"):
                            y_pred = model.predict(x_test)
                            st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
                            st.write("**Classification Report:**")
                            st.code(classification_report(y_test, y_pred))

# Run the app
if __name__ == "__main__":
    main()
