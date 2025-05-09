import streamlit as st
from liver_prediction import LiverDiseasePredictor
from Kidney_prediction import KidneyDiseasePredictor
from parkinsons import ParkinsonsPredictor
# Set page to wide layout
st.set_page_config(layout="wide")

# Custom CSS for full-width equal tabs with minimal styling
st.markdown("""
<style>
    /* Remove all default padding/margins */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Full-width equal distribution */
    .stTabs [data-baseweb="tab"] {
        flex: 1 !important;
        text-align: center !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        padding: 12px 0 !important;
        margin: 0 !important;
        border: none !important;
        border-radius: 0 !important;
        background: #f0f2f6 !important;
    }
    
    /* Active tab styling */
    .stTabs [aria-selected="true"] {
        background: #000 !important;
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* Remove tab highlight bar */
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }
    
    /* Remove container padding */
    .stTabs {
        padding: 0 !important;
    }
</style>
""", unsafe_allow_html=True)
liver_predictor = LiverDiseasePredictor()
obj2= KidneyDiseasePredictor()
obj3 = ParkinsonsPredictor()

# Create full-width tabs
tab1, tab2, tab3 = st.tabs([
    "Liver Prediction", 
    "Kidney Prediction",
    "Parkinsons Prediction"
])

# Tab 1 Content
with tab1:
    st.header("Liver Prediction")
    liver_predictor.show_liver()

# Tab 2 Content
with tab2:
    st.header("Kidney Prediction")
    obj2.show_kidney()

# Tab 3 Content  
with tab3:
    st.header("Parkinsons Prediction")
    obj3.show_parkinsons()
