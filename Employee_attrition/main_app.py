import streamlit as st
# Set page config
st.set_page_config(
    page_title="Employee Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

from attrition_page import show as show_attrition
from performance_page import show as show_performance
from eda_page import show_eda 


# Sidebar navigation
def sidebar_navigation():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Select Analysis", 
                               ["Employee Attrition", 
                                "Performance Rating", 
                                "Interactive Workforce Insight"])
    return app_mode

# Main app
def main():
    app_mode = sidebar_navigation()
    
    if app_mode == "Employee Attrition":
        show_attrition()
    elif app_mode == "Performance Rating":
        show_performance()
    elif app_mode == "Interactive Workforce Insight":
        show_eda()

if __name__ == "__main__":
    main()