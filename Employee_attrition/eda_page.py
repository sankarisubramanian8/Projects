# eda_page.py
import streamlit as st
import pandas as pd
import plotly.express as px

def show_eda():
    """Interactive EDA page with proper HTML/CSS formatting"""
    # Page header with HTML styling
    st.markdown("""
    <div class="main-header">
        <h1>📊 Interactive EDA Dashboard</h1>
    </div>
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 600;
            color: #ffffff;
            text-align: center;
            margin-bottom: 1rem;
            background-color: #1E1E1E;
            padding: 1rem;
            border-radius: 10px;
        }
        .chart-container {
            background-color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)

    @st.cache_data
    def load_data():
        df = pd.read_csv(r'C:\Users\snbal\Downloads\Emp_attrition.csv')
        
        # Data preprocessing
        if 'WorkLifeBalance' in df.columns:
            wlb_mapping = {1: 'Poor', 2: 'Fair', 3: 'Good', 4: 'Excellent'}
            df['WorkLifeBalanceLevel'] = df['WorkLifeBalance'].map(wlb_mapping)
        
        return df

    df = load_data()
    
    # Chart selection
    chart_options = [
        "WorkLifeBalance by TotalWorkingYears",
        "Gender Distribution by Department",
        "EmployeeCount by Department"
    ]
    
    if 'chart_index' not in st.session_state:
        st.session_state.chart_index = 0

    # Navigation handlers
    def navigate(direction):
        if direction == "next":
            st.session_state.chart_index = (st.session_state.chart_index + 1) % len(chart_options)
        else:
            st.session_state.chart_index = (st.session_state.chart_index - 1) % len(chart_options)

    # Chart rendering
    def render_chart(selected_chart):
        if selected_chart == "WorkLifeBalance by TotalWorkingYears":
            wlb_data = df.groupby(['TotalWorkingYears', 'WorkLifeBalanceLevel']).size().reset_index(name='Count')
            fig = px.bar(
                wlb_data,
                x='TotalWorkingYears',
                y='Count',
                color='WorkLifeBalanceLevel',
                color_discrete_map={
                    'Poor': '#ff9999',
                    'Fair': '#66b3ff',
                    'Good': '#99ff99',
                    'Excellent': '#ffcc99'
                },
                title='<b>Work-Life Balance by Experience</b>'
            )
            
        elif selected_chart == "Gender Distribution by Department":
            gender_data = df.groupby(['Department', 'Gender']).size().unstack(fill_value=0)
            fig = px.bar(
                gender_data,
                barmode='group',
                title='<b>Gender Distribution Across Departments</b>',
                labels={'value': 'Employee Count'}
            )
            
        elif selected_chart == "EmployeeCount by Department":
            dept_data = df['Department'].value_counts().reset_index()
            fig = px.pie(
                dept_data,
                names='Department',
                values='count',
                title='<b>Employee Distribution by Department</b>'
            )
            
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='black')
        )
        return fig

    # UI Layout
    selected_chart = st.selectbox(
        "Select Analysis View:",
        chart_options,
        index=st.session_state.chart_index
    )
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(render_chart(selected_chart), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 10, 1])
    with col1:
        if st.button("◄", key="prev_btn"):
            navigate("previous")
    with col3:
        if st.button("►", key="next_btn"):
            navigate("next")
    with col2:
        st.markdown(f"<h3 style='text-align: center;'>{selected_chart}</h3>", unsafe_allow_html=True)