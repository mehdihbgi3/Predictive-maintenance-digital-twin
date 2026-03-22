import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime
from sqlalchemy import create_engine
from pathlib import Path
import yaml

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def get_engine():
    config_path = Path("F:/predictive-maintenance-digital-twin/configs/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['database']
    connection_string = f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['name']}"
    return create_engine(connection_string)

@st.cache_data(ttl=300)
def load_overview_data():
    engine = get_engine()
    query = """
    SELECT 
        COUNT(*) as total_records,
        COUNT(DISTINCT serial_number) as unique_drives,
        COUNT(DISTINCT model) as unique_models,
        SUM(failure) as total_failures,
        MIN(date) as first_date,
        MAX(date) as last_date
    FROM hard_drive.smart_data
    """
    return pd.read_sql(query, engine)

@st.cache_data(ttl=300)
def load_failure_by_model():
    engine = get_engine()
    query = """
    SELECT 
        model,
        COUNT(DISTINCT serial_number) as total_drives,
        SUM(failure) as failures,
        ROUND(100.0 * SUM(failure) / COUNT(DISTINCT serial_number), 2) as failure_rate
    FROM hard_drive.smart_data
    GROUP BY model
    HAVING COUNT(DISTINCT serial_number) >= 100
    ORDER BY failure_rate DESC
    LIMIT 15
    """
    return pd.read_sql(query, engine)

@st.cache_data(ttl=300)
def load_daily_failures():
    engine = get_engine()
    query = """
    SELECT 
        date,
        COUNT(DISTINCT serial_number) as active_drives,
        SUM(failure) as failures
    FROM hard_drive.smart_data
    GROUP BY date
    ORDER BY date
    """
    return pd.read_sql(query, engine)

@st.cache_data(ttl=300)
def load_smart_comparison():
    engine = get_engine()
    query = """
    SELECT 
        CASE WHEN failure = 1 THEN 'Failed' ELSE 'Healthy' END as status,
        AVG(smart_5_raw) as reallocated_sectors,
        AVG(smart_187_raw) as uncorrectable_errors,
        AVG(smart_197_raw) as pending_sectors,
        AVG(smart_198_raw) as offline_uncorrectable,
        AVG(smart_194_raw) as temperature
    FROM hard_drive.smart_data
    GROUP BY CASE WHEN failure = 1 THEN 'Failed' ELSE 'Healthy' END
    """
    return pd.read_sql(query, engine)

@st.cache_data(ttl=300)
def load_capacity_analysis():
    engine = get_engine()
    query = """
    SELECT 
        CASE 
            WHEN capacity_bytes < 4000000000000 THEN '< 4TB'
            WHEN capacity_bytes < 8000000000000 THEN '4-8TB'
            WHEN capacity_bytes < 12000000000000 THEN '8-12TB'
            WHEN capacity_bytes < 16000000000000 THEN '12-16TB'
            ELSE '16TB+'
        END as capacity_range,
        COUNT(DISTINCT serial_number) as drives,
        SUM(failure) as failures
    FROM hard_drive.smart_data
    WHERE capacity_bytes IS NOT NULL
    GROUP BY 1
    ORDER BY MIN(capacity_bytes)
    """
    return pd.read_sql(query, engine)

API_URL = "http://localhost:8080"

def predict_drive(data):
    try:
        response = requests.post(f"{API_URL}/predict", json=data, timeout=10)
        return response.json()
    except:
        return None

st.sidebar.title("🔧 Predictive Maintenance")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Overview", "📈 Analytics", "🔮 Predict", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Performance**")
st.sidebar.metric("ROC AUC", "0.886")
st.sidebar.metric("Recall", "77%")
st.sidebar.markdown("---")
st.sidebar.markdown("*Built with Streamlit*")

if page == "📊 Overview":
    st.title("📊 Dataset Overview")
    st.markdown("Real-time statistics from the Backblaze hard drive dataset.")
    
    overview = load_overview_data()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{overview['total_records'].values[0]:,}")
    with col2:
        st.metric("Unique Drives", f"{overview['unique_drives'].values[0]:,}")
    with col3:
        st.metric("Drive Models", f"{overview['unique_models'].values[0]}")
    with col4:
        st.metric("Total Failures", f"{overview['total_failures'].values[0]:,}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Failure Rate by Model")
        model_data = load_failure_by_model()
        fig = px.bar(
            model_data.head(10), 
            x='failure_rate', 
            y='model',
            orientation='h',
            color='failure_rate',
            color_continuous_scale='Reds'
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Drives by Capacity")
        capacity_data = load_capacity_analysis()
        fig = px.pie(
            capacity_data, 
            values='drives', 
            names='capacity_range',
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Daily Failure Trend")
    daily_data = load_daily_failures()
    fig = px.line(
        daily_data, 
        x='date', 
        y='failures',
        title='Failures Over Time'
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


elif page == "📈 Analytics":
    st.title("📈 Analytics & Insights")
    
    st.subheader("SMART Values: Failed vs Healthy Drives")
    
    smart_data = load_smart_comparison()
    
    metrics = ['reallocated_sectors', 'uncorrectable_errors', 'pending_sectors', 'offline_uncorrectable']
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        'Reallocated Sectors', 'Uncorrectable Errors', 
        'Pending Sectors', 'Offline Uncorrectable'
    ])
    
    colors = {'Failed': '#EF553B', 'Healthy': '#00CC96'}
    
    for i, metric in enumerate(metrics):
        row = i // 2 + 1
        col = i % 2 + 1
        
        for status in ['Failed', 'Healthy']:
            val = smart_data[smart_data['status'] == status][metric].values[0]
            fig.add_trace(
                go.Bar(name=status, x=[status], y=[val], marker_color=colors[status], showlegend=(i==0)),
                row=row, col=col
            )
    
    fig.update_layout(height=500, barmode='group')
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    failed = smart_data[smart_data['status'] == 'Failed']
    healthy = smart_data[smart_data['status'] == 'Healthy']
    
    with col1:
        ratio = failed['reallocated_sectors'].values[0] / max(healthy['reallocated_sectors'].values[0], 1)
        st.metric("Reallocated Sectors Ratio", f"{ratio:.0f}x", "higher in failed drives")
    
    with col2:
        ratio = failed['pending_sectors'].values[0] / max(healthy['pending_sectors'].values[0], 1)
        st.metric("Pending Sectors Ratio", f"{ratio:.0f}x", "higher in failed drives")
    
    with col3:
        ratio = failed['uncorrectable_errors'].values[0] / max(healthy['uncorrectable_errors'].values[0], 1)
        st.metric("Uncorrectable Errors Ratio", f"{ratio:.0f}x", "higher in failed drives")


elif page == "🔮 Predict":
    st.title("🔮 Failure Prediction")
    st.markdown("Enter SMART attributes to predict failure probability.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Drive Information")
        serial = st.text_input("Serial Number", "DRIVE_001")
        model = st.text_input("Model", "ST12000NM0008")
        capacity = st.number_input("Capacity (TB)", value=12.0, min_value=0.0)
        age = st.number_input("Age (days)", value=365, min_value=0)
        power_hours = st.number_input("Power-on Hours", value=8760, min_value=0)
    
    with col2:
        st.subheader("SMART Attributes")
        smart_5 = st.number_input("SMART 5 (Reallocated Sectors)", value=0, min_value=0)
        smart_187 = st.number_input("SMART 187 (Uncorrectable Errors)", value=0, min_value=0)
        smart_197 = st.number_input("SMART 197 (Pending Sectors)", value=0, min_value=0)
        smart_198 = st.number_input("SMART 198 (Offline Uncorrectable)", value=0, min_value=0)
        smart_194 = st.number_input("SMART 194 (Temperature °C)", value=35, min_value=0, max_value=100)
    
    if st.button("🔮 Predict Failure", type="primary"):
        data = {
            "serial_number": serial,
            "model": model,
            "capacity_tb": capacity,
            "age_days": age,
            "power_on_hours": power_hours,
            "smart_5_raw": smart_5,
            "smart_9_raw": power_hours,
            "smart_187_raw": smart_187,
            "smart_188_raw": 0,
            "smart_194_raw": smart_194,
            "smart_197_raw": smart_197,
            "smart_198_raw": smart_198,
            "smart_199_raw": 0
        }
        
        with st.spinner("Analyzing..."):
            result = predict_drive(data)
        
        if result:
            st.markdown("---")
            st.subheader("Prediction Result")
            
            prob = result['failure_probability']
            risk = result['risk_level']
            
            if risk == "CRITICAL":
                color = "🔴"
            elif risk == "HIGH":
                color = "🟠"
            elif risk == "MEDIUM":
                color = "🟡"
            else:
                color = "🟢"
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Failure Probability", f"{prob*100:.1f}%")
            with col2:
                st.metric("Risk Level", f"{color} {risk}")
            with col3:
                st.metric("30-Day Failure", "Yes" if result['will_fail_30d'] else "No")
            
            st.info(f"**Recommendation:** {result['recommendation']}")
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                title = {'text': "Failure Probability (%)"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 10], 'color': "lightgreen"},
                        {'range': [10, 30], 'color': "yellow"},
                        {'range': [30, 50], 'color': "orange"},
                        {'range': [50, 70], 'color': "orangered"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Failed to connect to prediction API. Make sure it's running on port 8080.")


elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    
    st.markdown("""
    ## Predictive Maintenance Digital Twin
    
    This project demonstrates a production-grade predictive maintenance system 
    for hard drive failure prediction using the Backblaze dataset.
    
    ### Features
    - **86+ million records** from real-world hard drive data
    - **Machine Learning models** (XGBoost, LightGBM) with 88.6% ROC AUC
    - **Statistical analysis** including Weibull reliability modeling
    - **REST API** for real-time predictions
    - **Interactive dashboard** for data exploration
    
    ### Technology Stack
    - **Database:** PostgreSQL
    - **ML/Data:** Python, Pandas, Scikit-learn, XGBoost
    - **API:** FastAPI
    - **Dashboard:** Streamlit, Plotly
    - **Statistical:** Weibull Analysis, Kaplan-Meier Survival
    
    ### Key Findings
    - **SMART 197** (Pending Sectors) is the strongest predictor
    - **SMART 5** (Reallocated Sectors) shows 96x higher values in failed drives
    - **Weibull β > 1** indicates wear-out failure pattern
    - **B10 Life:** 722 days (10% of drives fail within 2 years)
    
    ### Author
    **Mehdi Hassanbeigi** 
    """)

st.sidebar.markdown("---")
st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
