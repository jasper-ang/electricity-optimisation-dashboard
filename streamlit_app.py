from utils.data_loader import load_network_data, preprocess_data
from utils.ai_models import predict_optimizations, calculate_savings, forecast_consumption
from utils.visualization import plot_energy_consumption, plot_savings_potential, plot_network_heatmap
import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import math
from pathlib import Path

# Add the utils folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Import utility modules

# Page configuration
st.set_page_config(
    page_title="AI-Powered Network Energy Optimization",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86C1;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2874A6;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .good-metric {
        color: #28a745;
    }
    .warning-metric {
        color: #ffc107;
    }
    .danger-metric {
        color: #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("# ⚙️ Controls")
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(datetime.now() - timedelta(days=30), datetime.now())
)

facility_options = ["All Facilities", "Data Center A",
                    "Data Center B", "Edge Location C", "Edge Location D"]
selected_facility = st.sidebar.selectbox("Select Facility", facility_options)

network_components = ["Routers", "Switches", "Servers", "Storage", "Cooling"]
selected_components = st.sidebar.multiselect(
    "Filter Components", network_components, default=network_components)

optimization_target = st.sidebar.slider(
    "Optimization Target (%)", min_value=5, max_value=30, value=15, step=5)

# Main page
st.markdown("<h1 class='main-header'>AI-Powered Network Energy Optimization</h1>",
            unsafe_allow_html=True)

# Load data
data = load_network_data()
filtered_data = preprocess_data(
    data, date_range, selected_facility, selected_components)

# Top metrics
col1, col2, col3, col4 = st.columns(4)

total_consumption = filtered_data['energy_consumption'].sum()
avg_efficiency = filtered_data['efficiency_score'].mean()
total_cost = total_consumption * 0.12  # Assuming $0.12 per kWh
carbon_footprint = total_consumption * 0.85  # Assuming 0.85 kg CO2 per kWh

with col1:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("Total Energy Consumption", f"{total_consumption:,.2f} kWh")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    efficiency_color = "good-metric" if avg_efficiency > 80 else "warning-metric" if avg_efficiency > 60 else "danger-metric"
    st.markdown(
        f"<h3>Efficiency Score</h3><h2 class='{efficiency_color}'>{avg_efficiency:.1f}%</h2>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("Energy Cost", f"${total_cost:,.2f}")
    st.markdown("</div>", unsafe_allow_html=True)

with col4:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("Carbon Footprint", f"{carbon_footprint:,.2f} kg CO2")
    st.markdown("</div>", unsafe_allow_html=True)

# Energy consumption over time
st.markdown("<h2 class='sub-header'>Energy Consumption Analysis</h2>",
            unsafe_allow_html=True)
time_fig = plot_energy_consumption(filtered_data)
st.plotly_chart(time_fig, use_container_width=True)

# Network optimization
st.markdown("<h2 class='sub-header'>AI-Powered Optimization Recommendations</h2>",
            unsafe_allow_html=True)
optimizations = predict_optimizations(filtered_data, optimization_target)

col1, col2 = st.columns([2, 1])

with col1:
    network_heatmap = plot_network_heatmap(filtered_data)
    st.plotly_chart(network_heatmap, use_container_width=True)

with col2:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("<h3>Top Optimization Opportunities</h3>",
                unsafe_allow_html=True)

    for i, (component, saving) in enumerate(optimizations.items()):
        st.markdown(f"**{i+1}. {component}**")
        st.markdown(f"Potential Savings: {saving:.2f} kWh/day")
        st.markdown(f"Annual Cost Reduction: ${saving * 0.12 * 365:,.2f}")
        st.markdown("---")

    st.markdown("</div>", unsafe_allow_html=True)

# Savings potential
st.markdown("<h2 class='sub-header'>Projected Savings</h2>",
            unsafe_allow_html=True)
savings_fig = plot_savings_potential(filtered_data, optimization_target)
st.plotly_chart(savings_fig, use_container_width=True)

# Forecast analysis
st.markdown("<h2 class='sub-header'>Energy Consumption Forecast</h2>",
            unsafe_allow_html=True)
forecast_days = st.slider("Forecast Days", min_value=7,
                          max_value=90, value=30, step=7)
with_optimization = st.checkbox("Show with optimization applied", value=True)

forecast_data = forecast_consumption(
    filtered_data, forecast_days, with_optimization, optimization_target)


# Create forecast figure
forecast_fig = go.Figure()

forecast_fig.add_trace(go.Scatter(
    x=forecast_data['date'],
    y=forecast_data['original_forecast'],
    mode='lines',
    name='Baseline Forecast',
    line=dict(color='#1F77B4', width=2)
))

if with_optimization:
    forecast_fig.add_trace(go.Scatter(
        x=forecast_data['date'],
        y=forecast_data['optimized_forecast'],
        mode='lines',
        name='Optimized Forecast',
        line=dict(color='#2CA02C', width=2)
    ))

forecast_fig.update_layout(
    title='Energy Consumption Forecast',
    xaxis_title='Date',
    yaxis_title='Energy Consumption (kWh)',
    legend=dict(orientation='h', yanchor='bottom',
                y=1.02, xanchor='right', x=1),
    height=500
)

st.plotly_chart(forecast_fig, use_container_width=True)

# Summary of potential savings
if with_optimization:
    original_total = forecast_data['original_forecast'].sum()
    optimized_total = forecast_data['optimized_forecast'].sum()
    savings = original_total - optimized_total
    savings_percent = (savings / original_total) * 100
    cost_savings = savings * 0.12

    st.markdown(f"""
    ### Optimization Impact Summary
    - Total forecasted consumption: {original_total:,.2f} kWh
    - With optimizations applied: {optimized_total:,.2f} kWh
    - Potential savings: {savings:,.2f} kWh ({savings_percent:.1f}%)
    - Estimated cost savings: ${cost_savings:,.2f}
    """)

# Additional insights and recommendations
st.markdown("<h2 class='sub-header'>AI Insights</h2>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("### Anomaly Detection")

    # Example anomalies for demonstration
    anomalies = [
        {"component": "Router R7", "date": "2023-10-05",
            "consumption": "142 kWh", "expected": "95 kWh"},
        {"component": "Cooling Unit C3", "date": "2023-10-12",
            "consumption": "205 kWh", "expected": "160 kWh"}
    ]

    for anomaly in anomalies:
        st.markdown(f"**{anomaly['component']}** ({anomaly['date']})")
        st.markdown(
            f"Consumed {anomaly['consumption']} vs expected {anomaly['expected']}")
        st.markdown("---")

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("### Intelligent Recommendations")

    # Example recommendations for demonstration
    recommendations = [
        "Implement dynamic cooling adjustments based on server load patterns",
        "Consolidate network traffic to fewer routers during low-usage periods (11 PM - 5 AM)",
        "Upgrade switch firmware on 'Switch S9' to improve power management capability"
    ]

    for i, rec in enumerate(recommendations):
        st.markdown(f"**{i+1}.** {rec}")

    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("© 2023 AI-Powered Network Energy Optimization | Dashboard v1.0")
