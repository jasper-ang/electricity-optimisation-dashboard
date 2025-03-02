import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta


def plot_energy_consumption(data):
    """
    Plot energy consumption over time

    Parameters:
    - data: DataFrame with energy consumption data

    Returns:
    - Plotly figure object
    """
    # Group by date and component type
    daily_by_component = data.groupby(['date', 'component_type'])[
        'energy_consumption'].sum().reset_index()

    # Create stacked area chart
    fig = px.area(
        daily_by_component,
        x='date',
        y='energy_consumption',
        color='component_type',
        title='Daily Energy Consumption by Component Type',
        labels={
            'energy_consumption': 'Energy Consumption (kWh)', 'date': 'Date'},
        color_discrete_sequence=px.colors.qualitative.Plotly
    )

    # Add overall trend line
    daily_total = data.groupby(
        'date')['energy_consumption'].sum().reset_index()
    fig.add_trace(
        go.Scatter(
            x=daily_total['date'],
            y=daily_total['energy_consumption'],
            mode='lines',
            line=dict(color='black', width=2, dash='dot'),
            name='Total Consumption'
        )
    )

    # Improve layout
    fig.update_layout(
        legend=dict(orientation='h', yanchor='bottom',
                    y=1.02, xanchor='right', x=1),
        height=500,
        hovermode='x unified'
    )

    return fig


def plot_savings_potential(data, optimization_target):
    """
    Plot potential savings by component type

    Parameters:
    - data: DataFrame with energy consumption data
    - optimization_target: Target optimization percentage

    Returns:
    - Plotly figure object
    """
    # Group by component type
    consumption_by_type = data.groupby('component_type').agg({
        'energy_consumption': 'sum',
        'efficiency_score': 'mean'
    }).reset_index()

    # Calculate potential savings based on consumption and efficiency
    consumption_by_type['potential_savings'] = consumption_by_type['energy_consumption'] * \
        (optimization_target / 100) * \
        ((100 -
          consumption_by_type['efficiency_score']) / 50)

    # Create bar chart
    fig = go.Figure()

    # Add current consumption bars
    fig.add_trace(go.Bar(
        x=consumption_by_type['component_type'],
        y=consumption_by_type['energy_consumption'],
        name='Current Consumption',
        marker_color='lightslategray'
    ))

    # Add optimized consumption bars
    fig.add_trace(go.Bar(
        x=consumption_by_type['component_type'],
        y=consumption_by_type['energy_consumption'] -
        consumption_by_type['potential_savings'],
        name='Optimized Consumption',
        marker_color='mediumseagreen'
    ))

    # Improve layout
    fig.update_layout(
        title='Energy Consumption and Potential Savings by Component Type',
        xaxis_title='Component Type',
        yaxis_title='Energy Consumption (kWh)',
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom',
                    y=1.02, xanchor='right', x=1),
        height=500
    )

    return fig


def plot_network_heatmap(data):
    """
    Create a heatmap of energy consumption by component and facility

    Parameters:
    - data: DataFrame with energy consumption data

    Returns:
    - Plotly figure object
    """
    # Group by facility and component type
    heatmap_data = data.groupby(['facility', 'component_type'])[
        'energy_consumption'].sum().reset_index()

    # Pivot the data for heatmap format
    heatmap_pivot = heatmap_data.pivot(
        index='facility', columns='component_type', values='energy_consumption')

    # Create heatmap
    fig = px.imshow(
        heatmap_pivot,
        text_auto='.1f',
        labels=dict(x="Component Type", y="Facility",
                    color="Energy Consumption (kWh)"),
        color_continuous_scale='RdYlGn_r',
        title='Energy Consumption Heatmap by Facility and Component Type'
    )

    # Improve layout
    fig.update_layout(
        height=500,
        xaxis={'side': 'top'},
        # Added more top margin to create space between title and content
        margin=dict(t=120)
    )

    return fig
