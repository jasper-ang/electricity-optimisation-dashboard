
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, IsolationForest


def predict_optimizations(data, target_percentage=15):
    """
    Predict optimization opportunities in the network data

    Parameters:
    - data: DataFrame with network energy data
    - target_percentage: Target optimization percentage

    Returns:
    - Dictionary of components with highest optimization potential and estimated savings
    """
    # Group by component type and component ID, calculating average consumption
    component_consumption = data.groupby(['component_type', 'component_id']).agg({
        'energy_consumption': 'mean',
        'efficiency_score': 'mean',
        'utilization': 'mean'
    }).reset_index()

    # Calculate optimization potential based on efficiency score and utilization
    component_consumption['optimization_potential'] = (100 - component_consumption['efficiency_score']) / 100 * \
        component_consumption['energy_consumption']

    # Additional factor: low utilization with high energy consumption suggests waste
    component_consumption['utilization_factor'] = (1 - component_consumption['utilization'] / 100) * \
        component_consumption['energy_consumption'] * 0.3

    component_consumption['total_potential'] = component_consumption['optimization_potential'] + \
        component_consumption['utilization_factor']

    # Sort by total potential in descending order
    top_components = component_consumption.sort_values(
        'total_potential', ascending=False).head(5)

    # Create dictionary of components and their potential savings
    optimizations = {}
    for _, row in top_components.iterrows():
        component_name = f"{row['component_type']} {row['component_id']}"
        # Calculate savings based on target percentage of consumption, adjusted by optimization potential
        savings = row['energy_consumption'] * (target_percentage / 100) * \
            (row['optimization_potential'] / row['energy_consumption'])
        optimizations[component_name] = savings

    return optimizations


def calculate_savings(data, optimization_target):
    """
    Calculate potential energy savings based on optimization target

    Parameters:
    - data: DataFrame with network energy data
    - optimization_target: Target optimization percentage

    Returns:
    - Dictionary with savings metrics
    """
    # Calculate total consumption
    total_consumption = data['energy_consumption'].sum()

    # Calculate average efficiency
    avg_efficiency = data['efficiency_score'].mean()

    # Adjust optimization target based on efficiency
    # Lower efficiency means higher potential for savings
    efficiency_factor = (100 - avg_efficiency) / 100 * 1.5
    adjusted_target = optimization_target * max(efficiency_factor, 0.5)

    # Calculate potential savings
    potential_savings = total_consumption * (adjusted_target / 100)

    # Calculate cost savings (assuming $0.12 per kWh)
    cost_savings = potential_savings * 0.12

    # Calculate carbon reduction (assuming 0.85 kg CO2 per kWh)
    carbon_reduction = potential_savings * 0.85

    return {
        'energy_savings': potential_savings,
        'cost_savings': cost_savings,
        'carbon_reduction': carbon_reduction,
        'savings_percentage': adjusted_target
    }


def forecast_consumption(data, forecast_days=30, with_optimization=False, optimization_target=15):
    """
    Forecast future energy consumption

    Parameters:
    - data: DataFrame with historical network energy data
    - forecast_days: Number of days to forecast
    - with_optimization: Whether to apply optimization
    - optimization_target: Target optimization percentage

    Returns:
    - DataFrame with forecasted energy consumption
    """
    # Calculate daily consumption
    daily_consumption = data.groupby(
        'date')['energy_consumption'].sum().reset_index()
    daily_consumption['date'] = pd.to_datetime(daily_consumption['date'])

    # Sort by date
    daily_consumption = daily_consumption.sort_values('date')

    # If we have less than 14 data points, we can't do a good forecast
    if len(daily_consumption) < 14:
        # Just use average with random noise for demonstration
        avg_consumption = daily_consumption['energy_consumption'].mean()
        last_date = daily_consumption['date'].max()

        forecast_dates = [last_date +
                          timedelta(days=i+1) for i in range(forecast_days)]

        # Generate forecast with some trending and weekly patterns
        forecast = []
        for i, date in enumerate(forecast_dates):
            # Add trend factor (slight increase over time)
            trend_factor = 1 + (i * 0.002)

            # Add weekly pattern (weekends about 20% lower)
            day_factor = 0.8 if date.weekday() >= 5 else 1.0

            # Generate forecast with random noise
            forecast_value = avg_consumption * trend_factor * \
                day_factor * np.random.normal(1.0, 0.05)
            forecast.append(forecast_value)
    else:
        # Use a simple model for forecasting
        # Convert dates to numeric feature (days since first date)
        first_date = daily_consumption['date'].min()
        daily_consumption['day_number'] = (
            daily_consumption['date'] - first_date).dt.days

        # Add day of week as feature (one-hot encoded)
        for i in range(7):
            daily_consumption[f'day_{i}'] = (
                daily_consumption['date'].dt.dayofweek == i).astype(int)

        # Prepare features and target for model
        X = daily_consumption[['day_number'] + [f'day_{i}' for i in range(7)]]
        y = daily_consumption['energy_consumption']

        # Train a random forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Generate forecast dates
        last_date = daily_consumption['date'].max()
        last_day_number = daily_consumption['day_number'].max()

        forecast_dates = [last_date +
                          timedelta(days=i+1) for i in range(forecast_days)]
        forecast_features = []

        for date in forecast_dates:
            day_number = last_day_number + (date - last_date).days

            # One-hot encode day of week
            day_features = [1 if date.weekday() == i else 0 for i in range(7)]

            forecast_features.append([day_number] + day_features)

        # Make predictions
        forecast = model.predict(forecast_features)

    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'original_forecast': forecast
    })

    # If optimization is requested, calculate optimized forecast
    if with_optimization:
        # Start with small savings, gradually increasing to target
        savings_rate = np.linspace(
            0.01, optimization_target/100, forecast_days)
        forecast_df['optimized_forecast'] = forecast_df['original_forecast'] * \
            (1 - savings_rate)
    else:
        forecast_df['optimized_forecast'] = forecast_df['original_forecast']

    return forecast_df


def detect_anomalies(data, contamination=0.02):
    """
    Detect anomalies in energy consumption data

    Parameters:
    - data: DataFrame with network energy data
    - contamination: Expected proportion of anomalies

    Returns:
    - DataFrame with anomaly flags
    """
    # Copy data to avoid modifying original
    df = data.copy()

    # Group by component type and component ID
    groups = df.groupby(['component_type', 'component_id'])

    # Initialize anomaly column
    df['is_anomaly'] = 0

    # Process each component separately
    for name, group in groups:
        if len(group) < 10:  # Skip if not enough data
            continue

        # Extract features
        X = group[['energy_consumption', 'utilization', 'temperature']].values

        # Train isolation forest
        model = IsolationForest(contamination=contamination, random_state=42)
        predictions = model.fit_predict(X)

        # Mark anomalies (isolation forest returns -1 for anomalies)
        is_anomaly = predictions == -1

        # Update anomaly flags in the original dataframe
        df.loc[group.index[is_anomaly], 'is_anomaly'] = 1

    return df
