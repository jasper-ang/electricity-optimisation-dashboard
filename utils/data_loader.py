
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def load_network_data():
    """
    Load network energy consumption data from CSV or generate synthetic data if file doesn't exist
    """
    data_path = os.path.join(os.path.dirname(
        os.path.dirname(__file__)), 'data', 'sample_network_data.csv')

    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    else:
        # Generate synthetic data if file doesn't exist
        return generate_synthetic_data()


def generate_synthetic_data():
    """
    Generate synthetic network energy consumption data for demonstration
    """
    # Set seed for reproducibility
    np.random.seed(42)

    # Create date range for the past 60 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    dates = [start_date + timedelta(days=i) for i in range(61)]

    # Create facilities and components
    facilities = ["Data Center A", "Data Center B",
                  "Edge Location C", "Edge Location D"]
    components = {
        "Router": ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"],
        "Switch": ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"],
        "Server": ["SRV1", "SRV2", "SRV3", "SRV4", "SRV5", "SRV6", "SRV7", "SRV8", "SRV9", "SRV10",
                   "SRV11", "SRV12", "SRV13", "SRV14", "SRV15", "SRV16"],
        "Storage": ["ST1", "ST2", "ST3", "ST4", "ST5", "ST6"],
        "Cooling": ["C1", "C2", "C3", "C4"]
    }

    # Base consumption values for different component types
    base_consumption = {
        "Router": 85,
        "Switch": 45,
        "Server": 120,
        "Storage": 70,
        "Cooling": 180
    }

    # Base efficiency scores for different component types
    base_efficiency = {
        "Router": 78,
        "Switch": 82,
        "Server": 75,
        "Storage": 80,
        "Cooling": 65
    }

    # Create empty list to store records
    records = []

    # Generate data for each day, facility, and component
    for date in dates:
        # Add daily variation (weekdays vs weekend, seasonal effects)
        day_factor = 0.8 if date.weekday() >= 5 else 1.0  # Weekend vs weekday
        seasonal_factor = 1.0 + 0.2 * \
            np.sin(2 * np.pi * (date.timetuple().tm_yday / 365)
                   )  # Seasonal variation

        for facility in facilities:
            # Different facilities have different scales
            facility_factor = 1.0
            if facility == "Data Center A":
                facility_factor = 1.2
            elif facility == "Data Center B":
                facility_factor = 1.1
            elif facility == "Edge Location C":
                facility_factor = 0.8
            else:  # Edge Location D
                facility_factor = 0.7

            for component_type, component_list in components.items():
                for component_id in component_list:
                    # Base energy consumption with variations
                    base = base_consumption[component_type]

                    # Add random variation and factors
                    random_factor = np.random.normal(1.0, 0.15)  # Random noise

                    # Calculate energy consumption
                    energy_consumption = base * day_factor * \
                        seasonal_factor * facility_factor * random_factor

                    # Add some anomalies (unusually high values) for demonstration
                    if np.random.random() < 0.02:  # 2% chance of anomaly
                        energy_consumption *= np.random.uniform(1.5, 2.5)

                    # Calculate efficiency score
                    base_eff = base_efficiency[component_type]
                    efficiency_variation = np.random.normal(
                        0, 5)  # Random variation in efficiency
                    # Clamp between 40 and 100
                    efficiency_score = max(
                        min(base_eff + efficiency_variation, 100), 40)

                    # Create record
                    record = {
                        "date": date.strftime("%Y-%m-%d"),
                        "facility": facility,
                        "component_type": component_type,
                        "component_id": component_id,
                        "energy_consumption": round(energy_consumption, 2),
                        "efficiency_score": round(efficiency_score, 1),
                        "utilization": round(np.random.uniform(10, 95), 1),
                        "temperature": round(np.random.normal(24, 3), 1)
                    }
                    records.append(record)

    # Create DataFrame
    df = pd.DataFrame(records)

    # Save to CSV for future use
    os.makedirs(os.path.dirname(os.path.dirname(
        __file__)) + '/data', exist_ok=True)
    df.to_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)),
              'data', 'sample_network_data.csv'), index=False)

    return df


def preprocess_data(data, date_range=None, facility=None, components=None):
    """
    Preprocess and filter the data based on user selections

    Parameters:
    - data: The raw data DataFrame
    - date_range: Optional tuple of (start_date, end_date)
    - facility: Optional facility name to filter by
    - components: Optional list of component types to include

    Returns:
    - Filtered and processed DataFrame
    """
    # Create a copy to avoid modifying the original
    df = data.copy()

    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Filter by date range if provided
    if date_range and len(date_range) == 2:
        start_date, end_date = pd.to_datetime(
            date_range[0]), pd.to_datetime(date_range[1])
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    # Filter by facility if provided
    if facility and facility != "All Facilities":
        df = df[df['facility'] == facility]

    # Filter by component types if provided
    if components and len(components) > 0:
        df = df[df['component_type'].isin(components)]

    return df
