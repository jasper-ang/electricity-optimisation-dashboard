# ⚡ AI-Powered Network Energy Optimization Dashboard

A Streamlit dashboard application for monitoring and optimizing electricity consumption across network infrastructure components. This tool leverages AI-based forecasting and anomaly detection to identify energy optimization opportunities.

## Features

- **Energy Consumption Analysis**: Visualize consumption patterns and trends over time
- **Component-level Heatmap**: Identify high-consumption components across facilities
- **AI-Powered Optimization**: Get recommendations for optimizing energy usage
- **Projected Savings**: Calculate potential energy, cost and carbon savings
- **Consumption Forecasting**: Predict future energy usage with and without optimizations
- **Anomaly Detection**: Identify unusual energy consumption patterns

## How to Run

1. Install the required dependencies

   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit app

   ```bash
   streamlit run streamlit_app.py
   ```

3. Open your browser and navigate to the URL provided by Streamlit (typically http://localhost:8501)

## Data

The application uses either:
- Sample network data if available in the `/data` directory
- Auto-generated synthetic data that simulates network component energy consumption

## Project Structure

- `streamlit_app.py` - Main application file
- `utils/`
  - `visualization.py` - Visualization functions using Plotly
  - `ai_models.py` - AI forecasting and optimization models
  - `data_loader.py` - Data loading and preprocessing functions
- `requirements.txt` - Required Python packages

## Dependencies

- Streamlit
- Pandas
- NumPy
- Plotly
- scikit-learn
- Matplotlib
- SciPy

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
