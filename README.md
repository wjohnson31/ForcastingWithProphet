# 📊 Facebook Prophet Forecasting Project

A comprehensive machine learning project for time series forecasting using Facebook Prophet. This project provides a complete solution for generating forecasts with advanced visualization capabilities and model evaluation.

## 🚀 Features

- **Easy-to-use Forecasting Model**: Simple API for creating and training Prophet models
- **Data Generation**: Built-in tools to generate realistic sample data for testing
- **Advanced Visualizations**: Interactive plots using Plotly and comprehensive matplotlib visualizations
- **Model Evaluation**: Built-in metrics (MAE, MAPE, RMSE) for model performance assessment
- **Multiple Use Cases**: Examples for sales forecasting, basic time series, and custom data
- **Export Capabilities**: Save forecasts to CSV and generate HTML reports
- **Seasonality Analysis**: Automatic detection and visualization of seasonal patterns

## 📋 Requirements

- Python 3.7+
- Facebook Prophet
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn

## 🛠️ Installation

1. **Clone or download this project**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import prophet; print('Prophet installed successfully!')"
   ```

## 📖 Quick Start

### Basic Usage

```python
from forecasting_model import ForecastingModel
from data_generator import generate_sales_data

# Generate sample data
data = generate_sales_data()

# Create and train model
forecaster = ForecastingModel(df=data)
forecaster.create_model()
forecaster.fit_model()

# Make forecast
forecast = forecaster.make_forecast(periods=30)

# Get results
summary = forecaster.get_forecast_summary()
print(summary)
```

### Run the Complete Demo

```bash
python forecasting_model.py
```

This will:
- Generate sample sales data
- Train a Prophet model
- Make a 30-day forecast
- Display results and create visualizations

## 📁 Project Structure

```
├── requirements.txt          # Python dependencies
├── data_generator.py         # Sample data generation
├── forecasting_model.py      # Main forecasting class
├── visualization.py          # Advanced plotting tools
├── example_usage.py          # Usage examples
├── README.md                 # This file
├── sales_data.csv           # Generated sample data
├── forecast_results.csv      # Forecast output
└── sales_forecast_report.html # HTML report
```

## 🔧 Usage Examples

### Example 1: Basic Forecasting

```python
from forecasting_model import ForecastingModel
from data_generator import generate_sample_data

# Generate data
data = generate_sample_data(periods=200, trend=0.2)

# Create model
forecaster = ForecastingModel(df=data)
forecaster.create_model(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)
forecaster.fit_model()

# Make forecast
forecast = forecaster.make_forecast(periods=30)

# Evaluate model
evaluation = forecaster.evaluate_model(test_periods=20)
print(f"MAE: {evaluation['mae']:.2f}")
```

### Example 2: Sales Forecasting

```python
from forecasting_model import ForecastingModel
from data_generator import generate_sales_data

# Generate sales data
data = generate_sales_data(periods=365)

# Create model optimized for sales
forecaster = ForecastingModel(df=data)
forecaster.create_model(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='multiplicative',  # Better for sales
    changepoint_prior_scale=0.1
)
forecaster.fit_model()

# Make forecast
forecast = forecaster.make_forecast(periods=60)

# Get recent forecast
recent = forecaster.get_recent_forecast(days=10)
print(recent)
```

### Example 3: Using Your Own Data

```python
from forecasting_model import ForecastingModel

# Load your data (must have 'ds' and 'y' columns)
forecaster = ForecastingModel(data_path='your_data.csv')

# Or use a DataFrame
import pandas as pd
df = pd.read_csv('your_data.csv')
df['ds'] = pd.to_datetime(df['ds'])
forecaster = ForecastingModel(df=df)

# Train and forecast
forecaster.create_model()
forecaster.fit_model()
forecast = forecaster.make_forecast(periods=30)
```

## 📊 Visualization Features

### Interactive Plots

```python
from visualization import ForecastingVisualizer

# Create visualizer
visualizer = ForecastingVisualizer(data, forecast)

# Interactive forecast plot
visualizer.plot_interactive_forecast()

# Component analysis
visualizer.plot_components_interactive()

# Performance metrics dashboard
visualizer.plot_forecast_metrics(evaluation)

# Seasonality analysis
visualizer.plot_seasonality_analysis()
```

### Export Results

```python
# Save forecast to CSV
forecaster.save_forecast('my_forecast.csv')

# Generate HTML report
visualizer.create_forecast_report(evaluation, 'my_report.html')
```

## 🎯 Model Parameters

### Key Prophet Parameters

- **`yearly_seasonality`**: Include yearly patterns (default: True)
- **`weekly_seasonality`**: Include weekly patterns (default: True)
- **`daily_seasonality`**: Include daily patterns (default: False)
- **`seasonality_mode`**: 'additive' or 'multiplicative' (default: 'additive')
- **`changepoint_prior_scale`**: Trend flexibility (default: 0.05)
- **`seasonality_prior_scale`**: Seasonality flexibility (default: 10.0)

### Parameter Guidelines

- **Sales Data**: Use `seasonality_mode='multiplicative'` and higher `changepoint_prior_scale`
- **Stock Prices**: Use `seasonality_mode='additive'` and lower `changepoint_prior_scale`
- **Website Traffic**: Include `daily_seasonality=True`
- **Conservative Forecasts**: Use lower `changepoint_prior_scale` values

## 📈 Model Evaluation

The project includes comprehensive evaluation metrics:

- **MAE (Mean Absolute Error)**: Average absolute difference between predictions and actual values
- **MAPE (Mean Absolute Percentage Error)**: Average percentage error
- **RMSE (Root Mean Square Error)**: Square root of average squared errors

```python
evaluation = forecaster.evaluate_model(test_periods=30)
print(f"MAE: {evaluation['mae']:.2f}")
print(f"MAPE: {evaluation['mape']:.2f}%")
print(f"RMSE: {evaluation['rmse']:.2f}")
```

## 🚀 Running Examples

### Run All Examples

```bash
python example_usage.py
```

This will run:
1. Basic forecasting example
2. Sales forecasting with advanced parameters
3. Custom data usage example
4. Model comparison across different configurations

### Individual Examples

```bash
# Basic forecasting
python forecasting_model.py

# Generate sample data only
python data_generator.py

# Run specific example
python -c "from example_usage import example_1_basic_forecasting; example_1_basic_forecasting()"
```

## 📝 Data Format

Your data should be in CSV format with two columns:

```csv
ds,y
2020-01-01,100
2020-01-02,105
2020-01-03,98
...
```

- **`ds`**: Date column (will be converted to datetime)
- **`y`**: Target variable to forecast

## 🔍 Troubleshooting

### Common Issues

1. **Prophet Installation Issues**:
   ```bash
   # On macOS/Linux
   conda install -c conda-forge prophet
   
   # Or with pip
   pip install prophet --no-cache-dir
   ```

2. **Plotly Display Issues**:
   ```python
   import plotly.io as pio
   pio.renderers.default = "browser"  # Opens in browser
   ```

3. **Memory Issues with Large Datasets**:
   ```python
   # Use data filtering
   forecaster.prepare_data(min_date='2020-01-01', max_date='2021-12-31')
   ```

### Performance Tips

- Use `daily_seasonality=False` for datasets with > 1000 observations
- Adjust `changepoint_prior_scale` based on expected trend changes
- Use `seasonality_mode='multiplicative'` for data with increasing variance
- Plotly for interactive visualizations
- The open-source community for various supporting libraries

---

**Happy Forecasting! 🚀** 
