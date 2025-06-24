#!/usr/bin/env python3
"""
Example usage script for the Facebook Prophet forecasting project.
This script demonstrates various ways to use the forecasting model.
"""

import pandas as pd
import numpy as np
from forecasting_model import ForecastingModel
from visualization import ForecastingVisualizer
from data_generator import generate_sample_data, generate_sales_data

def example_1_basic_forecasting():
    """Example 1: Basic forecasting with sample data."""
    print("=" * 50)
    print("EXAMPLE 1: Basic Forecasting")
    print("=" * 50)
    
    # Generate sample data
    data = generate_sample_data(periods=200, trend=0.2)
    
    # Initialize model
    forecaster = ForecastingModel(df=data)
    
    # Create and fit model
    forecaster.create_model(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    forecaster.fit_model()
    
    # Make forecast
    forecast = forecaster.make_forecast(periods=30)
    
    # Get summary
    summary = forecaster.get_forecast_summary()
    print(f"\nForecast Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Evaluate model
    evaluation = forecaster.evaluate_model(test_periods=20)
    
    # Create visualizations
    visualizer = ForecastingVisualizer(data, forecast)
    visualizer.plot_interactive_forecast("Basic Forecasting Example")
    visualizer.plot_components_interactive()
    
    return forecaster, visualizer, evaluation

def example_2_sales_forecasting():
    """Example 2: Sales forecasting with realistic data."""
    print("\n" + "=" * 50)
    print("EXAMPLE 2: Sales Forecasting")
    print("=" * 50)
    
    # Generate sales data
    data = generate_sales_data(periods=365)
    
    # Initialize model
    forecaster = ForecastingModel(df=data)
    
    # Prepare data (use last 300 days)
    forecaster.prepare_data(min_date='2020-03-01')
    
    # Create model with different parameters
    forecaster.create_model(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative',  # Better for sales data
        changepoint_prior_scale=0.1,  # More flexible trend
        seasonality_prior_scale=15.0  # More flexible seasonality
    )
    forecaster.fit_model()
    
    # Make forecast for next 60 days
    forecast = forecaster.make_forecast(periods=60)
    
    # Get recent forecast
    recent = forecaster.get_recent_forecast(days=10)
    print(f"\nNext 10 days sales forecast:")
    print(recent.to_string(index=False))
    
    # Evaluate model
    evaluation = forecaster.evaluate_model(test_periods=30)
    
    # Create comprehensive visualizations
    visualizer = ForecastingVisualizer(data, forecast)
    visualizer.plot_interactive_forecast("Sales Forecasting")
    visualizer.plot_forecast_metrics(evaluation)
    visualizer.plot_seasonality_analysis()
    
    # Save results
    forecaster.save_forecast('sales_forecast.csv')
    visualizer.create_forecast_report(evaluation, 'sales_forecast_report.html')
    
    return forecaster, visualizer, evaluation

def example_3_custom_data():
    """Example 3: Using custom data from CSV file."""
    print("\n" + "=" * 50)
    print("EXAMPLE 3: Custom Data from CSV")
    print("=" * 50)
    
    # This example shows how to use your own data
    print("To use your own data, create a CSV file with columns 'ds' and 'y':")
    print("ds,y")
    print("2020-01-01,100")
    print("2020-01-02,105")
    print("...")
    
    # Example of how to load custom data
    try:
        # Try to load existing data
        forecaster = ForecastingModel(data_path='sales_data.csv')
        print("Loaded existing sales data")
    except FileNotFoundError:
        print("No custom data found. Using generated data instead.")
        data = generate_sales_data()
        forecaster = ForecastingModel(df=data)
    
    # Configure model for your specific use case
    forecaster.create_model(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    forecaster.fit_model()
    
    # Make forecast
    forecast = forecaster.make_forecast(periods=30)
    
    # Show results
    summary = forecaster.get_forecast_summary()
    print(f"\nForecast Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    return forecaster

def example_4_model_comparison():
    """Example 4: Compare different model configurations."""
    print("\n" + "=" * 50)
    print("EXAMPLE 4: Model Comparison")
    print("=" * 50)
    
    # Generate data
    data = generate_sales_data(periods=200)
    
    # Define different model configurations
    configs = {
        'Basic': {
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'seasonality_mode': 'additive',
            'changepoint_prior_scale': 0.05
        },
        'Flexible': {
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'seasonality_mode': 'multiplicative',
            'changepoint_prior_scale': 0.1,
            'seasonality_prior_scale': 15.0
        },
        'Conservative': {
            'yearly_seasonality': True,
            'weekly_seasonality': False,
            'daily_seasonality': False,
            'seasonality_mode': 'additive',
            'changepoint_prior_scale': 0.01
        }
    }
    
    results = {}
    
    for config_name, config_params in configs.items():
        print(f"\nTesting {config_name} configuration...")
        
        # Create model
        forecaster = ForecastingModel(df=data.copy())
        forecaster.create_model(**config_params)
        forecaster.fit_model()
        
        # Evaluate
        evaluation = forecaster.evaluate_model(test_periods=20)
        results[config_name] = evaluation
        
        print(f"  MAE: {evaluation['mae']:.2f}")
        print(f"  MAPE: {evaluation['mape']:.2f}%")
        print(f"  RMSE: {evaluation['rmse']:.2f}")
    
    # Compare results
    print(f"\nModel Comparison Summary:")
    print(f"{'Model':<12} {'MAE':<8} {'MAPE':<8} {'RMSE':<8}")
    print("-" * 40)
    for config_name, evaluation in results.items():
        print(f"{config_name:<12} {evaluation['mae']:<8.2f} {evaluation['mape']:<8.2f} {evaluation['rmse']:<8.2f}")
    
    return results

def main():
    """Run all examples."""
    print("Facebook Prophet Forecasting Examples")
    print("=" * 60)
    
    try:
        # Run examples
        print("\nRunning example 1...")
        forecaster1, visualizer1, eval1 = example_1_basic_forecasting()
        
        print("\nRunning example 2...")
        forecaster2, visualizer2, eval2 = example_2_sales_forecasting()
        
        print("\nRunning example 3...")
        forecaster3 = example_3_custom_data()
        
        print("\nRunning example 4...")
        results = example_4_model_comparison()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("Check the generated plots and CSV files for results.")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main() 