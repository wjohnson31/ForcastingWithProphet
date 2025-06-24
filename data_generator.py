import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_data(start_date='2020-01-01', periods=365, trend=0.1, seasonality=True):
    """
    Generate sample time series data for forecasting demonstration.
    
    Args:
        start_date (str): Start date for the time series
        periods (int): Number of periods to generate
        trend (float): Trend coefficient
        seasonality (bool): Whether to add seasonality
    
    Returns:
        pd.DataFrame: DataFrame with 'ds' (date) and 'y' (value) columns
    """
    # Create date range
    dates = pd.date_range(start=start_date, periods=periods, freq='D')
    
    # Generate base values with trend
    base_values = np.arange(periods) * trend
    
    # Add some noise
    noise = np.random.normal(0, 5, periods)
    
    # Add seasonality if requested
    if seasonality:
        # Weekly seasonality
        weekly_seasonality = 10 * np.sin(2 * np.pi * np.arange(periods) / 7)
        # Monthly seasonality
        monthly_seasonality = 15 * np.sin(2 * np.pi * np.arange(periods) / 30)
        # Yearly seasonality
        yearly_seasonality = 20 * np.sin(2 * np.pi * np.arange(periods) / 365)
        
        values = base_values + weekly_seasonality + monthly_seasonality + yearly_seasonality + noise
    else:
        values = base_values + noise
    
    # Ensure all values are positive
    values = np.maximum(values, 0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'ds': dates,
        'y': values
    })
    
    return df

def generate_sales_data(start_date='2020-01-01', periods=365):
    """
    Generate realistic sales data with trend, seasonality, and holidays.
    
    Args:
        start_date (str): Start date for the time series
        periods (int): Number of periods to generate
    
    Returns:
        pd.DataFrame: DataFrame with 'ds' (date) and 'y' (sales) columns
    """
    dates = pd.date_range(start=start_date, periods=periods, freq='D')
    
    # Base sales with upward trend
    base_sales = 100 + np.arange(periods) * 0.5
    
    # Weekly seasonality (higher on weekends)
    day_of_week = np.array([d.weekday() for d in dates])
    weekly_pattern = np.where(day_of_week >= 5, 1.3, 1.0)  # 30% higher on weekends
    
    # Monthly seasonality (higher in December, lower in January)
    month = np.array([d.month for d in dates])
    monthly_pattern = np.where(month == 12, 1.5, 1.0)  # 50% higher in December
    monthly_pattern = np.where(month == 1, 0.8, monthly_pattern)  # 20% lower in January
    
    # Add some random noise
    noise = np.random.normal(0, 10, periods)
    
    # Calculate final sales
    sales = base_sales * weekly_pattern * monthly_pattern + noise
    sales = np.maximum(sales, 0)  # Ensure non-negative sales
    
    df = pd.DataFrame({
        'ds': dates,
        'y': sales
    })
    
    return df

if __name__ == "__main__":
    # Generate sample data
    print("Generating sample forecasting data...")
    
    # Generate basic time series data
    basic_data = generate_sample_data()
    basic_data.to_csv('sample_data.csv', index=False)
    print(f"Generated basic sample data: {len(basic_data)} records")
    
    # Generate sales data
    sales_data = generate_sales_data()
    sales_data.to_csv('sales_data.csv', index=False)
    print(f"Generated sales data: {len(sales_data)} records")
    
    print("Data generation complete!") 