import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ForecastingModel:
    def __init__(self, data_path=None, df=None):
        """
        Initialize the forecasting model.
        
        Args:
            data_path (str): Path to CSV file with data
            df (pd.DataFrame): DataFrame with 'ds' and 'y' columns
        """
        self.model = None
        self.forecast = None
        self.data = None
        
        if data_path:
            self.load_data(data_path)
        elif df is not None:
            self.data = df
        else:
            raise ValueError("Either data_path or df must be provided")
    
    def load_data(self, data_path):
        """Load data from CSV file."""
        self.data = pd.read_csv(data_path)
        self.data['ds'] = pd.to_datetime(self.data['ds'])
        print(f"Loaded data: {len(self.data)} records from {self.data['ds'].min()} to {self.data['ds'].max()}")
    
    def prepare_data(self, min_date=None, max_date=None):
        """
        Prepare data for modeling by filtering date range if needed.
        
        Args:
            min_date (str): Minimum date to include
            max_date (str): Maximum date to include
        """
        if min_date:
            min_date_dt = pd.to_datetime(min_date)
            self.data = self.data[self.data['ds'] >= min_date_dt]
        if max_date:
            max_date_dt = pd.to_datetime(max_date)
            self.data = self.data[self.data['ds'] <= max_date_dt]
        
        print(f"Prepared data: {len(self.data)} records")
        return self.data
    
    def create_model(self, 
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    seasonality_mode='additive',
                    changepoint_prior_scale=0.05,
                    seasonality_prior_scale=10.0):
        """
        Create and configure the Prophet model.
        
        Args:
            yearly_seasonality (bool): Whether to include yearly seasonality
            weekly_seasonality (bool): Whether to include weekly seasonality
            daily_seasonality (bool): Whether to include daily seasonality
            seasonality_mode (str): 'additive' or 'multiplicative'
            changepoint_prior_scale (float): Flexibility of the trend
            seasonality_prior_scale (float): Flexibility of the seasonality
        """
        self.model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale
        )
        
        print("Prophet model created with the following parameters:")
        print(f"- Yearly seasonality: {yearly_seasonality}")
        print(f"- Weekly seasonality: {weekly_seasonality}")
        print(f"- Daily seasonality: {daily_seasonality}")
        print(f"- Seasonality mode: {seasonality_mode}")
        print(f"- Changepoint prior scale: {changepoint_prior_scale}")
        print(f"- Seasonality prior scale: {seasonality_prior_scale}")
    
    def fit_model(self):
        """Fit the Prophet model to the data."""
        if self.model is None:
            self.create_model()
        
        print("Fitting Prophet model...")
        self.model.fit(self.data)
        print("Model fitting complete!")
    
    def make_forecast(self, periods=30, freq='D'):
        """
        Make a forecast for future periods.
        
        Args:
            periods (int): Number of periods to forecast
            freq (str): Frequency of the forecast ('D' for daily, 'W' for weekly, etc.)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making forecasts")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        
        # Make forecast
        self.forecast = self.model.predict(future)
        
        print(f"Forecast generated for {periods} periods with frequency '{freq}'")
        return self.forecast
    
    def plot_forecast(self, figsize=(15, 10)):
        """Plot the forecast results."""
        if self.forecast is None:
            raise ValueError("No forecast available. Run make_forecast() first.")
        
        fig = self.model.plot(self.forecast, figsize=figsize)
        plt.title('Prophet Forecast', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    def plot_components(self, figsize=(15, 12)):
        """Plot the forecast components (trend, seasonality)."""
        if self.forecast is None:
            raise ValueError("No forecast available. Run make_forecast() first.")
        
        fig = self.model.plot_components(self.forecast, figsize=figsize)
        plt.tight_layout()
        plt.show()
    
    def get_forecast_summary(self):
        """Get a summary of the forecast results."""
        if self.forecast is None:
            raise ValueError("No forecast available. Run make_forecast() first.")
        
        # Get the forecasted values (excluding historical data)
        historical_end = self.data['ds'].max()
        future_forecast = self.forecast[self.forecast['ds'] > historical_end]
        
        summary = {
            'forecast_start': future_forecast['ds'].min(),
            'forecast_end': future_forecast['ds'].max(),
            'forecast_periods': len(future_forecast),
            'mean_forecast': future_forecast['yhat'].mean(),
            'min_forecast': future_forecast['yhat'].min(),
            'max_forecast': future_forecast['yhat'].max(),
            'forecast_std': future_forecast['yhat'].std()
        }
        
        return summary
    
    def evaluate_model(self, test_periods=30):
        """
        Evaluate the model performance using historical data.
        
        Args:
            test_periods (int): Number of periods to use for testing
        """
        if len(self.data) < test_periods:
            raise ValueError(f"Not enough data for {test_periods} test periods")
        
        # Split data into train and test
        train_data = self.data.iloc[:-test_periods].copy()
        test_data = self.data.iloc[-test_periods:].copy()
        
        # Create and fit model on training data
        train_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        train_model.fit(train_data)
        
        # Make forecast for test period
        future = train_model.make_future_dataframe(periods=test_periods)
        forecast = train_model.predict(future)
        
        # Get only the forecasted values for test period
        test_forecast = forecast[forecast['ds'] >= test_data['ds'].min()]
        
        # Calculate metrics
        mae = np.mean(np.abs(test_forecast['yhat'] - test_data['y']))
        mape = np.mean(np.abs((test_forecast['yhat'] - test_data['y']) / test_data['y'])) * 100
        rmse = np.sqrt(np.mean((test_forecast['yhat'] - test_data['y']) ** 2))
        
        evaluation = {
            'mae': mae,
            'mape': mape,
            'rmse': rmse,
            'test_periods': test_periods
        }
        
        print("Model Evaluation Results:")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print(f"Root Mean Square Error (RMSE): {rmse:.2f}")
        
        return evaluation
    
    def save_forecast(self, filename='forecast_results.csv'):
        """Save forecast results to CSV file."""
        if self.forecast is None:
            raise ValueError("No forecast available. Run make_forecast() first.")
        
        # Select relevant columns
        forecast_export = self.forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        forecast_export.columns = ['date', 'forecast', 'lower_bound', 'upper_bound']
        
        forecast_export.to_csv(filename, index=False)
        print(f"Forecast saved to {filename}")
    
    def get_recent_forecast(self, days=7):
        """Get the most recent forecast values."""
        if self.forecast is None:
            raise ValueError("No forecast available. Run make_forecast() first.")
        
        # Get the forecasted values (excluding historical data)
        historical_end = self.data['ds'].max()
        future_forecast = self.forecast[self.forecast['ds'] > historical_end]
        
        # Get the most recent days
        recent_forecast = future_forecast.head(days)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        recent_forecast.columns = ['date', 'forecast', 'lower_bound', 'upper_bound']
        
        return recent_forecast

def main():
    """Main function to demonstrate the forecasting model."""
    print("=== Facebook Prophet Forecasting Demo ===\n")
    
    # Generate sample data if it doesn't exist
    try:
        data = pd.read_csv('sales_data.csv', parse_dates=['ds'])
        print("Using existing sales data...")
    except FileNotFoundError:
        print("Generating sample sales data...")
        from data_generator import generate_sales_data
        data = generate_sales_data()
        data.to_csv('sales_data.csv', index=False)
    
    # Initialize the forecasting model
    forecaster = ForecastingModel(df=data)
    
    # Prepare data (use last 300 days for training)
    forecaster.prepare_data(min_date='2020-03-01')
    
    # Create and fit the model
    forecaster.create_model(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='additive'
    )
    forecaster.fit_model()
    
    # Make forecast for next 30 days
    forecast = forecaster.make_forecast(periods=30, freq='D')
    
    # Display forecast summary
    summary = forecaster.get_forecast_summary()
    print("\nForecast Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Show recent forecast
    print("\nRecent Forecast (next 7 days):")
    recent = forecaster.get_recent_forecast(days=7)
    print(recent.to_string(index=False))
    
    # Evaluate model performance
    print("\nModel Evaluation:")
    evaluation = forecaster.evaluate_model(test_periods=30)
    
    # Save results
    forecaster.save_forecast('forecast_results.csv')
    
    # Plot results
    print("\nGenerating plots...")
    forecaster.plot_forecast()
    forecaster.plot_components()
    
    print("\nForecasting complete! Check the generated plots and CSV files.")

if __name__ == "__main__":
    main() 