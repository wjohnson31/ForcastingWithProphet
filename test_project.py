#!/usr/bin/env python3
"""
Simple test script to verify the forecasting project setup.
"""

import sys
import pandas as pd

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ Pandas imported successfully")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ Matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    try:
        import seaborn as sns
        print("✓ Seaborn imported successfully")
    except ImportError as e:
        print(f"✗ Seaborn import failed: {e}")
        return False
    
    try:
        import plotly.graph_objects as go
        print("✓ Plotly imported successfully")
    except ImportError as e:
        print(f"✗ Plotly import failed: {e}")
        return False
    
    try:
        from prophet import Prophet
        print("✓ Facebook Prophet imported successfully")
    except ImportError as e:
        print(f"✗ Facebook Prophet import failed: {e}")
        print("  Try: pip install prophet")
        return False
    
    return True

def test_data_generation():
    """Test data generation functionality."""
    print("\nTesting data generation...")
    
    try:
        from data_generator import generate_sample_data, generate_sales_data
        
        # Test sample data generation
        sample_data = generate_sample_data(periods=50)
        print(f"✓ Sample data generated: {len(sample_data)} records")
        
        # Test sales data generation
        sales_data = generate_sales_data(periods=50)
        print(f"✓ Sales data generated: {len(sales_data)} records")
        
        # Check data format
        assert 'ds' in sample_data.columns, "Missing 'ds' column"
        assert 'y' in sample_data.columns, "Missing 'y' column"
        print("✓ Data format is correct")
        
        return True
        
    except Exception as e:
        print(f"✗ Data generation failed: {e}")
        return False

def test_forecasting_model():
    """Test the forecasting model functionality."""
    print("\nTesting forecasting model...")
    
    try:
        from forecasting_model import ForecastingModel
        from data_generator import generate_sample_data
        
        # Generate small dataset for testing
        data = generate_sample_data(periods=30)
        
        # Create model
        forecaster = ForecastingModel(df=data)
        print("✓ ForecastingModel created successfully")
        
        # Create and fit model
        forecaster.create_model(
            yearly_seasonality=False,  # Disable for small dataset
            weekly_seasonality=False,  # Disable for small dataset
            daily_seasonality=False
        )
        forecaster.fit_model()
        print("✓ Model fitted successfully")
        
        # Make forecast
        forecast = forecaster.make_forecast(periods=5)
        print(f"✓ Forecast generated: {len(forecast)} records")
        
        # Get summary
        summary = forecaster.get_forecast_summary()
        print("✓ Forecast summary generated")
        
        return True
        
    except Exception as e:
        print(f"✗ Forecasting model failed: {e}")
        return False

def test_visualization():
    """Test visualization functionality."""
    print("\nTesting visualization...")
    
    try:
        from visualization import ForecastingVisualizer
        from data_generator import generate_sample_data
        from forecasting_model import ForecastingModel
        
        # Generate data and forecast
        data = generate_sample_data(periods=30)
        forecaster = ForecastingModel(df=data)
        forecaster.create_model(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        forecaster.fit_model()
        forecast = forecaster.make_forecast(periods=5)
        
        # Create visualizer
        visualizer = ForecastingVisualizer(data, forecast)
        print("✓ ForecastingVisualizer created successfully")
        
        # Test getting recent forecast
        recent = forecaster.get_recent_forecast(days=3)
        print(f"✓ Recent forecast retrieved: {len(recent)} records")
        
        return True
        
    except Exception as e:
        print(f"✗ Visualization failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("Facebook Prophet Forecasting Project - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Data Generation", test_data_generation),
        ("Forecasting Model", test_forecasting_model),
        ("Visualization", test_visualization)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
            print(f"✓ {test_name} PASSED")
        else:
            print(f"✗ {test_name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your forecasting project is ready to use.")
        print("\nNext steps:")
        print("1. Run: python forecasting_model.py")
        print("2. Run: python example_usage.py")
        print("3. Check the README.md for detailed usage instructions")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("\nTroubleshooting:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. For Prophet issues: pip install prophet --no-cache-dir")
        print("3. Check Python version (requires 3.7+)")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 