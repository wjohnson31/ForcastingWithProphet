import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

class ForecastingVisualizer:
    def __init__(self, data, forecast):
        """
        Initialize the visualizer with data and forecast results.
        
        Args:
            data (pd.DataFrame): Historical data with 'ds' and 'y' columns
            forecast (pd.DataFrame): Prophet forecast results
        """
        self.data = data
        self.forecast = forecast
        self.historical_end = data['ds'].max()
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_interactive_forecast(self, title="Interactive Forecast"):
        """
        Create an interactive plotly forecast visualization.
        
        Args:
            title (str): Plot title
        """
        # Separate historical and future data
        historical = self.forecast[self.forecast['ds'] <= self.historical_end]
        future = self.forecast[self.forecast['ds'] > self.historical_end]
        
        # Create the plot
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=historical['ds'],
            y=historical['yhat'],
            mode='lines',
            name='Historical (Fitted)',
            line=dict(color='blue', width=2)
        ))
        
        # Add actual historical data
        fig.add_trace(go.Scatter(
            x=self.data['ds'],
            y=self.data['y'],
            mode='markers',
            name='Actual Data',
            marker=dict(color='black', size=4)
        ))
        
        # Add forecast
        fig.add_trace(go.Scatter(
            x=future['ds'],
            y=future['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Add confidence intervals
        fig.add_trace(go.Scatter(
            x=future['ds'],
            y=future['yhat_upper'],
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=future['ds'],
            y=future['yhat_lower'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.1)',
            name='Confidence Interval',
            line=dict(width=0)
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add vertical line for forecast start
        fig.add_vline(
            x=self.historical_end,
            line_dash="dash",
            line_color="gray",
            annotation_text="Forecast Start"
        )
        
        fig.show()
        return fig
    
    def plot_components_interactive(self):
        """Create interactive component plots."""
        # Get components
        components = ['trend', 'yearly', 'weekly']
        available_components = [comp for comp in components if comp in self.forecast.columns]
        
        # Create subplots
        fig = make_subplots(
            rows=len(available_components),
            cols=1,
            subplot_titles=[comp.title() for comp in available_components],
            vertical_spacing=0.1
        )
        
        for i, component in enumerate(available_components, 1):
            fig.add_trace(
                go.Scatter(
                    x=self.forecast['ds'],
                    y=self.forecast[component],
                    mode='lines',
                    name=component.title(),
                    line=dict(width=2)
                ),
                row=i, col=1
            )
        
        fig.update_layout(
            title="Forecast Components",
            height=300 * len(available_components),
            showlegend=False
        )
        
        fig.show()
        return fig
    
    def plot_forecast_metrics(self, evaluation_results):
        """
        Create a dashboard of forecast metrics.
        
        Args:
            evaluation_results (dict): Model evaluation results
        """
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Forecast Performance Metrics', fontsize=16, fontweight='bold')
        
        # 1. Error distribution
        historical = self.forecast[self.forecast['ds'] <= self.historical_end]
        errors = historical['yhat'] - self.data['y']
        
        axes[0, 0].hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.8)
        axes[0, 0].set_title('Error Distribution')
        axes[0, 0].set_xlabel('Prediction Error')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Actual vs Predicted scatter
        axes[0, 1].scatter(self.data['y'], historical['yhat'], alpha=0.6, color='green')
        axes[0, 1].plot([self.data['y'].min(), self.data['y'].max()], 
                       [self.data['y'].min(), self.data['y'].max()], 
                       'r--', alpha=0.8)
        axes[0, 1].set_title('Actual vs Predicted')
        axes[0, 1].set_xlabel('Actual Values')
        axes[0, 1].set_ylabel('Predicted Values')
        
        # 3. Metrics summary
        metrics_text = f"""
        MAE: {evaluation_results['mae']:.2f}
        MAPE: {evaluation_results['mape']:.2f}%
        RMSE: {evaluation_results['rmse']:.2f}
        Test Periods: {evaluation_results['test_periods']}
        """
        axes[1, 0].text(0.1, 0.5, metrics_text, transform=axes[1, 0].transAxes, 
                       fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        axes[1, 0].set_title('Model Performance Metrics')
        axes[1, 0].axis('off')
        
        # 4. Time series of errors
        axes[1, 1].plot(historical['ds'], errors, alpha=0.7, color='orange')
        axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.8)
        axes[1, 1].set_title('Prediction Errors Over Time')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Error')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_seasonality_analysis(self):
        """Analyze and plot seasonality patterns."""
        # Extract day of week and month
        self.forecast['day_of_week'] = self.forecast['ds'].dt.day_name()
        self.forecast['month'] = self.forecast['ds'].dt.month_name()
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Seasonality Analysis', fontsize=16, fontweight='bold')
        
        # 1. Weekly seasonality
        if 'weekly' in self.forecast.columns:
            weekly_data = self.forecast.groupby('day_of_week')['weekly'].mean()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekly_data = weekly_data.reindex(day_order)
            
            axes[0, 0].bar(weekly_data.index, weekly_data.values, color='lightcoral')
            axes[0, 0].set_title('Weekly Seasonality')
            axes[0, 0].set_ylabel('Seasonal Effect')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Monthly seasonality
        if 'yearly' in self.forecast.columns:
            monthly_data = self.forecast.groupby('month')['yearly'].mean()
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            monthly_data = monthly_data.reindex(month_order)
            
            axes[0, 1].bar(monthly_data.index, monthly_data.values, color='lightgreen')
            axes[0, 1].set_title('Monthly Seasonality')
            axes[0, 1].set_ylabel('Seasonal Effect')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Trend analysis
        if 'trend' in self.forecast.columns:
            axes[1, 0].plot(self.forecast['ds'], self.forecast['trend'], color='purple', linewidth=2)
            axes[1, 0].set_title('Trend Component')
            axes[1, 0].set_xlabel('Date')
            axes[1, 0].set_ylabel('Trend')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Forecast confidence intervals
        future = self.forecast[self.forecast['ds'] > self.historical_end]
        if len(future) > 0:
            axes[1, 1].fill_between(future['ds'], future['yhat_lower'], future['yhat_upper'], 
                                  alpha=0.3, color='lightblue', label='Confidence Interval')
            axes[1, 1].plot(future['ds'], future['yhat'], color='red', linewidth=2, label='Forecast')
            axes[1, 1].set_title('Forecast with Confidence Intervals')
            axes[1, 1].set_xlabel('Date')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].legend()
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def create_forecast_report(self, evaluation_results, save_path='forecast_report.html'):
        """
        Create a comprehensive HTML report of the forecast results.
        
        Args:
            evaluation_results (dict): Model evaluation results
            save_path (str): Path to save the HTML report
        """
        # Create interactive plots
        forecast_fig = self.plot_interactive_forecast()
        components_fig = self.plot_components_interactive()
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Forecast Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Forecasting Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>📈 Model Performance</h2>
                <div class="metric">
                    <strong>MAE:</strong> {evaluation_results['mae']:.2f}
                </div>
                <div class="metric">
                    <strong>MAPE:</strong> {evaluation_results['mape']:.2f}%
                </div>
                <div class="metric">
                    <strong>RMSE:</strong> {evaluation_results['rmse']:.2f}
                </div>
                <div class="metric">
                    <strong>Test Periods:</strong> {evaluation_results['test_periods']}
                </div>
            </div>
            
            <div class="section">
                <h2>📅 Forecast Summary</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Data Points</td>
                        <td>{len(self.data)}</td>
                    </tr>
                    <tr>
                        <td>Forecast Periods</td>
                        <td>{len(self.forecast[self.forecast['ds'] > self.historical_end])}</td>
                    </tr>
                    <tr>
                        <td>Date Range</td>
                        <td>{self.data['ds'].min().strftime('%Y-%m-%d')} to {self.data['ds'].max().strftime('%Y-%m-%d')}</td>
                    </tr>
                </table>
            </div>
        </body>
        </html>
        """
        
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        print(f"Forecast report saved to {save_path}")
        return save_path

def main():
    """Demo function for the visualizer."""
    # This would typically be called after running the forecasting model
    print("Visualization module loaded. Use this with your forecasting results!")
    print("Example usage:")
    print("visualizer = ForecastingVisualizer(data, forecast)")
    print("visualizer.plot_interactive_forecast()")

if __name__ == "__main__":
    main() 