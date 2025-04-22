#!/usr/bin/env python3
"""
Demand Forecasting Model

This module implements a time series forecasting model to predict
customer service ticket volumes at various time granularities.
The model helps optimize agent staffing and scheduling.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import joblib
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TicketDemandForecaster:
    """
    Time series forecasting model for customer service ticket volume.
    Uses Facebook's Prophet algorithm with custom seasonality and regressor features.
    """
    
    def __init__(
        self,
        interval='H',  # Forecast granularity: H=hourly, D=daily, W=weekly
        forecast_horizon=168,  # Default: forecast 1 week ahead (if hourly)
        include_holidays=True,
        seasonality_mode='multiplicative',
        model_path=None
    ):
        """
        Initialize the forecaster.
        
        Parameters:
        -----------
        interval : str
            Time interval for forecasting (H=hourly, D=daily, W=weekly)
        forecast_horizon : int
            Number of periods to forecast ahead
        include_holidays : bool
            Whether to include holiday effects in the model
        seasonality_mode : str
            'additive' or 'multiplicative' seasonality
        model_path : str
            Path to load a saved model from (if None, creates a new model)
        """
        self.interval = interval
        self.forecast_horizon = forecast_horizon
        self.include_holidays = include_holidays
        self.seasonality_mode = seasonality_mode
        self.model = None
        self.is_fitted = False
        self.metrics = {}
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def _prepare_data(self, ticket_data, date_column='created_at'):
        """
        Prepare the data for Prophet model.
        
        Parameters:
        -----------
        ticket_data : pandas.DataFrame
            Raw ticket data with at least a date column
        date_column : str
            Name of the column containing the timestamp
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame in Prophet format with 'ds' and 'y' columns
        """
        logger.info(f"Preparing data with {self.interval} interval")
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(ticket_data[date_column]):
            ticket_data[date_column] = pd.to_datetime(ticket_data[date_column])
        
        # Group by time interval and count tickets
        if self.interval == 'H':
            # Hourly aggregation
            df = ticket_data.set_index(date_column).resample('H').size().reset_index()
        elif self.interval == 'D':
            # Daily aggregation
            df = ticket_data.set_index(date_column).resample('D').size().reset_index()
        elif self.interval == 'W':
            # Weekly aggregation
            df = ticket_data.set_index(date_column).resample('W').size().reset_index()
        else:
            raise ValueError(f"Unsupported interval: {self.interval}")
        
        # Rename columns to Prophet format
        df.columns = ['ds', 'y']
        
        return df
    
    def _add_custom_features(self, df):
        """
        Add custom features for the model.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with at least 'ds' column
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with additional feature columns
        """
        # Extract time-based features
        df['hour'] = df['ds'].dt.hour
        df['day_of_week'] = df['ds'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['month'] = df['ds'].dt.month
        df['quarter'] = df['ds'].dt.quarter
        
        # Business hours indicator (for hourly models)
        if self.interval == 'H':
            df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] < 17) & 
                                     (~df['is_weekend'])).astype(int)
        
        return df
    
    def fit(self, ticket_data, date_column='created_at', custom_events=None, 
            add_holiday_effects=True, add_seasonality=True, validation_fraction=0.2):
        """
        Fit the forecasting model to historical ticket data.
        
        Parameters:
        -----------
        ticket_data : pandas.DataFrame
            Raw ticket data with at least a date column
        date_column : str
            Name of the column containing the timestamp
        custom_events : dict
            Dictionary of custom events with dates and scaling factors
            Format: {'event_name': {'dates': [...], 'scale': 1.5}}
        add_holiday_effects : bool
            Whether to include holiday effects
        add_seasonality : bool
            Whether to add custom seasonality based on interval
        validation_fraction : float
            Fraction of data to use for validation (0 to 1)
            
        Returns:
        --------
        self
        """
        # Prepare data in Prophet format
        df = self._prepare_data(ticket_data, date_column)
        
        # Add custom features
        df = self._add_custom_features(df)
        
        # Create model with appropriate configuration
        self.model = Prophet(
            seasonality_mode=self.seasonality_mode,
            daily_seasonality=True if self.interval == 'H' else False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            interval_width=0.95,  # 95% prediction intervals
            mcmc_samples=0  # Faster fitting without MCMC
        )
        
        # Add country holidays if requested
        if add_holiday_effects and self.include_holidays:
            logger.info("Adding holiday effects")
            self.model.add_country_holidays(country_name='US')
            self.model.add_country_holidays(country_name='CA')
        
        # Add custom seasonality based on interval
        if add_seasonality:
            if self.interval == 'H':
                # Add hourly seasonality
                self.model.add_seasonality(
                    name='hourly', 
                    period=24, 
                    fourier_order=12
                )
            
            # Add weekly seasonality with higher Fourier order for better fitting
            self.model.add_seasonality(
                name='weekly', 
                period=7, 
                fourier_order=10
            )
        
        # Add custom event regressors if provided
        if custom_events:
            for event_name, event_data in custom_events.items():
                event_dates = pd.to_datetime(event_data['dates'])
                scale = event_data.get('scale', 1.5)
                
                # Create dataframe with event dates
                event_df = pd.DataFrame({
                    'ds': event_dates,
                    'event': 1  # Event indicator
                })
                
                # Create regressor column in training data
                df[event_name] = 0
                for date in event_dates:
                    # Mark dates around the event
                    date_range = pd.date_range(
                        start=date - timedelta(days=1),
                        end=date + timedelta(days=1)
                    )
                    for d in date_range:
                        mask = (df['ds'].dt.date == d.date())
                        df.loc[mask, event_name] = 1
                
                # Add regressor to model
                self.model.add_regressor(event_name, mode='multiplicative' if scale > 1 else 'additive')
                
                logger.info(f"Added custom event: {event_name} with {len(event_dates)} occurrences")
        
        # Add business hours regressor for hourly forecasts
        if self.interval == 'H':
            self.model.add_regressor('is_business_hours', mode='multiplicative')
        
        # Split data for training and validation if requested
        if validation_fraction > 0:
            split_idx = int(len(df) * (1 - validation_fraction))
            train_df = df.iloc[:split_idx]
            valid_df = df.iloc[split_idx:]
            
            logger.info(f"Fitting model on {len(train_df)} data points, validating on {len(valid_df)}")
            
            # Fit the model
            self.model.fit(train_df)
            
            # Validate on holdout set
            future = self.model.make_future_dataframe(
                periods=len(valid_df),
                freq='H' if self.interval == 'H' else 'D',
                include_history=True
            )
            
            # Add custom regressors to future dataframe
            for col in df.columns:
                if col not in ['ds', 'y'] and col in train_df:
                    future[col] = df[col]
            
            # Make predictions
            forecast = self.model.predict(future)
            
            # Calculate validation metrics
            valid_comparison = pd.merge(
                valid_df[['ds', 'y']], 
                forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
                on='ds'
            )
            
            # Calculate metrics
            self.metrics = {
                'mape': np.mean(np.abs((valid_comparison['y'] - valid_comparison['yhat']) 
                                     / valid_comparison['y'])) * 100,
                'rmse': np.sqrt(np.mean((valid_comparison['y'] - valid_comparison['yhat'])**2)),
                'coverage': np.mean((valid_comparison['y'] >= valid_comparison['yhat_lower']) & 
                                  (valid_comparison['y'] <= valid_comparison['yhat_upper'])) * 100
            }
            
            logger.info(f"Validation metrics: MAPE={self.metrics['mape']:.2f}%, " 
                      f"RMSE={self.metrics['rmse']:.2f}, "
                      f"Coverage={self.metrics['coverage']:.2f}%")
            
            # Refit on all data for final model
            logger.info("Refitting model on all data")
        
        # Fit on all data
        self.model.fit(df)
        self.is_fitted = True
        
        return self
    
    def predict(self, periods=None, freq=None, include_history=True, custom_events=None):
        """
        Generate predictions for future time periods.
        
        Parameters:
        -----------
        periods : int
            Number of periods to predict (defaults to self.forecast_horizon)
        freq : str
            Frequency of predictions (defaults to self.interval)
        include_history : bool
            Whether to include historical data in the predictions
        custom_events : dict
            Dictionary of future events to include in the forecast
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with forecast results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if periods is None:
            periods = self.forecast_horizon
            
        if freq is None:
            if self.interval == 'H':
                freq = 'H'
            elif self.interval == 'D':
                freq = 'D'
            elif self.interval == 'W':
                freq = 'W'
        
        logger.info(f"Generating forecast for {periods} periods with frequency {freq}")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(
            periods=periods,
            freq=freq,
            include_history=include_history
        )
        
        # Add custom features
        future = self._add_custom_features(future)
        
        # Add custom events for future predictions
        if custom_events:
            for event_name, event_data in custom_events.items():
                if event_name not in future.columns:
                    future[event_name] = 0
                    
                event_dates = pd.to_datetime(event_data['dates'])
                
                for date in event_dates:
                    # Mark dates around the event
                    date_range = pd.date_range(
                        start=date - timedelta(days=1),
                        end=date + timedelta(days=1)
                    )
                    for d in date_range:
                        mask = (future['ds'].dt.date == d.date())
                        future.loc[mask, event_name] = 1
        
        # Make predictions
        forecast = self.model.predict(future)
        
        return forecast
    
    def evaluate(self, horizon=None, period='7D', initial='730D'):
        """
        Evaluate model with cross-validation.
        
        Parameters:
        -----------
        horizon : str
            Prediction horizon for cross-validation
        period : str
            Period between cutoff dates
        initial : str
            Initial training period
            
        Returns:
        --------
        dict
            Dictionary of performance metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        if horizon is None:
            if self.interval == 'H':
                horizon = '168 hours'  # 1 week
            elif self.interval == 'D':
                horizon = '30 days'    # 30 days
            elif self.interval == 'W':
                horizon = '12 weeks'   # 12 weeks
        
        logger.info(f"Running cross-validation with horizon={horizon}, period={period}, initial={initial}")
        
        # Run cross-validation
        cv_results = cross_validation(
            model=self.model,
            horizon=horizon,
            period=period,
            initial=initial,
            parallel="processes"
        )
        
        # Calculate performance metrics
        cv_metrics = performance_metrics(cv_results)
        
        # Store metrics
        self.metrics = {
            'mape': cv_metrics['mape'].mean(),
            'rmse': cv_metrics['rmse'].mean(),
            'coverage': (cv_metrics['coverage'].mean() * 100)
        }
        
        logger.info(f"CV metrics: MAPE={self.metrics['mape']:.2f}%, " 
                  f"RMSE={self.metrics['rmse']:.2f}, "
                  f"Coverage={self.metrics['coverage']:.2f}%")
        
        return self.metrics
    
    def plot_forecast(self, forecast=None, figsize=(12, 8), include_components=False):
        """
        Plot forecast results.
        
        Parameters:
        -----------
        forecast : pandas.DataFrame
            DataFrame with forecast results (if None, generates new forecast)
        figsize : tuple
            Figure size (width, height)
        include_components : bool
            Whether to include component plots
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure with forecast plots
        """
        if forecast is None:
            forecast = self.predict()
        
        if include_components:
            fig = self.model.plot_components(forecast, figsize=figsize)
        else:
            fig = self.model.plot(forecast, figsize=figsize)
            plt.title(f'Ticket Volume Forecast ({self.interval} interval)')
            plt.xlabel('Date')
            plt.ylabel('Ticket Volume')
            
        return fig
    
    def save_model(self, path):
        """
        Save the trained model to disk.
        
        Parameters:
        -----------
        path : str
            Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        model_data = {
            'model': self.model,
            'interval': self.interval,
            'forecast_horizon': self.forecast_horizon,
            'include_holidays': self.include_holidays,
            'seasonality_mode': self.seasonality_mode,
            'metrics': self.metrics,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load a trained model from disk.
        
        Parameters:
        -----------
        path : str
            Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.interval = model_data['interval']
        self.forecast_horizon = model_data['forecast_horizon']
        self.include_holidays = model_data['include_holidays']
        self.seasonality_mode = model_data['seasonality_mode']
        self.metrics = model_data['metrics']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Model loaded from {path}")
        
    def get_optimal_staffing(self, forecast=None, tickets_per_agent=5, 
                           buffer_factor=1.2, grouping='D'):
        """
        Calculate optimal staffing levels based on forecasted ticket volume.
        
        Parameters:
        -----------
        forecast : pandas.DataFrame
            DataFrame with forecast results (if None, generates new forecast)
        tickets_per_agent : float
            Number of tickets an agent can handle per hour
        buffer_factor : float
            Buffer to account for unexpected volume (e.g., 1.2 = 20% buffer)
        grouping : str
            How to group results ('H' for hourly, 'D' for daily)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with recommended staffing levels
        """
        if forecast is None:
            forecast = self.predict()
        
        # Focus on future predictions only
        future_forecast = forecast[forecast['ds'] > datetime.now()]
        
        # Calculate required agents
        future_forecast['required_agents'] = np.ceil(
            future_forecast['yhat'] / tickets_per_agent * buffer_factor
        )
        
        # Add upper bound staffing for peak scenarios
        future_forecast['required_agents_upper'] = np.ceil(
            future_forecast['yhat_upper'] / tickets_per_agent * buffer_factor
        )
        
        # Group if needed
        if grouping == 'D' and self.interval == 'H':
            # Group by day
            staffing = future_forecast.set_index('ds').resample('D').agg({
                'yhat': 'sum',
                'yhat_upper': 'sum',
                'required_agents': 'max',
                'required_agents_upper': 'max'
            }).reset_index()
            
            # Recalculate daily requirements
            staffing['daily_tickets'] = staffing['yhat']
            staffing['daily_tickets_upper'] = staffing['yhat_upper']
            staffing['business_hours'] = 8  # Assuming 8 business hours
            staffing['agents_needed'] = np.ceil(
                staffing['daily_tickets'] / (tickets_per_agent * staffing['business_hours']) * buffer_factor
            )
            staffing['agents_needed_upper'] = np.ceil(
                staffing['daily_tickets_upper'] / (tickets_per_agent * staffing['business_hours']) * buffer_factor
            )
        else:
            staffing = future_forecast[['ds', 'yhat', 'yhat_upper', 'required_agents', 'required_agents_upper']]
            staffing['agents_needed'] = staffing['required_agents']
            staffing['agents_needed_upper'] = staffing['required_agents_upper']
        
        return staffing[['ds', 'yhat', 'agents_needed', 'agents_needed_upper']]


# Example usage
if __name__ == "__main__":
    import sys
    import os
    
    # Add parent directory to path for imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from data.generate_sample_data import generate_tickets, generate_agents, generate_customers
    
    # Generate sample data
    print("Generating sample data...")
    agents_df = generate_agents()
    customers_df = generate_customers()
    tickets_df = generate_tickets(agents_df, customers_df)
    
    # Create forecaster
    print("Creating and fitting forecaster...")
    forecaster = TicketDemandForecaster(interval='H', forecast_horizon=168)
    
    # Define some custom events
    custom_events = {
        'marketing_campaign': {
            'dates': ['2023-03-15', '2023-06-20', '2023-09-10', '2023-11-25'],
            'scale': 1.8  # Expect 80% more tickets
        },
        'product_release': {
            'dates': ['2023-02-01', '2023-05-15', '2023-08-30', '2023-12-01'],
            'scale': 2.0  # Expect 100% more tickets
        }
    }
    
    # Fit model
    forecaster.fit(
        tickets_df,
        date_column='created_at',
        custom_events=custom_events,
        validation_fraction=0.2
    )
    
    # Make predictions
    forecast = forecaster.predict(periods=168)  # 1 week ahead (hourly)
    
    # Evaluate model
    metrics = forecaster.evaluate()
    
    # Get staffing recommendations
    staffing = forecaster.get_optimal_staffing(forecast, tickets_per_agent=4)
    
    # Print staffing for next 7 days (daily)
    print("\nRecommended staffing for next 7 days:")
    daily_staffing = staffing.set_index('ds').resample('D').max()['agents_needed']
    print(daily_staffing.head(7))
    
    # Save model
    os.makedirs('../models/saved', exist_ok=True)
    forecaster.save_model('../models/saved/ticket_demand_forecaster.joblib')
    
    # Plot results
    fig = forecaster.plot_forecast(forecast)
    plt.tight_layout()
    
    # Save plot
    os.makedirs('../output', exist_ok=True)
    plt.savefig('../output/ticket_demand_forecast.png') 