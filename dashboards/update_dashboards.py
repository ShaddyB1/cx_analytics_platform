#!/usr/bin/env python3
"""
Dashboard Update Script

This script updates dashboard data by generating forecasts and refreshing
cached metrics. It is designed to be run by Airflow as part of the daily pipeline.
"""

import os
import sys
import logging
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from models.demand_forecasting import TicketDemandForecaster
from models.agent_assignment import AgentAssignmentOptimizer

def get_database_connection():
    """Create a database connection"""
    conn_string = os.getenv('DB_CONNECTION_STRING', 'postgresql://user:password@localhost:5432/cx_analytics')
    return create_engine(conn_string)

def update_forecast():
    """Update ticket volume forecast and save to database"""
    logger.info("Updating ticket volume forecast")
    
    # Connect to database
    engine = get_database_connection()
    
    # Load the saved model
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'models/saved/ticket_demand_forecaster.joblib')
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return False
    
    # Load the model
    forecaster = TicketDemandForecaster(model_path=model_path)
    
    # Generate forecast for next 7 days (hourly)
    forecast = forecaster.predict(periods=168)  # 7 days * 24 hours
    
    # Prepare forecast data for database
    forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast_df.columns = ['forecast_date', 'predicted_tickets', 'lower_bound', 'upper_bound']
    
    # Save to database
    forecast_df.to_sql('ticket_volume_forecast', engine, schema='marts', if_exists='replace', index=False)
    
    logger.info(f"Saved forecast for {len(forecast_df)} periods to database")
    return True

def update_agent_schedule():
    """Update agent schedule based on forecast and save to database"""
    logger.info("Generating optimal agent schedule")
    
    # Connect to database
    engine = get_database_connection()
    
    # Get the forecast
    forecast_query = """
    SELECT forecast_date, predicted_tickets
    FROM marts.ticket_volume_forecast
    ORDER BY forecast_date
    """
    forecast_df = pd.read_sql(forecast_query, engine)
    forecast_df.set_index('forecast_date', inplace=True)
    
    # Get available agents
    agents_query = """
    SELECT agent_id, name, team, skill_level, specialties
    FROM marts.dim_agents
    WHERE is_active = true
    """
    agents_df = pd.read_sql(agents_query, engine)
    
    # Create optimizer
    optimizer = AgentAssignmentOptimizer()
    
    # Generate optimal schedule
    schedule_df = optimizer.optimize_agent_scheduling(
        forecast_df, 
        agents_df,
        shift_duration=8,
        window_days=7
    )
    
    # Save to database
    schedule_df.to_sql('agent_schedule', engine, schema='marts', if_exists='replace', index=False)
    
    logger.info(f"Saved schedule with {len(schedule_df)} shifts to database")
    return True

def update_dashboard_metrics():
    """Update pre-calculated dashboard metrics for faster loading"""
    logger.info("Updating dashboard metrics")
    
    # Connect to database
    engine = get_database_connection()
    
    # Calculate key metrics for various time periods
    time_periods = [7, 30, 90]
    metrics = {}
    
    for days in time_periods:
        # Get ticket metrics for the time period
        query = f"""
        SELECT 
            COUNT(*) as total_tickets,
            AVG(response_time_min) as avg_response_time,
            AVG(resolution_time_min) as avg_resolution_time,
            AVG(satisfaction_score) as avg_satisfaction,
            COUNT(CASE WHEN status IN ('Resolved', 'Closed') THEN 1 END) / COUNT(*) as resolution_rate
        FROM marts.fact_tickets
        WHERE created_at >= CURRENT_DATE - INTERVAL '{days} days'
        """
        df = pd.read_sql(query, engine)
        
        # Store metrics
        metrics[f"{days}_days"] = {
            "total_tickets": int(df['total_tickets'].iloc[0]),
            "avg_response_time": float(df['avg_response_time'].iloc[0]),
            "avg_resolution_time": float(df['avg_resolution_time'].iloc[0]),
            "avg_satisfaction": float(df['avg_satisfaction'].iloc[0]),
            "resolution_rate": float(df['resolution_rate'].iloc[0]),
            "updated_at": datetime.now().isoformat()
        }
    
    # Save metrics to database or file
    metrics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cached_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Saved metrics to {metrics_path}")
    return True

def update_kpi_charts():
    """Update pre-generated chart data for KPIs"""
    logger.info("Updating KPI charts")
    
    # Connect to database
    engine = get_database_connection()
    
    # Generate time series data for key metrics
    query = """
    SELECT 
        DATE_TRUNC('day', created_at) as date,
        COUNT(*) as tickets,
        AVG(response_time_min) as avg_response_time,
        AVG(resolution_time_min) as avg_resolution_time,
        AVG(satisfaction_score) as avg_satisfaction
    FROM marts.fact_tickets
    WHERE created_at >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY DATE_TRUNC('day', created_at)
    ORDER BY date
    """
    kpi_df = pd.read_sql(query, engine)
    
    # Calculate 7-day moving averages
    kpi_df['tickets_7d_avg'] = kpi_df['tickets'].rolling(7).mean()
    kpi_df['response_7d_avg'] = kpi_df['avg_response_time'].rolling(7).mean()
    kpi_df['resolution_7d_avg'] = kpi_df['avg_resolution_time'].rolling(7).mean()
    kpi_df['satisfaction_7d_avg'] = kpi_df['avg_satisfaction'].rolling(7).mean()
    
    # Save to database
    kpi_df.to_sql('kpi_trends', engine, schema='marts', if_exists='replace', index=False)
    
    logger.info(f"Saved KPI trends for {len(kpi_df)} days to database")
    return True

def main():
    """Run all dashboard update tasks"""
    logger.info("Starting dashboard update process")
    
    # Update forecast
    if not update_forecast():
        logger.error("Failed to update forecast")
    
    # Update agent schedule
    if not update_agent_schedule():
        logger.error("Failed to update agent schedule")
    
    # Update dashboard metrics
    if not update_dashboard_metrics():
        logger.error("Failed to update dashboard metrics")
    
    # Update KPI charts
    if not update_kpi_charts():
        logger.error("Failed to update KPI charts")
    
    logger.info("Dashboard update process complete")

if __name__ == "__main__":
    main() 