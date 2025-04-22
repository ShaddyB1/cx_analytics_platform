"""
Customer Experience Analytics Pipeline

This DAG orchestrates the end-to-end data pipeline for the customer experience
analytics platform, including data ingestion, transformation, and model training.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.operators.dummy import DummyOperator
import os
import sys

# Add the project path to allow importing from project modules
sys.path.append('/opt/airflow/dags/cx_analytics_platform')

# Import custom modules
from models.demand_forecasting import TicketDemandForecaster
from models.ticket_categorization import TicketCategorizer

# Define default arguments
default_args = {
    'owner': 'data_science',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email': ['data_alerts@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'catchup': False,
}

# Define the DAG
dag = DAG(
    'cx_analytics_pipeline',
    default_args=default_args,
    description='End-to-end data pipeline for customer experience analytics',
    schedule_interval='0 2 * * *',  # Run daily at 2 AM
    max_active_runs=1,
    tags=['customer_experience', 'analytics', 'data_science'],
)

# Define the tasks
start = DummyOperator(
    task_id='start',
    dag=dag,
)

# Task to ingest data from various sources
ingest_data = BashOperator(
    task_id='ingest_data',
    bash_command='python /opt/airflow/dags/cx_analytics_platform/data/ingest_data.py',
    dag=dag,
)

# Task to check data quality
check_data_quality = BashOperator(
    task_id='check_data_quality',
    bash_command='python /opt/airflow/dags/cx_analytics_platform/data/check_data_quality.py',
    dag=dag,
)

# Task to run dbt transformations
run_dbt = BashOperator(
    task_id='run_dbt',
    bash_command='cd /opt/airflow/dags/cx_analytics_platform/dbt_project && dbt run --profiles-dir /opt/airflow/dags/cx_analytics_platform/dbt_project/profiles',
    dag=dag,
)

# Task to test dbt transformations
test_dbt = BashOperator(
    task_id='test_dbt',
    bash_command='cd /opt/airflow/dags/cx_analytics_platform/dbt_project && dbt test --profiles-dir /opt/airflow/dags/cx_analytics_platform/dbt_project/profiles',
    dag=dag,
)

# Task to generate documentation
generate_docs = BashOperator(
    task_id='generate_docs',
    bash_command='cd /opt/airflow/dags/cx_analytics_platform/dbt_project && dbt docs generate --profiles-dir /opt/airflow/dags/cx_analytics_platform/dbt_project/profiles',
    dag=dag,
)

# Function to train demand forecasting model
def train_demand_forecast():
    """Train the demand forecasting model on the latest data"""
    import pandas as pd
    from sqlalchemy import create_engine
    
    # Connect to the database
    engine = create_engine('postgresql://user:password@localhost:5432/cx_analytics')
    
    # Query ticket data from the data mart
    query = """
    SELECT 
        ticket_id, 
        created_at, 
        type, 
        priority,
        status
    FROM marts.fact_tickets
    WHERE created_at >= CURRENT_DATE - INTERVAL '365 days'
    """
    tickets_df = pd.read_sql(query, engine)
    
    # Initialize and train the forecaster
    forecaster = TicketDemandForecaster(interval='H', forecast_horizon=168)
    
    # Define custom events
    custom_events = {
        'marketing_campaign': {
            'dates': ['2023-03-15', '2023-06-20', '2023-09-10', '2023-11-25'],
            'scale': 1.8
        },
        'product_release': {
            'dates': ['2023-02-01', '2023-05-15', '2023-08-30', '2023-12-01'],
            'scale': 2.0
        }
    }
    
    # Fit the model
    forecaster.fit(
        tickets_df,
        date_column='created_at',
        custom_events=custom_events,
        validation_fraction=0.2
    )
    
    # Save the model
    forecaster.save_model('/opt/airflow/dags/cx_analytics_platform/models/saved/ticket_demand_forecaster.joblib')
    
    # Generate predictions for the next 7 days
    forecast = forecaster.predict(periods=168)
    
    # Save forecast to database
    forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast_df.columns = ['forecast_date', 'predicted_tickets', 'lower_bound', 'upper_bound']
    forecast_df.to_sql('ticket_volume_forecast', engine, schema='marts', if_exists='replace', index=False)
    
    return "Demand forecasting model trained and forecast saved"

# Task to train demand forecasting model
train_forecasting_model = PythonOperator(
    task_id='train_forecasting_model',
    python_callable=train_demand_forecast,
    dag=dag,
)

# Function to train ticket categorization model
def train_categorization_model():
    """Train the ticket categorization model on the latest data"""
    import pandas as pd
    from sqlalchemy import create_engine
    
    # Connect to the database
    engine = create_engine('postgresql://user:password@localhost:5432/cx_analytics')
    
    # Query ticket data from the data mart
    ticket_query = """
    SELECT 
        ticket_id, 
        customer_id,
        content,
        type, 
        priority,
        status,
        created_at
    FROM marts.fact_tickets
    WHERE created_at >= CURRENT_DATE - INTERVAL '365 days'
    """
    tickets_df = pd.read_sql(ticket_query, engine)
    
    # Query customer data
    customer_query = """
    SELECT 
        customer_id,
        account_type,
        ticket_frequency,
        signup_date
    FROM marts.dim_customers
    """
    customers_df = pd.read_sql(customer_query, engine)
    
    # Initialize and train the categorizer
    categorizer = TicketCategorizer(use_bert=False)
    
    # Fit the model
    categorizer.fit(
        tickets_df,
        text_column='content',
        type_column='type',
        priority_column='priority',
        customer_data=customers_df,
        customer_id_column='customer_id'
    )
    
    # Save the model
    categorizer.save_model('/opt/airflow/dags/cx_analytics_platform/models/saved/ticket_categorizer.joblib')
    
    return "Ticket categorization model trained and saved"

# Task to train ticket categorization model
train_categorization_model = PythonOperator(
    task_id='train_categorization_model',
    python_callable=train_categorization_model,
    dag=dag,
)

# Task to update dashboards
update_dashboards = BashOperator(
    task_id='update_dashboards',
    bash_command='python /opt/airflow/dags/cx_analytics_platform/dashboards/update_dashboards.py',
    dag=dag,
)

# Task to send a notification when the pipeline is complete
send_notification = BashOperator(
    task_id='send_notification',
    bash_command='python /opt/airflow/dags/cx_analytics_platform/utils/send_notification.py --subject "CX Analytics Pipeline Complete" --body "The customer experience analytics pipeline has completed successfully."',
    dag=dag,
)

end = DummyOperator(
    task_id='end',
    dag=dag,
)

# Define the task dependencies
start >> ingest_data >> check_data_quality >> run_dbt
run_dbt >> test_dbt >> generate_docs
test_dbt >> train_forecasting_model
test_dbt >> train_categorization_model
generate_docs >> update_dashboards
train_forecasting_model >> update_dashboards
train_categorization_model >> update_dashboards
update_dashboards >> send_notification >> end 