#!/usr/bin/env python3
"""
Customer Experience Analytics Dashboard

A Streamlit app for visualizing customer service analytics and metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from sqlalchemy import create_engine
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.demand_forecasting import TicketDemandForecaster
from models.ticket_categorization import TicketCategorizer
from models.agent_assignment import AgentAssignmentOptimizer

# Page configuration
st.set_page_config(
    page_title="CX Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Database connection
@st.cache_resource
def get_database_connection():
    """Create a database connection that can be reused"""
    conn_string = os.getenv('DB_CONNECTION_STRING', 'postgresql://user:password@localhost:5432/cx_analytics')
    return create_engine(conn_string)

# Load data functions
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_ticket_data(days=30):
    """Load ticket data from the database"""
    engine = get_database_connection()
    query = f"""
    SELECT 
        t.ticket_id, 
        t.customer_id,
        t.agent_id,
        t.created_at,
        t.resolved_at,
        t.type, 
        t.priority,
        t.status,
        t.channel,
        t.response_time_min,
        t.resolution_time_min,
        t.satisfaction_score,
        a.name as agent_name,
        a.team as agent_team,
        c.account_type
    FROM marts.fact_tickets t
    LEFT JOIN marts.dim_agents a ON t.agent_id = a.agent_id
    LEFT JOIN marts.dim_customers c ON t.customer_id = c.customer_id
    WHERE t.created_at >= CURRENT_DATE - INTERVAL '{days} days'
    """
    return pd.read_sql(query, engine)

@st.cache_data(ttl=3600)
def load_agent_performance(days=30):
    """Load agent performance data from the database"""
    engine = get_database_connection()
    query = f"""
    SELECT * 
    FROM marts.agent_performance_metrics
    WHERE date >= CURRENT_DATE - INTERVAL '{days} days'
    """
    return pd.read_sql(query, engine)

@st.cache_data(ttl=3600)
def load_ticket_forecast():
    """Load ticket volume forecast from the database"""
    engine = get_database_connection()
    query = """
    SELECT * 
    FROM marts.ticket_volume_forecast
    WHERE forecast_date >= CURRENT_DATE
    ORDER BY forecast_date
    """
    return pd.read_sql(query, engine)

# Helper functions for visualizations
def create_ticket_volume_chart(tickets_df):
    """Create a chart showing ticket volume over time"""
    # Group by day and count tickets
    daily_volume = tickets_df.groupby(pd.Grouper(key='created_at', freq='D')).size().reset_index(name='count')
    
    # Create figure
    fig = px.line(
        daily_volume, 
        x='created_at', 
        y='count',
        title='Daily Ticket Volume',
        labels={'created_at': 'Date', 'count': 'Number of Tickets'},
        line_shape='spline'
    )
    fig.update_layout(height=400)
    return fig

def create_ticket_type_chart(tickets_df):
    """Create a chart showing distribution of ticket types"""
    type_counts = tickets_df['type'].value_counts().reset_index()
    type_counts.columns = ['type', 'count']
    
    fig = px.pie(
        type_counts, 
        values='count', 
        names='type',
        title='Ticket Types Distribution',
        hole=0.4
    )
    fig.update_layout(height=400)
    return fig

def create_response_time_chart(tickets_df):
    """Create a chart showing average response time by priority"""
    response_by_priority = tickets_df.groupby('priority')['response_time_min'].mean().reset_index()
    
    # Sort by expected response time
    priority_order = ['Critical', 'High', 'Medium', 'Low']
    response_by_priority['priority'] = pd.Categorical(
        response_by_priority['priority'], 
        categories=priority_order, 
        ordered=True
    )
    response_by_priority = response_by_priority.sort_values('priority')
    
    fig = px.bar(
        response_by_priority,
        x='priority',
        y='response_time_min',
        title='Average Response Time by Priority',
        labels={'priority': 'Priority', 'response_time_min': 'Response Time (minutes)'},
        color='priority',
        color_discrete_map={
            'Critical': '#FF0000',
            'High': '#FF8000',
            'Medium': '#FFFF00',
            'Low': '#00FF00'
        }
    )
    fig.update_layout(height=400)
    return fig

def create_agent_performance_chart(performance_df):
    """Create a chart showing agent performance metrics"""
    # Calculate average performance metrics per agent
    agent_metrics = performance_df.groupby('agent_name').agg({
        'tickets_resolved': 'sum',
        'avg_resolution_time_min': 'mean',
        'sla_compliance_rate': 'mean',
        'first_contact_resolution_rate': 'mean',
        'avg_satisfaction_score': 'mean'
    }).reset_index()
    
    # Sort by tickets resolved
    agent_metrics = agent_metrics.sort_values('tickets_resolved', ascending=False).head(10)
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add bars for tickets resolved
    fig.add_trace(
        go.Bar(
            x=agent_metrics['agent_name'],
            y=agent_metrics['tickets_resolved'],
            name='Tickets Resolved',
            marker_color='rgb(55, 83, 109)'
        ),
        secondary_y=False,
    )
    
    # Add line for avg resolution time
    fig.add_trace(
        go.Scatter(
            x=agent_metrics['agent_name'],
            y=agent_metrics['avg_resolution_time_min'],
            name='Avg Resolution Time (min)',
            mode='lines+markers',
            marker=dict(size=8),
            line=dict(width=2, color='red')
        ),
        secondary_y=True,
    )
    
    # Set titles
    fig.update_layout(
        title_text='Top 10 Agents by Tickets Resolved',
        height=500
    )
    
    # Set x-axis title
    fig.update_xaxes(title_text='Agent')
    
    # Set y-axes titles
    fig.update_yaxes(title_text='Tickets Resolved', secondary_y=False)
    fig.update_yaxes(title_text='Resolution Time (min)', secondary_y=True)
    
    return fig

def create_forecast_chart(forecast_df):
    """Create a chart showing ticket volume forecast"""
    fig = go.Figure()
    
    # Add predicted volume line
    fig.add_trace(
        go.Scatter(
            x=forecast_df['forecast_date'],
            y=forecast_df['predicted_tickets'],
            mode='lines',
            name='Predicted Volume',
            line=dict(color='blue', width=2)
        )
    )
    
    # Add confidence interval
    fig.add_trace(
        go.Scatter(
            x=forecast_df['forecast_date'],
            y=forecast_df['upper_bound'],
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=forecast_df['forecast_date'],
            y=forecast_df['lower_bound'],
            mode='lines',
            name='Lower Bound',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(0, 176, 246, 0.2)',
            showlegend=False
        )
    )
    
    fig.update_layout(
        title='Ticket Volume Forecast (Next 7 Days)',
        xaxis_title='Date',
        yaxis_title='Predicted Tickets',
        height=400
    )
    
    return fig

def create_performance_metrics(tickets_df, performance_df):
    """Calculate and display key performance metrics"""
    # Calculate metrics
    total_tickets = len(tickets_df)
    avg_response_time = tickets_df['response_time_min'].mean()
    avg_resolution_time = tickets_df['resolution_time_min'].mean()
    resolved_tickets = tickets_df[tickets_df['status'].isin(['Resolved', 'Closed'])]
    resolution_rate = len(resolved_tickets) / total_tickets if total_tickets > 0 else 0
    csat = tickets_df['satisfaction_score'].mean()
    
    # Calculate SLA compliance
    sla_thresholds = {
        'Critical': 30,
        'High': 60,
        'Medium': 240,
        'Low': 480
    }
    
    tickets_df['within_sla'] = False
    for priority, threshold in sla_thresholds.items():
        mask = (tickets_df['priority'] == priority) & (tickets_df['response_time_min'] <= threshold)
        tickets_df.loc[mask, 'within_sla'] = True
    
    sla_compliance = tickets_df['within_sla'].mean()
    
    # Average agent performance metrics
    avg_tickets_per_hour = performance_df['tickets_per_hour'].mean()
    avg_fcr = performance_df['first_contact_resolution_rate'].mean()
    
    # Display metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Total Tickets", value=f"{total_tickets:,}")
        st.metric(label="Avg Response Time", value=f"{avg_response_time:.1f} min")
        st.metric(label="SLA Compliance", value=f"{sla_compliance:.1%}")
        
    with col2:
        st.metric(label="Resolution Rate", value=f"{resolution_rate:.1%}")
        st.metric(label="Avg Resolution Time", value=f"{avg_resolution_time:.1f} min")
        st.metric(label="Tickets/Hour", value=f"{avg_tickets_per_hour:.2f}")
        
    with col3:
        st.metric(label="Customer Satisfaction", value=f"{csat:.2f}/5.0")
        st.metric(label="First Contact Resolution", value=f"{avg_fcr:.1%}")
        st.metric(label="Active Agents", value=f"{performance_df['agent_id'].nunique()}")

# Main application
def main():
    # Sidebar for filtering
    st.sidebar.title("CX Analytics Dashboard")
    
    # Date range filter
    date_range = st.sidebar.slider(
        "Select Date Range (days)",
        min_value=7,
        max_value=90,
        value=30,
        step=1
    )
    
    # Load data
    with st.spinner("Loading data..."):
        tickets_df = load_ticket_data(days=date_range)
        performance_df = load_agent_performance(days=date_range)
        forecast_df = load_ticket_forecast()
    
    # Title and summary
    st.title("Customer Experience Analytics Dashboard")
    
    # Current state overview
    st.header("Current State Overview")
    create_performance_metrics(tickets_df, performance_df)
    
    # First row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_ticket_volume_chart(tickets_df), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_ticket_type_chart(tickets_df), use_container_width=True)
    
    # Second row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_response_time_chart(tickets_df), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_forecast_chart(forecast_df), use_container_width=True)
    
    # Agent performance section
    st.header("Agent Performance")
    st.plotly_chart(create_agent_performance_chart(performance_df), use_container_width=True)
    
    # Raw data section (expandable)
    with st.expander("View Raw Data"):
        tab1, tab2 = st.tabs(["Ticket Data", "Agent Performance Data"])
        
        with tab1:
            st.dataframe(tickets_df)
            
        with tab2:
            st.dataframe(performance_df)

if __name__ == "__main__":
    main()