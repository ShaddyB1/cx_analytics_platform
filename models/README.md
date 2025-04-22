# Predictive Models

This directory contains the implementation of three predictive models crucial for optimizing customer service operations:

## 1. Demand Forecasting Model

Location: `demand_forecasting.py`

**Purpose**: Predict expected ticket volume by time of day, day of week, and other temporal factors to optimize agent staffing levels.

**Model Type**: Time series forecasting using Facebook Prophet with:
- Daily, weekly, and annual seasonality components
- Holiday effects
- Special event adjustments
- Customizable forecasting horizon (1 day to 4 weeks)

**Features**:
- Historical ticket volume by time interval
- Special event flags
- Marketing campaign indicators
- Product release dates

**Performance Metrics**:
- MAPE (Mean Absolute Percentage Error)
- RMSE (Root Mean Square Error)
- Coverage (% of actual values within prediction intervals)

## 2. Ticket Categorization Model

Location: `ticket_categorization.py`

**Purpose**: Automatically classify and prioritize incoming customer service tickets.

**Model Type**: NLP-based classification using:
- BERT-based text embedding
- Multi-label classification for ticket type
- Priority scoring based on ticket content, customer history, and business rules

**Features**:
- Ticket text content
- Customer attributes
- Historical resolution patterns
- Word patterns indicating urgency

**Performance Metrics**:
- Classification accuracy
- Precision/Recall by category
- Priority scoring correlation with agent-assigned priority

## 3. Agent-Ticket Assignment Optimization

Location: `agent_assignment.py`

**Purpose**: Optimize the assignment of tickets to agents to minimize wait times and maximize resolution efficiency.

**Model Type**: Reinforcement learning and combinatorial optimization:
- Agent skill/specialty matching
- Workload balancing
- Priority-based queueing
- Real-time reassignment as conditions change

**Features**:
- Agent skills and specialties
- Current workload
- Ticket priority and complexity
- Historical performance with similar tickets

**Performance Metrics**:
- Average resolution time
- Agent utilization balance
- First contact resolution rate
- Customer satisfaction correlation

## Model Training Pipeline

The `training_pipeline.py` script orchestrates the end-to-end training of all models:

1. Data preprocessing and feature engineering
2. Hyperparameter tuning using cross-validation
3. Model training with early stopping
4. Model evaluation on holdout test sets
5. Model serialization and versioning

## Model Serving

The `model_server.py` module provides an API for real-time predictions:

- RESTful endpoints for each model
- Batch prediction capabilities
- Monitoring for model drift
- A/B testing of model versions

## Usage

See `model_examples.ipynb` for interactive examples of how to use these models.

For production deployment, refer to the deployment documentation in `../docs/deployment.md`. 