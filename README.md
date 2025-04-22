# Customer Experience Analytics Platform



<p align="center">
  <img src="docs/architecture_diagram.png" alt="Architecture Diagram" width="800"/>
</p>

## Project Overview

This platform demonstrates a full-stack data science approach to customer experience optimization with the following components:

1. **Predictive Models** for:
   - Customer service demand forecasting (time series)
   - Ticket categorization and prioritization (NLP)
   - Optimal agent-ticket assignment (optimization algorithms)

2. **Data Engineering Pipeline**:
   - Data extraction and transformation using dbt
   - Scheduled workflows with Airflow
   - Containerized deployment with Docker

3. **Analytics Dashboard**:
   - Interactive KPI visualization
   - Agent performance metrics
   - Customer satisfaction tracking

## 📊 Live Demo

Screenshots of the analytics dashboard:

<p align="center">
  <img src="docs/dashboard_screenshot.png" alt="Dashboard Screenshot" width="800"/>
</p>

## 🧠 Machine Learning Models

The platform includes three key predictive models:

### Demand Forecasting Model
- Predicts ticket volume by time period
- Uses Facebook Prophet for time series forecasting
- Accounts for seasonality, holidays, and special events

### Ticket Categorization Model
- Classifies tickets by type and priority
- Implements NLP techniques for text analysis
- Helps route tickets to appropriate agents

### Agent Assignment Optimizer
- Optimizes assignment of tickets to agents
- Considers agent skills, workload, and ticket priority
- Improves response time and customer satisfaction

## 🛠️ Technology Stack

- **Data Processing**: Python, pandas
- **Data Transformation**: dbt
- **Machine Learning**: scikit-learn, Prophet, NLP libraries
- **Orchestration**: Apache Airflow
- **Database**: PostgreSQL
- **Visualization**: Streamlit, Plotly
- **Deployment**: Docker, Docker Compose

## 🚀 Installation & Setup

### Prerequisites
- Docker and Docker Compose
- Python 3.10+

### Quick Start
1. Clone the repository:
   ```bash
   git clone https://github.com/ShaddyB1/cx_analytics_platform.git
   cd cx_analytics_platform
   ```

2. Start the containers:
   ```bash
   docker-compose up -d
   ```

3. Access the components:
   - Streamlit Dashboard: http://localhost:8501
   - Airflow: http://localhost:8080

## 📁 Project Structure

```
cx_analytics_platform/
├── data/                  # Sample datasets and data generators
├── models/                # Predictive models implementations
├── dashboards/            # Dashboard code and visualization
├── dbt_project/           # dbt models for data transformation
├── airflow_dags/          # Airflow DAGs for workflow orchestration
├── docs/                  # Documentation and architecture diagrams
├── docker-compose.yml     # Docker Compose configuration
└── requirements.txt       # Project dependencies
```

## 📊 Features & Capabilities

- **Real-time Analytics**: Monitor key performance indicators in real-time
- **Predictive Insights**: Forecast future ticket volumes and resource needs
- **Optimization**: Improve agent scheduling and ticket routing
- **Automated Pipelines**: Schedule data transformations and model updates
- **Containerized Deployment**: Easy setup and consistent environment

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Shadrack Boateng Addo**

- Portfolio: [https://shadrackaddo.net](https://shadrackaddo.net)
- GitHub: [@ShaddyB1](https://github.com/ShaddyB1)
- LinkedIn: [Shadrack Addo](https://www.linkedin.com/in/shadrackaddo/) 
