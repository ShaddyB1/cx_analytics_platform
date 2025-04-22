FROM python:3.10-slim

WORKDIR /app

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make directory structure
RUN mkdir -p /app/models /app/dashboards /app/data

# Set Python path
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Copy application files
COPY ./dashboards /app/dashboards
COPY ./models /app/models

# Expose the port for Streamlit
EXPOSE 8501

# Set the startup command to run the Streamlit dashboard
CMD ["streamlit", "run", "/app/dashboards/app.py", "--server.port=8501", "--server.address=0.0.0.0"] 