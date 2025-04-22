FROM python:3.10-slim

WORKDIR /dbt

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    ssh-client \
    software-properties-common \
    make \
    build-essential \
    ca-certificates \
    libpq-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install dbt and dependencies
RUN pip install --no-cache-dir \
    dbt-core==1.5.0 \
    dbt-postgres==1.5.0 \
    dbt-utils==1.1.1

# Create dbt directories
RUN mkdir -p /dbt/logs /dbt/target /dbt/profiles

# Copy dbt project files
COPY ./dbt_project /dbt/

# Create dbt profile
RUN echo "\
cx_analytics:\n\
  outputs:\n\
    dev:\n\
      type: postgres\n\
      host: postgres\n\
      user: '{{ env_var(\"POSTGRES_USER\", \"cxuser\") }}'\n\
      password: '{{ env_var(\"POSTGRES_PASSWORD\", \"cxpassword\") }}'\n\
      port: 5432\n\
      dbname: '{{ env_var(\"POSTGRES_DB\", \"cx_analytics\") }}'\n\
      schema: marts\n\
      threads: 4\n\
  target: dev\n\
" > /dbt/profiles/profiles.yml

# Set workdir for dbt commands
WORKDIR /dbt

# Default command
CMD ["dbt", "run", "--profiles-dir", "/dbt/profiles"] 