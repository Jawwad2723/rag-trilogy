# Use lightweight python image
FROM python:3.11-slim

# Install system dependencies needed for python packages and tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install poetry
RUN pip install --no-cache-dir poetry

# Configure poetry to not create a virtual environment inside container
RUN poetry config virtualenvs.create false

# Copy dependency configuration
COPY pyproject.toml poetry.lock* /app/

# Install dependencies including dev dependencies (needed for evaluation and tools)
RUN poetry install --no-interaction --no-ansi --no-root

# Copy application code
COPY src /app/src
COPY config.yaml /app/config.yaml

# Expose port
EXPOSE 8000

# Start command
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
