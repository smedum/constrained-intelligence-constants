
# Dockerfile for Constrained Intelligence Constants
# Build a containerized environment for running the package

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r requirements-dev.txt

# Copy the entire package
COPY . .

# Install the package in development mode
RUN pip install -e .

# Run tests to verify installation
RUN pytest tests/ -v

# Default command: Run validation
CMD ["python", "validation/experimental_validation.py"]

# Optional: Run Jupyter notebook
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
