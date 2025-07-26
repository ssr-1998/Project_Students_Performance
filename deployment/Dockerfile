# Using slim Python Image that is suitable for ML Inference
FROM python:3.11.9-slim

# Environment Variables
#   PYTHONDONTWRITEBYTECODE=1: Prevents Python from creating `.pyc` files → reduces clutter.
#   PYTHONUNBUFFERED=1: Forces stdout/stderr to be sent directly to the terminal (real-time logs), especially important in Docker logs.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install System-level Dependencies required by ML Packages:
#   Installs build tools (gcc, make, etc.) needed by:
#    - numpy
#    - scikit-learn
#    - catboost
#    - `--no-install-recommends` keeps image smaller.
#    - `rm -rf /var/lib/apt/lists/*` cleans up cache → reduces image size.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the Working Directory:
WORKDIR /app

# Copy & Install requirements.txt:
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy remaining files and folders:
COPY . /app

# Exposing the Flask's Default Port:
EXPOSE 5000

# Start the Flask App
CMD ["python", "app.py"]
