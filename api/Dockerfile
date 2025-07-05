# FastAPI Backend Docker Image
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy application files
COPY . .

# Expose FastAPI port
EXPOSE 8080

# Default command to run Uvicorn server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080", "--reload"] 