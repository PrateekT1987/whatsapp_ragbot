FROM python:3.11-slim

WORKDIR /app

# Install system dependencies with retry logic
RUN for i in 1 2 3; do apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/* && break || sleep 10; done

# Copy requirements and install Python dependencies with longer timeout
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=300 --retries=10 -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p documents cache

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "run.py"]
