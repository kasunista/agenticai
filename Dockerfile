FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port 8000
EXPOSE 8000

# Set environment variables
ENV PORT=8000

# Add healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8000/_stcore/health || exit 1

# Command to run the application
CMD streamlit run azure_rag_app.py --server.port $PORT --server.address 0.0.0.0 
