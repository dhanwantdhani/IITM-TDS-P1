# Use Python 3.11 as base image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    nodejs \
    npm \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install uv and faker
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
# Install prettier globally
RUN npm install -g prettier@3.4.2

# Copy the rest of the application
COPY . .

# Create data directory
RUN mkdir -p data/logs data/docs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV AIPROXY_TOKEN=""
# Add uv to PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 