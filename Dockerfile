# Use a Python image that is compatible with C++ tools
FROM python:3.11-slim-bookworm

# Install system dependencies (Clang and build tools)
RUN apt-get update && apt-get install -y \
    clang \
    build-essential \
    libstdc++-12-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY . .

# Expose the port Render will use
EXPOSE 10000

# Start your application (assuming it's a Flask or FastAPI bridge)
CMD ["python", "api/index.py"]