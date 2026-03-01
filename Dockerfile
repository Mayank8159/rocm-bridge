# =============================================================================
# ROCm Bridge - Production Docker Image
# =============================================================================
FROM python:3.11-slim-bookworm

# Prevent Python bytecode caching
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    gnupg \
    lsb-release \
    software-properties-common \
    clang-17 \
    libclang-17-dev \
    llvm-17-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV CC=clang-17
ENV CXX=clang++-17
ENV LLVM_LIB_PATH=/usr/lib/x86_64-linux-gnu/libclang.so.17
ENV PATH="/usr/lib/llvm-17/bin:${PATH}"

# Set working directory
WORKDIR /app

# Install Python dependencies (before copying code for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create output directories
RUN mkdir -p /app/output/hip_output \
    /app/output/patches \
    /app/output/reports \
    /app/output/logs \
    /app/profiles

# Set permissions
RUN chmod -R 755 /app

# Expose port
EXPOSE 10000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:10000/_stcore/health || exit 1

# Start application
CMD ["streamlit", "run", "app/main.py", \
     "--server.port=10000", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]