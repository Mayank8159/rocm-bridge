# 1. Use a robust Python base
FROM python:3.12-slim-bookworm

# 2. Install Clang 17 (Required for your ROCm Bridge)
RUN apt-get update && apt-get install -y \
    wget gnupg lsb-release software-properties-common build-essential \
    && wget https://apt.llvm.org/llvm.sh \
    && chmod +x llvm.sh \
    && ./llvm.sh 17 \
    && rm llvm.sh \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. Set environment variables for the compiler
ENV CC=clang-17
ENV CXX=clang++-17

# 4. Set the working directory (Streamlit prefers not running from /)
WORKDIR /app

# 5. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy your application code
COPY . .

# 7. Render uses port 10000 by default
EXPOSE 10000

# 8. Start Streamlit (Update 'api/index.py' if you rename the file!)
CMD ["streamlit", "run", "api/index.py", "--server.port=10000", "--server.address=0.0.0.0"]