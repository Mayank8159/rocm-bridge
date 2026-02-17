# 1. Use a robust Python base
FROM python:3.12-slim-bookworm

# 2. Install Clang 17 (Required for your ROCm Bridge)
RUN apt-get update && apt-get install -y \
    wget gnupg lsb-release software-properties-common build-essential curl \
    && wget https://apt.llvm.org/llvm.sh \
    && chmod +x llvm.sh \
    && ./llvm.sh 17 \
    && rm llvm.sh \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. Set environment variables for the compiler
ENV CC=clang-17
ENV CXX=clang++-17
# IMPORTANT: This allows your Python 'clang' library to find the system library
ENV LLVM_LIB_PATH=/usr/lib/x86_64-linux-gnu/libclang.so.1

# 4. Set the working directory
WORKDIR /app

# 5. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy your application code
# This copies everything from your local folder into the /app folder in Docker
COPY . .

# 7. Render uses port 10000 by default
EXPOSE 10000

# 8. Start Streamlit (CORRECTED PATH)
# Since your file is in 'app/main.py', we target that exactly.
CMD ["streamlit", "run", "app/main.py", "--server.port=10000", "--server.address=0.0.0.0"]