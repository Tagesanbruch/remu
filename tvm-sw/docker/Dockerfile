# TVM Docker Environment for REMU NPU Compiler
# Uses Python 3.11 with Apache TVM for ARM64 Linux

FROM python:3.11-slim-bookworm

LABEL maintainer="REMU NPU Team"
LABEL description="TVM compiler environment for REMU NPU backend"

# Use USTC mirrors for pip (Python packages)
ENV PIP_INDEX_URL=https://pypi.mirrors.ustc.edu.cn/simple
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TVM_HOME=/opt/tvm
ENV PYTHONPATH=/opt/tvm/python:$PYTHONPATH

# Replace apt sources with USTC mirror (Debian bookworm)
RUN cat > /etc/apt/sources.list <<'EOF'
deb https://mirrors.ustc.edu.cn/debian/ bookworm main contrib non-free
deb-src https://mirrors.ustc.edu.cn/debian/ bookworm main contrib non-free
deb https://mirrors.ustc.edu.cn/debian-security bookworm-security main contrib non-free
deb-src https://mirrors.ustc.edu.cn/debian-security bookworm-security main contrib non-free
deb https://mirrors.ustc.edu.cn/debian/ bookworm-updates main contrib non-free
deb-src https://mirrors.ustc.edu.cn/debian/ bookworm-updates main contrib non-free
EOF

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    ca-certificates \
    apt-transport-https \
    libgomp1 \
    gfortran \
    pkg-config  \
    meson \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment directory (will be mounted from host)
RUN mkdir -p /workspace /venv

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple

RUN pip install uv

# Install Apache TVM and dependencies
# Version compatibility matrix:
#   - apache-tvm 0.12.0 requires numpy<2
#   - onnx 1.15.x requires ml_dtypes>=0.3,<0.5
#   - onnx 1.16+ requires ml_dtypes>=0.4 with new float4/uint4 types
# We use onnx 1.15.0 for maximum compatibility
RUN uv pip install --index-url https://pypi.mirrors.ustc.edu.cn/simple \
    --system --no-cache-dir \
    "numpy>=1.23,<2" \
    "ml_dtypes>=0.3,<0.4" \
    "protobuf>=3.20,<5" \
    "onnx==1.15.0" \
    "onnxruntime>=1.16,<1.18" \
    "scipy>=1.10,<1.14" \
    "typing_extensions>=4.0" \
    "pillow>=9.0"

# Install TVM separately to avoid version conflicts
RUN uv pip install --index-url https://pypi.mirrors.ustc.edu.cn/simple \
    --system --no-cache-dir \
    apache-tvm==0.12.0

# Verify installations
RUN python -c "import tvm; print('TVM version:', tvm.__version__); \
    import onnx; print('ONNX version:', onnx.__version__); \
    from tvm import relay; print('Relay OK')"

# Set working directory
WORKDIR /workspace

# Default command
CMD ["python", "--version"]