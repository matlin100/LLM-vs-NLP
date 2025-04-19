# Use NVIDIA PyTorch container optimized for Hopper architecture
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# Copy project files
COPY . .

# Download spacy model
RUN python -m spacy download en_core_web_sm

# Set environment variables for optimal performance
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV CUDA_LAUNCH_BLOCKING=0
ENV TORCH_CUDA_ARCH_LIST="hopper"
ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV TORCH_ALLOW_TF32=1
ENV TORCH_FLOAT32_MATMUL_PRECISION=high
ENV NCCL_P2P_LEVEL=NVL

# Set Python path
ENV PYTHONPATH=/workspace

# Default command
CMD ["bash"] 