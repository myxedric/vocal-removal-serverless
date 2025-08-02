# RunPod Serverless Dockerfile for Vocal Removal
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Demucs and audio processing libraries
RUN pip install --no-cache-dir \
    demucs \
    librosa \
    soundfile \
    runpod

# Create temp directory
RUN mkdir -p /tmp/vocal_removal

# Copy handler
COPY handler.py .

# Pre-download models to reduce cold start time
RUN python3 -c "\
from demucs.pretrained import get_model; \
import torch; \
print('Pre-downloading models...'); \
try: \
    model = get_model('htdemucs_ft'); \
    print('Downloaded htdemucs_ft'); \
    del model; \
    torch.cuda.empty_cache(); \
except: \
    pass; \
try: \
    model = get_model('htdemucs'); \
    print('Downloaded htdemucs'); \
    del model; \
    torch.cuda.empty_cache(); \
except: \
    pass; \
print('Model pre-download complete')"

# Set environment variables
ENV PYTHONPATH=/app
ENV RUNPOD_WORKER_TYPE=SERVERLESS

# Run the handler
CMD ["python", "handler.py"]
