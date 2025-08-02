# RunPod Serverless Dockerfile for Vocal Removal - SIMPLE VERSION
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first
RUN pip install --no-cache-dir \
    runpod>=1.6.0 \
    demucs \
    librosa \
    soundfile

# Copy requirements and handler
COPY requirements.txt .
COPY handler.py .

# Create temp directory
RUN mkdir -p /tmp/vocal_removal

# Set environment variables
ENV PYTHONPATH=/app
ENV RUNPOD_WORKER_TYPE=SERVERLESS

# Run the handler
CMD ["python", "handler.py"]
