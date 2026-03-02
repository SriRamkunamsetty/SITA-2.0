FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements
COPY huggingface_requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r huggingface_requirements.txt

# Copy source code
COPY backend/ ./backend/
COPY sita_core.py .
COPY huggingface_app.py .
# If you have custom weights, ensure they are copied or downloaded
COPY yolov8n.pt .

# Expose Hugging Face default port
EXPOSE 7860

# Run FastAPI via Uvicorn on 0.0.0.0:7860
CMD ["uvicorn", "huggingface_app:app", "--host", "0.0.0.0", "--port", "7860"]
