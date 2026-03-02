FROM python:3.10-slim

# Prepare system for OpenCV, EasyOCR, and create user
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

WORKDIR /home/user/app

# Install Python requirements
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy all project files
COPY --chown=user . .

# Run the FastAPI server
EXPOSE 7860
CMD ["uvicorn", "huggingface_app:app", "--host", "0.0.0.0", "--port", "7860"]
