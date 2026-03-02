---
title: SITA-Backend
sdk: docker
app_port: 7860
base_path: /docs
---

# SITA (Smart Intelligent Traffic Analyzer) - Hugging Face Space

This container runs the backend API for SITA using FastAPI.
It leverages YOLOv8n (optimized via ONNXRuntime) and EasyOCR to analyze video feeds, sync data directly to Firebase, and serve progress to the frontend.
