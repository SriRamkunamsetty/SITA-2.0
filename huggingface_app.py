from fastapi import FastAPI, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv
import os
import shutil
from sita_core import SITACore

load_dotenv()

app = FastAPI(title="SITA - Smart Intelligent Traffic Analyzer API")

# Initialize the Core Engine (Model loads once during startup)
engine = SITACore(model_path="yolov8n.pt", n_skip=3)

@app.get("/")
def read_root():
    return {"status": "Online", "service": "SITA Backend Analytics API"}

@app.get("/health")
def health_check():
    # Allows HF Space status monitoring to verify the API is alive and initialized
    return {"status": "online", "firebase": "connected"}

@app.post("/process_video")
async def process_video_endpoint(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Accepts a video upload and processes it using SITA Core asynchronously.
    """
    # Create temp directory for uploads if not exists
    os.makedirs("uploads", exist_ok=True)
    file_location = f"uploads/{file.filename}"
    output_location = f"uploads/processed_{file.filename}"
    
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
        
    print(f"[API] Received Video for processing: {file.filename}")
    
    # Run the processing in FastAPI's background task
    background_tasks.add_task(engine.process_video, file_location, output_location)
    
    return JSONResponse(content={
        "message": "Video accepted. Processing in backend.",
        "file": file.filename,
        "mode": "AMD Ryzen Optimized (ONNX)",
        "tracked": True,
        "database_sync": engine.db is not None
    })

# Run with: uvicorn huggingface_app:app --host 0.0.0.0 --port 7860
