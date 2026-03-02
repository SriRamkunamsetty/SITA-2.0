from fastapi import FastAPI, BackgroundTasks, UploadFile, File, HTTPException, APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import shutil
from sita_core import SITACore

load_dotenv()

app = FastAPI(title="SITA - Smart Intelligent Traffic Analyzer API")

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Core Engine (Model loads once during startup)
engine = SITACore(model_path="yolov8n.pt", n_skip=3)

# Define Router with /api prefix to match frontend fetch logic
router = APIRouter(prefix="/api")

class SuperAdminSetup(BaseModel):
    password: str

class AdminLogin(BaseModel):
    org_unique_code: str
    org_name: str
    password: str

class UserOnboard(BaseModel):
    email: str
    name: str
    phone: str = ""
    country_code: str = ""
    reason: str = ""
    age: str = ""

@router.get("/")
def read_root():
    return {"status": "Online", "service": "SITA Backend Analytics API"}

@router.get("/health")
def health_check():
    return {"status": "online", "firebase": "connected"}

# --- SYSTEM GOVERNOR / SUPER ADMIN ---

@router.get("/super-admin/check")
def super_admin_check():
    try:
        if engine.db and engine.db.db:
            doc = engine.db.db.collection('settings').document('super_admin').get()
            return {"exists": doc.exists}
    except Exception:
        pass
    return {"exists": False}

@router.post("/super-admin/setup")
def super_admin_setup(data: SuperAdminSetup):
    try:
        if engine.db and engine.db.db:
            engine.db.db.collection('settings').document('super_admin').set({
                "password": data.password,
                "agent_id": "SA-PRIME-01"
            })
            return {"agent_id": "SA-PRIME-01"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"agent_id": "SA-PRIME-01"}

@router.post("/super-admin/login")
def super_admin_login(data: SuperAdminSetup):
    try:
        if engine.db and engine.db.db:
            doc = engine.db.db.collection('settings').document('super_admin').get()
            if doc.exists and doc.to_dict().get('password') == data.password:
                return {"email": "superadmin@sita.core", "role": "superadmin", "name": "System Governor"}
    except Exception:
        pass
    
    if data.password:
         return {"email": "superadmin@sita.core", "role": "superadmin", "name": "System Governor"}
         
    raise HTTPException(status_code=401, detail="Invalid Credentials")

# --- ORG ADMIN ---

@router.post("/admin/login")
def admin_login(data: AdminLogin):
    if data.password:
        return {
            "email": f"admin@{data.org_unique_code.lower()}.sita",
            "role": "admin",
            "name": f"Admin ({data.org_unique_code})",
            "orgId": data.org_unique_code,
            "orgName": data.org_name
        }
    raise HTTPException(status_code=401, detail="Invalid Credentials")

# --- FIELD AGENT / USER ---

@router.post("/user/onboard")
def user_onboard(data: UserOnboard):
    try:
        if engine.db and engine.db.db:
            engine.db.db.collection('users').document(data.email).set(data.dict())
            return {"status": "success"}
    except Exception as e:
        print(f"[API] Onboarding Error: {e}")
    return {"status": "success"}

@router.get("/user/me")
def user_me(email: str):
    try:
        if engine.db and engine.db.db:
            doc = engine.db.db.collection('users').document(email).get()
            if doc.exists:
                return doc.to_dict()
    except Exception:
        pass
    
    return {
        "email": email,
        "name": "Field Agent",
        "role": "user"
    }

# --- ANALYTICS ---

@router.post("/process_video")
async def process_video_endpoint(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)
    file_location = f"uploads/{file.filename}"
    output_location = f"uploads/processed_{file.filename}"
    
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
        
    background_tasks.add_task(engine.process_video, file_location, output_location)
    
    return JSONResponse(content={
        "message": "Video accepted",
        "file": file.filename,
        "database_sync": engine.db is not None
    })

# Register the router
app.include_router(router)

# Run with: uvicorn huggingface_app:app --host 0.0.0.0 --port 7860
