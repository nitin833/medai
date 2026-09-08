"""FastAPI routes for MedAI."""

import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.api.schemas import ApproveRequest, ChatRequest, ChatResponse, ThreadStateResponse
from app.config import UPLOAD_DIR
from app.services.agent_service import get_thread_state, resume_agent, run_agent
from app.storage.patient_store import list_patients

app = FastAPI(title="MedAI", description="LangGraph medical assistant API", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = run_agent(req.message, thread_id=req.thread_id)
    return ChatResponse(**result)


@app.post("/upload-image", response_model=ChatResponse)
async def upload_image(
    file: UploadFile = File(...),
    message: str = Form(default="Analyze this medical image"),
    thread_id: str | None = Form(default=None),
    image_type: str | None = Form(default=None),
):
    suffix = Path(file.filename or "image.png").suffix or ".png"
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    with dest.open("wb") as buf:
        shutil.copyfileobj(file.file, buf)

    inputs_extra = {}
    if image_type:
        inputs_extra["image_type"] = image_type

    result = run_agent(
        message,
        thread_id=thread_id,
        image_path=str(dest),
        image_type=image_type,
    )
    if image_type and result.get("status") == "awaiting_approval":
        result.setdefault("partial_state", {})["image_type"] = image_type

    return ChatResponse(**result)


@app.post("/approve/{thread_id}", response_model=ChatResponse)
def approve(thread_id: str, req: ApproveRequest):
    try:
        result = resume_agent(thread_id, action=req.action, note=req.note)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Thread not found or resume failed: {exc}")
    return ChatResponse(**result)


@app.get("/thread/{thread_id}", response_model=ThreadStateResponse)
def thread_state(thread_id: str):
    try:
        state = get_thread_state(thread_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return ThreadStateResponse(**state)


@app.get("/patients")
def patients(limit: int = 50):
    return {"patients": list_patients(limit=limit)}
