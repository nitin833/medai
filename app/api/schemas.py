"""Pydantic schemas for the MedAI API."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ChatResponse(BaseModel):
    thread_id: str
    status: str
    response: str | None = None
    interrupt: dict | None = None
    detected_disease: str | None = None
    confidence: float | None = None
    prescription_draft: str | None = None
    risk_level: str | None = None
    partial_state: dict | None = None


class ApproveRequest(BaseModel):
    action: str = Field(default="approve", pattern="^(approve|reject|edit|change_type)$")
    note: str | None = None


class ThreadStateResponse(BaseModel):
    thread_id: str
    messages: list[dict]
    detected_disease: str | None = None
    confidence: float | None = None
    prescription_draft: str | None = None
    risk_level: str | None = None
    final_response: str | None = None
