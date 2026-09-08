"""Human-in-the-loop interrupt handlers."""

from langgraph.types import interrupt

from app.graph.state import MedAIState
from app.models.image_models import IMAGE_MODELS


def image_hitl_node(state: MedAIState) -> dict:
    feedback = interrupt({
        "type": "image_confirmation",
        "message": f"Detected image type: {state.get('image_type')}. Proceed with analysis?",
        "image_path": state.get("image_path"),
        "available_types": list(IMAGE_MODELS.keys()),
        "options": ["approve", "reject", "change_type"],
    })

    action = feedback.get("action", "approve") if isinstance(feedback, dict) else "approve"
    note = feedback.get("note") if isinstance(feedback, dict) else None

    updates: dict = {
        "hitl_action": action,
        "human_feedback": note,
        "hitl_pending": action == "reject",
    }

    if action == "change_type" and note and note in IMAGE_MODELS:
        updates["image_type"] = note

    return updates


def info_hitl_node(state: MedAIState) -> dict:
    from app.storage.patient_store import save_patient

    feedback = interrupt({
        "type": "save_confirmation",
        "message": "Save this patient information?",
        "data": state.get("patient_info"),
        "options": ["approve", "reject", "edit"],
    })

    action = feedback.get("action", "approve") if isinstance(feedback, dict) else "approve"
    note = feedback.get("note") if isinstance(feedback, dict) else None

    updates: dict = {
        "hitl_action": action,
        "human_feedback": note,
        "hitl_pending": False,
    }

    if action == "edit" and note:
        info = dict(state.get("patient_info") or {})
        info["clinician_notes"] = note
        updates["patient_info"] = info

    if action == "approve":
        patient_info = updates.get("patient_info") or state.get("patient_info") or {}
        if patient_info:
            patient_id = save_patient(patient_info)
            updates["patient_info"] = {**patient_info, "saved_id": patient_id}

    return updates


def safety_hitl_node(state: MedAIState) -> dict:
    feedback = interrupt({
        "type": "clinical_review",
        "message": "High-risk response requires clinician review.",
        "detected_disease": state.get("detected_disease"),
        "prescription_draft": state.get("prescription_draft"),
        "risk_flags": state.get("safety_flags"),
        "options": ["approve", "reject", "edit"],
    })

    action = feedback.get("action", "approve") if isinstance(feedback, dict) else "approve"
    note = feedback.get("note") if isinstance(feedback, dict) else None

    updates: dict = {
        "hitl_action": action,
        "human_feedback": note,
        "hitl_pending": False,
    }

    if action == "reject":
        updates["prescription_draft"] = None
        updates["final_response"] = (
            "Response withheld pending clinician review. "
            "Please consult a licensed healthcare provider."
        )
    elif action == "edit" and note:
        updates["prescription_draft"] = note

    return updates
