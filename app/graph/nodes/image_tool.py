"""Image tool — prepares image for classification."""

from pathlib import Path

from app.graph.state import MedAIState
from app.models.image_type_detector import detect_image_type


def image_tool_node(state: MedAIState) -> dict:
    image_path = state.get("image_path") or ""

    user_text = ""
    for msg in reversed(state.get("messages", [])):
        content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")
        if content and not content.startswith("[router]"):
            user_text = content
            break

    image_type = state.get("image_type") or detect_image_type(user_text, image_path)

    if image_path and not Path(image_path).is_file():
        return {
            "image_path": image_path,
            "image_type": image_type,
            "safety_flags": [f"image_not_found:{image_path}"],
            "hitl_pending": False,
        }

    return {
        "image_path": image_path,
        "image_type": image_type,
        "hitl_pending": True,
    }
