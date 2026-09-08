"""Image classifier — runs HF vision models."""

import logging

from app.graph.state import MedAIState
from app.models.image_inference import classify_image
from app.models.image_models import get_models_for_type

logger = logging.getLogger(__name__)


def image_classifier_node(state: MedAIState) -> dict:
    image_path = state.get("image_path") or ""
    image_type = state.get("image_type", "chest_xray")
    models = get_models_for_type(image_type)

    if not image_path:
        return {
            "detected_disease": None,
            "confidence": 0.0,
            "model_used": None,
            "safety_flags": ["missing_image_path"],
        }

    if not models:
        return {
            "detected_disease": None,
            "confidence": 0.0,
            "model_used": None,
            "safety_flags": ["unknown_image_type"],
        }

    try:
        result = classify_image(image_path, image_type)
        flags = []
        if result["confidence"] < 0.7:
            flags.append("low_confidence_diagnosis")
        if not result["detected_disease"]:
            flags.append("no_disease_detected")

        return {
            "detected_disease": result["detected_disease"],
            "confidence": result["confidence"],
            "model_used": result["model_used"],
            "classification_details": result["classification_details"],
            "safety_flags": flags,
            "hitl_pending": False,
        }
    except FileNotFoundError:
        return {
            "detected_disease": None,
            "confidence": 0.0,
            "model_used": None,
            "safety_flags": [f"image_not_found:{image_path}"],
        }
    except Exception as exc:
        logger.exception("Image classification failed")
        return {
            "detected_disease": None,
            "confidence": 0.0,
            "model_used": None,
            "safety_flags": [f"classification_error:{exc}"],
        }
