"""Hugging Face image classification inference."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from app.config import HF_TOKEN
from app.models.image_models import get_models_for_type

logger = logging.getLogger(__name__)

_pipeline_cache: dict[str, object] = {}

POSITIVE_LABEL_HINTS = (
    "pneumonia", "tuberculosis", "tb", "tumor", "cancer", "disease",
    "fracture", "abnormal", "positive", "retinopathy", "malignant",
    "benign", "opacity", "infiltrate", "pathology", "yes",
)

NEGATIVE_LABEL_HINTS = (
    "normal", "healthy", "no", "negative", "no_fracture", "no fracture",
    "notumor", "no_tumor", "clear",
)

DISEASE_DISPLAY: dict[str, str] = {
    "pneumonia": "Pneumonia",
    "tuberculosis": "Tuberculosis",
    "chest_abnormal": "Chest Abnormality",
    "brain_tumor": "Brain Tumor",
    "skin_disease": "Skin Disease",
    "diabetic_retinopathy": "Diabetic Retinopathy",
    "fracture": "Bone Fracture",
}


@dataclass
class ClassificationResult:
    disease_key: str
    disease_name: str
    confidence: float
    model_id: str
    label: str
    all_scores: list[dict]


def _load_pipeline(model_id: str):
    if model_id in _pipeline_cache:
        return _pipeline_cache[model_id]

    from transformers import pipeline

    kwargs: dict = {"model": model_id}
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN

    pipe = pipeline("image-classification", **kwargs)
    _pipeline_cache[model_id] = pipe
    return pipe


def _is_positive_label(label: str) -> bool:
    lower = label.lower().replace("-", "_").replace(" ", "_")
    if any(h in lower for h in NEGATIVE_LABEL_HINTS):
        return False
    if any(h in lower for h in POSITIVE_LABEL_HINTS):
        return True
    return lower not in ("normal", "healthy", "negative", "no")


def _run_single_model(image: Image.Image, disease_key: str, model_id: str) -> ClassificationResult | None:
    try:
        pipe = _load_pipeline(model_id)
        raw = pipe(image)
        if not raw:
            return None

        best = max(raw, key=lambda x: x["score"])
        label = best["label"]
        score = float(best["score"])

        if not _is_positive_label(label):
            return ClassificationResult(
                disease_key=disease_key,
                disease_name=DISEASE_DISPLAY.get(disease_key, disease_key.replace("_", " ").title()),
                confidence=round(1.0 - score, 4) if score > 0.5 else score,
                model_id=model_id,
                label=label,
                all_scores=raw,
            )

        return ClassificationResult(
            disease_key=disease_key,
            disease_name=DISEASE_DISPLAY.get(disease_key, disease_key.replace("_", " ").title()),
            confidence=round(score, 4),
            model_id=model_id,
            label=label,
            all_scores=raw,
        )
    except Exception as exc:
        logger.warning("Model %s failed: %s", model_id, exc)
        return None


def classify_image(image_path: str, image_type: str) -> dict:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(path).convert("RGB")
    models = get_models_for_type(image_type)

    if not models:
        return {
            "detected_disease": None,
            "confidence": 0.0,
            "model_used": None,
            "classification_details": [],
        }

    results: list[ClassificationResult] = []
    for disease_key, model_id in models.items():
        result = _run_single_model(image, disease_key, model_id)
        if result:
            results.append(result)

    if not results:
        return {
            "detected_disease": None,
            "confidence": 0.0,
            "model_used": None,
            "classification_details": [],
        }

    best = max(results, key=lambda r: r.confidence)
    details = [
        {
            "disease_key": r.disease_key,
            "disease_name": r.disease_name,
            "confidence": r.confidence,
            "label": r.label,
            "model_id": r.model_id,
        }
        for r in results
    ]

    return {
        "detected_disease": best.disease_name if best.confidence >= 0.5 else None,
        "confidence": best.confidence,
        "model_used": best.model_id,
        "classification_details": details,
    }
