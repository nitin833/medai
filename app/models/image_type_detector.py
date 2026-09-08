"""Detect medical image type from user text and file path."""

import re
from pathlib import Path

from app.models.image_models import IMAGE_MODELS

_TYPE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("brain_mri", re.compile(r"\b(mri|brain|tumor|glioma|meningioma)\b", re.I)),
    ("retina", re.compile(r"\b(retina|retinal|fundus|ophthalm|diabetic.?retinopathy|eye)\b", re.I)),
    ("skin", re.compile(r"\b(skin|derma|melanoma|eczema|psoriasis|rash|lesion)\b", re.I)),
    ("bone_xray", re.compile(r"\b(bone|fracture|orthop|skeletal)\b", re.I)),
    ("chest_xray", re.compile(r"\b(chest|xray|x-ray|cxr|lung|pneumonia|tb|tuberculosis|chexpert)\b", re.I)),
]


def detect_image_type(text: str = "", image_path: str = "") -> str:
    combined = f"{text} {Path(image_path).name} {image_path}".strip()
    scores: dict[str, int] = {t: 0 for t in IMAGE_MODELS}

    for image_type, pattern in _TYPE_PATTERNS:
        if pattern.search(combined):
            scores[image_type] += 2

    path_lower = combined.lower()
    if any(k in path_lower for k in ("mri", "brain")):
        scores["brain_mri"] += 1
    if any(k in path_lower for k in ("retina", "fundus", "eye")):
        scores["retina"] += 1
    if any(k in path_lower for k in ("skin", "derma")):
        scores["skin"] += 1
    if any(k in path_lower for k in ("bone", "fracture")):
        scores["bone_xray"] += 1
    if any(k in path_lower for k in ("chest", "cxr", "xray", "x-ray", "lung")):
        scores["chest_xray"] += 1

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "chest_xray"
