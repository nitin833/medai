"""Hugging Face image model registry."""

IMAGE_MODELS: dict[str, dict[str, str]] = {
    "chest_xray": {
        "pneumonia": "ianpan/pneumonia-cxr",
        "tuberculosis": "sukhmani1303/tuberculosis-vit-model",
        "chest_abnormal": "itsomk/chexpert-densenet121",
    },
    "brain_mri": {
        "brain_tumor": "Raghava-Ram/brain-tumor-efficientnet",
    },
    "skin": {
        "skin_disease": "HotJellyBean/skin-disease-classifier",
    },
    "retina": {
        "diabetic_retinopathy": "Aditya-Sai-19/diabetic-retinopathy-swin",
    },
    "bone_xray": {
        "fracture": "Hemg/bone-fracture-detection-using-x-rays",
    },
}


def get_models_for_type(image_type: str) -> dict[str, str]:
    return IMAGE_MODELS.get(image_type, {})
