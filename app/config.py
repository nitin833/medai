import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HF_TOKEN = os.getenv("HF_TOKEN")
LLM_MODEL = "Intelligent-Internet/II-Medical-8B"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_PATH = os.getenv("QDRANT_PATH", str(PROJECT_ROOT / "qdrant_db"))
COLLECTION_MEDICAL = "medical"
COLLECTION_MERCK = "merck_manual"

RAG_TOP_K = 3

CHECKPOINT_DB = str(PROJECT_ROOT / "data" / "checkpoints.db")
PATIENT_DB = str(PROJECT_ROOT / "data" / "patients.db")
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"

