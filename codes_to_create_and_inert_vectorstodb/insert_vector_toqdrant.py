import json
import uuid
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# ============================================
# Configuration
# ============================================

COLLECTION_NAME = "medical"

EMBEDDINGS_FILE = "embeddings.npy"
METADATA_FILE = "metadata.json"

# ============================================
# Connect to Qdrant
# ============================================

client = QdrantClient(path="./qdrant_db")

# ============================================
# Load data
# ============================================

vectors = np.load(EMBEDDINGS_FILE)

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# ============================================
# Create points
# ============================================

points = []

for vector, meta in zip(vectors, metadata):

    # Create a deterministic UUID
    unique_string = (
        meta.get("filename", "")
        + meta.get("text", "")
    )

    point_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_DNS,
            unique_string
        )
    )

    points.append(
        PointStruct(
            id=point_id,
            vector=vector.tolist(),
            payload=meta
        )
    )

# ============================================
# Upload
# ============================================

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

client.close()

print(f"Finished uploading {len(points)} documents.")