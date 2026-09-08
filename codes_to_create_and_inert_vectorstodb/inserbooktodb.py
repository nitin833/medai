import json
import uuid
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

client = QdrantClient(path="./qdrant_db")

embeddings = np.load("embeddings.npy")

metadata = []

with open("metadata.jsonl","r",encoding="utf-8") as f:
    for line in f:
        metadata.append(json.loads(line))

batch = []

for vector, payload in zip(embeddings, metadata):

    batch.append(
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector.tolist(),
            payload=payload
        )
    )

    if len(batch)==128:

        client.upsert(
            collection_name="merck_manual",
            points=batch
        )

        batch=[]

if batch:
    client.upsert(
        collection_name="merck_manual",
        points=batch
    )

print("Finished")