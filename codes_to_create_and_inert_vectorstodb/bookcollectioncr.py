from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient(path="./qdrant_db")

client.create_collection(
    collection_name="merck_manual",
    vectors_config=VectorParams(
        size=768,      # same as your embedding model
        distance=Distance.COSINE
    )
)