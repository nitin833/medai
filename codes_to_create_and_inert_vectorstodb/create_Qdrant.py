from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from qdrant_client.models import Distance
from qdrant_client.models import VectorParams

client=QdrantClient(path="./qdrant_db")

client.recreate_collection(

    collection_name="medical",

    vectors_config=VectorParams(

        size=768,

        distance=Distance.COSINE

    )
)