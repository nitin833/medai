import re
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from tqdm import tqdm

# ==========================
# Configuration
# ==========================

TEXT_FILE = "DrugCentral_KnowledgeBase.txt"

COLLECTION_NAME = "medical"

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"   # 768 dimensions

BATCH_SIZE = 100

# ==========================
# Load embedding model
# ==========================

print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL)

# ==========================
# Connect to Qdrant
# ==========================

client = QdrantClient(path="./qdrant_db")

# ==========================
# Read file
# ==========================

with open(TEXT_FILE, "r", encoding="utf-8") as f:
    text = f.read()

documents = [
    doc.strip()
    for doc in text.split("=" * 80)
    if doc.strip()
]

print(f"Found {len(documents)} drug documents")

# ==========================
# Helper functions
# ==========================

def extract_field(name, text):

    m = re.search(rf"{name}:\s*(.*)", text)

    if m:
        return m.group(1).strip()

    return ""


def extract_diseases(text):

    diseases = []

    if "Diseases:" not in text:
        return diseases

    section = text.split("Diseases:")[1]

    for line in section.splitlines():

        line = line.strip()

        if line.startswith("-"):

            diseases.append(line[1:].strip())

    return diseases


# ==========================
# Create points
# ==========================

points = []

point_id = 0

for doc in tqdm(documents):

    drug = extract_field("Drug", doc)

    formula = extract_field("Formula", doc)

    weight = extract_field("Molecular Weight", doc)

    smiles = extract_field("SMILES", doc)

    inchi = extract_field("InChI", doc)

    diseases = extract_diseases(doc)

    embedding = model.encode(
        doc,
        normalize_embeddings=True
    ).tolist()

    payload = {

        "type": "drug",

        "drug": drug,

        "formula": formula,

        "molecular_weight": weight,

        "smiles": smiles,

        "inchi": inchi,

        "diseases": diseases,

        "source": "DrugCentral",

        "text": doc
    }

    points.append(

        PointStruct(
            id=point_id,
            vector=embedding,
            payload=payload
        )

    )

    point_id += 1

# ==========================
# Upload
# ==========================

print("Uploading...")

for i in tqdm(range(0, len(points), BATCH_SIZE)):

    batch = points[i:i+BATCH_SIZE]

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=batch
    )

print()

print("Done!")

print("Uploaded:", len(points), "documents")