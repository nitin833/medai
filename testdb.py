import os
import time
from dotenv import load_dotenv

from huggingface_hub import HfApi, InferenceClient
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# ----------------------------------------------------
# Load Environment Variables
# ----------------------------------------------------

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise Exception("HF_TOKEN not found!")

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

MODEL = "Intelligent-Internet/II-Medical-8B"

COLLECTION_1 = "medical"
COLLECTION_2 = "merck_manual"      # Change if different

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_db")

# ----------------------------------------------------
# Embedding Model
# ----------------------------------------------------

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)

# ----------------------------------------------------
# Qdrant
# ----------------------------------------------------

if QDRANT_URL:
    client = QdrantClient(url=QDRANT_URL)
else:
    client = QdrantClient(path=QDRANT_PATH)

vectorstore1 = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_1,
    embedding=embeddings,
)

vectorstore2 = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_2,
    embedding=embeddings,
)

retriever1 = vectorstore1.as_retriever(
    search_kwargs={"k": 1}
)

retriever2 = vectorstore2.as_retriever(
    search_kwargs={"k": 1}
)

# ----------------------------------------------------
# Hugging Face Provider
# ----------------------------------------------------

api = HfApi()

info = api.model_info(
    MODEL,
    expand=["inferenceProviderMapping"]
)

if not info.inference_provider_mapping:
    raise Exception("No provider found!")

provider = info.inference_provider_mapping[0].provider

print(f"Using Provider: {provider}")

llm = InferenceClient(
    provider=provider,
    api_key=HF_TOKEN,
)

# ----------------------------------------------------
# Chat Loop
# ----------------------------------------------------

while True:

    question = input("\nQuestion (type exit to quit): ")

    if question.lower() == "exit":
        break

    # ---------------- Retrieval ----------------

    t1 = time.time()

    docs1 = retriever1.invoke(question)
    docs2 = retriever2.invoke(question)

    retrieval_time = time.time() - t1

    docs = docs1 + docs2

    # ---------------- Context ----------------

    context = ""

    for i, doc in enumerate(docs, 1):

        context += f"""
Document {i}

Metadata:
{doc.metadata}

Content:
{doc.page_content}

------------------------------------------
"""

    prompt = f"""
You are an expert biomedical assistant.

Answer ONLY using the retrieved context.

If the answer cannot be found,
reply:

"I don't have enough information in the retrieved documents."

Retrieved Context:

{context}

Question:
{question}

Answer:
"""

    print(f"\nRetrieved {len(docs)} document(s)")
    print(f"Prompt Length: {len(prompt)} characters")

    # ---------------- LLM ----------------

    t2 = time.time()

    response = llm.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an expert biomedical assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=300,
        temperature=0.2,
    )

    generation_time = time.time() - t2

    # ---------------- Output ----------------

    print("\n================ ANSWER ================\n")
    print(response.choices[0].message.content)

    print("\n================ TIMINGS ================\n")
    print(f"Retrieval Time : {retrieval_time:.2f} sec")
    print(f"Generation Time: {generation_time:.2f} sec")
    print(f"Total Time     : {retrieval_time + generation_time:.2f} sec")

    print("\n================ DOCUMENTS ================\n")

    for i, doc in enumerate(docs, 1):
        print(f"\nDocument {i}")
        print("-" * 70)
        print(doc.page_content[:700])