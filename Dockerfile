# ==========================================
# MedAI Dockerfile
# Multi-modal Clinical Assistant & RAG
# ==========================================

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies (build tools, libraries for OpenCV/Pillow, curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements file first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and runtime assets
COPY app/ ./app/
COPY DrugCentral_KnowledgeBase.txt ./
COPY testdb.py ./
COPY entrypoint.sh ./

# Make entrypoint script executable and create runtime directories
RUN chmod +x /app/entrypoint.sh && \
    mkdir -p /app/data/uploads /app/data

# Expose Streamlit (8501) and FastAPI (8000) ports
EXPOSE 8501
EXPOSE 8000

# Set entrypoint and default command
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["streamlit"]
