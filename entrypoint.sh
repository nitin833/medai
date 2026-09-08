#!/bin/sh
set -e

# Ensure runtime directories exist
mkdir -p /app/data/uploads /app/data

MODE="${1:-streamlit}"

case "$MODE" in
    streamlit)
        echo "Starting MedAI Streamlit Dashboard on port 8501..."
        exec streamlit run app/streamlit_app.py \
            --server.port=8501 \
            --server.address=0.0.0.0 \
            --server.headless=true \
            --browser.gatherUsageStats=false
        ;;
    api)
        echo "Starting MedAI FastAPI REST Service on port 8000..."
        exec uvicorn app.api.routes:app \
            --host 0.0.0.0 \
            --port 8000
        ;;
    both)
        echo "Starting MedAI FastAPI (port 8000) and Streamlit (port 8501)..."
        uvicorn app.api.routes:app --host 0.0.0.0 --port 8000 &
        exec streamlit run app/streamlit_app.py \
            --server.port=8501 \
            --server.address=0.0.0.0 \
            --server.headless=true \
            --browser.gatherUsageStats=false
        ;;
    cli)
        echo "Starting MedAI CLI Agent..."
        exec python -m app.main
        ;;
    *)
        echo "Executing custom command: $@"
        exec "$@"
        ;;
esac
