#!/bin/sh

set -e

echo "Récupération des événements OpenAgenda..."
python scripts/fetch_openagenda_events.py

echo "Construction de l'index FAISS..."
python scripts/build_faiss_index.py

echo "Démarrage de l'API FastAPI..."
exec uvicorn app.api:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}"