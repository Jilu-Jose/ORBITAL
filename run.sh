#!/usr/bin/env bash
# Starts the ORBITAL API (which also serves the frontend) at http://localhost:8000
# Usage:  ./run.sh                 # demo mode until you train a model
#         USE_SPARK_MODEL=1 ./run.sh   # use the trained Spark model for /api/predict
cd "$(dirname "$0")"
exec uvicorn backend.app.main:app --reload --port "${PORT:-8000}"
