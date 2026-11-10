#!/usr/bin/env bash
cd "$(dirname "$0")"
pip install -q -r backend/requirements.txt
exec uvicorn backend.main:app --reload --port 8000
