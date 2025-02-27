#!/bin/bash
if [ "$APP_ENV" = "development" ]; then
    # Development mode with hot reload
    echo "Starting in DEVELOPMENT mode with hot reload"
    exec uvicorn main:app --host 0.0.0.0 --port ${API_PORT:-8000} --reload
else
    # Production mode with multiple workers (based on CPU cores)
    WORKERS=${WORKERS:-$(( 2 * $(nproc) + 1 ))}
    echo "Starting in PRODUCTION mode with $WORKERS workers"
    exec uvicorn main:app --host 0.0.0.0 --port ${API_PORT:-8000} --workers $WORKERS
fi
