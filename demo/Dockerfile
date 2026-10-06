# Single-process demo: FastAPI serves the static frontend and the /api JSON from the exported frozen results.
# Copies only the demo app and its exported data. No research code, no datasets, no checkpoints.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY demo/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY demo/app /app/app
COPY demo/data /app/data

WORKDIR /app/app
EXPOSE 7860
# Hugging Face Spaces (Docker SDK) expects the app on port 7860.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
