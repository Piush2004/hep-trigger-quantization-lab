# Single-process demo: FastAPI serves the static frontend and the /api JSON from the exported frozen results.
# Copies only the demo app and its exported data. No research code, no datasets, no checkpoints, no .git.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY demo/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY demo/app /app/app
COPY demo/data /app/data

# Hugging Face Spaces (and most container platforms) run the image as UID 1000; create that user
# explicitly and own the app directory, rather than running as root.
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

WORKDIR /app/app
EXPOSE 7860
# Hugging Face Spaces (Docker SDK) requires the app on port 7860, bound to 0.0.0.0 (see demo/README.md
# for the README.md front matter this pairs with: sdk: docker, app_port: 7860).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
