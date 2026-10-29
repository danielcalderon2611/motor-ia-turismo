FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends --yes ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt /app/requirements-api.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/requirements-api.txt

RUN mkdir -p /app/models /app/fewshot_db /app/temp_uploads \
    && useradd --system --create-home --uid 10001 appuser

COPY api.py /app/api.py
COPY models/modelo_mobilenetv2_limpio_final.keras /app/models/
COPY fewshot_db/embeddings_v2.pkl /app/fewshot_db/
COPY lugares_turisticos_app_final_v4.xlsx /app/

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)" || exit 1

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
